//! Publish a new file: temp file beside the destination, preallocate, write,
//! `fdatasync`, `rename`, then `fsync` the directory.

use std::fs::{self, File as FsFile, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Result;

/// Removes the temp file unless the publish reached the rename.
pub(super) struct TempGuard {
    pub(super) path: PathBuf,
    pub(super) armed: bool,
}

impl Drop for TempGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub(super) fn temp_path(dest: &Path) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = parent_dir(dest);
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".into());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(
        ".{name}.{}.{nanos}.{seq}.segyrs-tmp",
        std::process::id()
    ))
}

pub(super) fn parent_dir(dest: &Path) -> PathBuf {
    match dest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Write `len` bytes to `dest` through `body`, atomically.
///
/// The destination is either the previous file or the complete new one. With
/// `sync`, the data and the directory entry are on stable storage on return.
pub(crate) fn publish(
    dest: &Path,
    len: u64,
    sync: bool,
    body: impl FnOnce(&FsFile) -> Result<()>,
) -> Result<()> {
    let tmp = temp_path(dest);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(crate::error::Error::path("write", dest))?;
    let mut guard = TempGuard {
        path: tmp.clone(),
        armed: true,
    };
    preallocate(&file, len)?;
    body(&file)?;
    let written = file.metadata()?.len();
    if written != len {
        return Err(
            io::Error::other(format!("writer produced {written} bytes, expected {len}")).into(),
        );
    }
    if sync {
        file.sync_data()?;
    }
    drop(file);
    fs::rename(&tmp, dest).map_err(crate::error::Error::path("write", dest))?;
    guard.armed = false;
    if sync {
        sync_dir(&parent_dir(dest))?;
    }
    Ok(())
}

/// `fsync` a directory so a rename inside it is durable.
pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        FsFile::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Reserve `len` bytes so later writes cannot fail for lack of space.
///
/// Mode 0 `fallocate` on Linux. A filesystem that does not support it is
/// skipped and the writes grow the file. Disk-full is reported here, before
/// any sample is encoded.
pub(super) fn preallocate(file: &FsFile, len: u64) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let Ok(size) = libc::off_t::try_from(len) else {
            return Ok(());
        };
        // SAFETY: the descriptor is open for writing and owned by `file` for
        // the whole call. fallocate does not touch user memory.
        let rc = unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, size) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            match err.raw_os_error() {
                Some(libc::EOPNOTSUPP) | Some(libc::ENOSYS) | Some(libc::EINVAL) => {}
                _ => return Err(err.into()),
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
    Ok(())
}
