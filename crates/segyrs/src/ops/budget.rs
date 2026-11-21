use std::io;

use crate::error::{Error, Result};

/// Bytes of RAM an operation may keep live for trace buffers and its index.
///
/// There is no compiled-in cap. [`Budget::from_available`] reads what the
/// machine (and its cgroup, when one limits this process) has free at the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    bytes: u64,
}

impl Budget {
    /// An explicit budget. Zero is raised to one byte so chunking still progresses.
    pub fn bytes(bytes: u64) -> Self {
        Self {
            bytes: bytes.max(1),
        }
    }

    /// `fraction` of the RAM available now, for example `0.5`.
    pub fn fraction_of_available(fraction: f64) -> Result<Self> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(crate::ops::invalid(format!(
                "budget fraction {fraction} is outside (0, 1]"
            )));
        }
        Ok(Self::bytes((available_ram()? as f64 * fraction) as u64))
    }

    /// Half of the RAM available now. The other half stays for the page cache
    /// and the caller.
    pub fn from_available() -> Result<Self> {
        Self::fraction_of_available(0.5)
    }

    pub fn get(self) -> u64 {
        self.bytes
    }

    /// `Err` naming both numbers when `need` bytes do not fit.
    pub(crate) fn check(self, need: u128, what: &str) -> Result<()> {
        if need > u128::from(self.bytes) {
            return Err(over_budget(need, self.bytes, what));
        }
        Ok(())
    }
}

/// Over-budget is `io::ErrorKind::OutOfMemory`, raised before the allocation.
pub(crate) fn over_budget(need: u128, budget: u64, what: &str) -> Error {
    Error::Io(io::Error::new(
        io::ErrorKind::OutOfMemory,
        format!("{what} needs {need} bytes, the RAM budget is {budget} bytes"),
    ))
}

/// RAM available to this process now, in bytes.
///
/// Linux: `MemAvailable` from `/proc/meminfo`, lowered to the cgroup v2 headroom
/// (`memory.max` minus `memory.current`) when a limit is set. Windows: the
/// smaller of the available physical memory and the remaining commit charge
/// (`GlobalMemoryStatusEx`). macOS: the installed physical memory
/// (`hw.memsize`), since macOS reports no single figure for reclaimable
/// memory. Other systems ask for [`Budget::bytes`].
pub fn available_ram() -> Result<u64> {
    #[cfg(windows)]
    let bytes = windows_available();
    #[cfg(target_os = "macos")]
    let bytes = macos_available();
    #[cfg(not(any(windows, target_os = "macos")))]
    let bytes = linux_available();
    bytes.ok_or_else(|| {
        Error::Io(io::Error::new(
            io::ErrorKind::Unsupported,
            "available RAM is unknown here, pass an explicit Budget::bytes",
        ))
    })
}

#[cfg(windows)]
fn windows_available() -> Option<u64> {
    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        ..Default::default()
    };
    // SAFETY: `status` is a correctly sized MEMORYSTATUSEX with `dwLength` set, as the call requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    (ok && status.avail_phys > 0).then(|| status.avail_phys.min(status.avail_page_file))
}

#[cfg(target_os = "macos")]
fn macos_available() -> Option<u64> {
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a 64-bit integer and `len` states the size of the buffer it fills.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<u64>() && bytes > 0).then_some(bytes)
}

#[cfg(not(any(windows, target_os = "macos")))]
fn linux_available() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| {
            rest.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()
        })?;
    let mut avail = kib.saturating_mul(1024);
    if let Some(room) = cgroup_headroom() {
        avail = avail.min(room);
    }
    Some(avail)
}

#[cfg(not(any(windows, target_os = "macos")))]
fn cgroup_headroom() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = own.lines().find_map(|l| l.strip_prefix("0::"))?;
    let dir = std::path::Path::new("/sys/fs/cgroup").join(rel.trim_start_matches('/'));
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
    let max = read("memory.max")?.trim().parse::<u64>().ok()?;
    let current = read("memory.current")?.trim().parse::<u64>().ok()?;
    Some(max.saturating_sub(current))
}
