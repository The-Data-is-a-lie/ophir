# Keep-awake + zombie-proof launch wrapper for unattended autoresearch sessions.
#
# Two failure modes are handled (both bit on 2026-07-08/09):
# 1. SLEEP: the box sleeps after 30 min idle (AC STANDBYIDLE=0x708) and the idle
#    session/display transition CTRL_CLOSE-aborts torch/MKL training (same
#    failure OphirRebalance hit in June -- see ophir-bot/rebalance.ps1). Fix:
#    hold ES_CONTINUOUS|ES_SYSTEM|ES_DISPLAY while the session runs.
# 2. ZOMBIES: if this wrapper is killed (task stop, console teardown), plain
#    child processes survive as orphans -- two dead sessions' loops kept
#    running, held the GPU, and threatened stale git resets. Fix: run the
#    session inside a Windows job object with KILL_ON_JOB_CLOSE; when this
#    process dies for ANY reason the OS kills the whole descendant tree.
#
# A preflight check refuses to launch while any autoresearch python is running.
#
# Usage (all arguments are forwarded verbatim to run_loop.sh):
#   powershell -NoProfile -File autoresearch/run_loop_awake.ps1 `
#     --session s3-20260709 --max-iters 3 --epsilon 0.049

$ErrorActionPreference = "Stop"

# --- preflight: never launch over a stale/zombie session ---------------------
$stale = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'autoresearch[\\/](loop|train_experiment|eval_harness)\.py' }
if ($stale) {
    Write-Error ("stale autoresearch python(s) still running: " +
        (($stale | ForEach-Object ProcessId) -join ", ") + " -- clean up first")
    exit 2
}

# --- keep-awake (mirrors ophir-bot/rebalance.ps1) -----------------------------
if (-not ("Win32.Power" -as [type])) {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
}
$ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM = [uint32]1; $ES_DISPLAY = [uint32]2
[void][Win32.Power]::SetThreadExecutionState([uint32]($ES_CONTINUOUS -bor $ES_SYSTEM -bor $ES_DISPLAY))
Write-Host "keep-awake armed (no sleep / display-off while the session runs)"

# --- kill-on-close job object -------------------------------------------------
if (-not ("Win32.JobKill" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace Win32 {
public static class JobKill {
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr CreateJobObject(IntPtr attrs, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit; public UIntPtr Affinity;
        public uint PriorityClass; public uint SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount,
                     ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed;
    }

    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;
    const int JobObjectExtendedLimitInformation = 9;

    // The job handle is deliberately never closed: it dies with THIS process,
    // and KILL_ON_JOB_CLOSE then terminates every process in the job.
    public static void AttachKillOnClose(IntPtr processHandle) {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new InvalidOperationException("CreateJobObject failed");
        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        int len = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
        IntPtr ptr = Marshal.AllocHGlobal(len);
        try {
            Marshal.StructureToPtr(info, ptr, false);
            if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, ptr, (uint)len))
                throw new InvalidOperationException("SetInformationJobObject failed");
        } finally { Marshal.FreeHGlobal(ptr); }
        if (!AssignProcessToJobObject(job, processHandle))
            throw new InvalidOperationException("AssignProcessToJobObject failed");
    }
}
}
'@
}

# Git Bash, not WSL bash: run_loop.sh expects the repo's POSIX toolchain.
$gitBash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $gitBash)) { $gitBash = "bash" }

$child = Start-Process -FilePath $gitBash `
    -ArgumentList (@(Join-Path $PSScriptRoot "run_loop.sh") + $args) `
    -NoNewWindow -PassThru
[Win32.JobKill]::AttachKillOnClose($child.Handle)
Write-Host "session tree attached to kill-on-close job (pid $($child.Id))"
$child.WaitForExit()
exit $child.ExitCode
