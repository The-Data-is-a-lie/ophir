# Keep-awake wrapper for unattended autoresearch sessions.
#
# Sessions s2/s3 (2026-07-08/09) died mid-GPU-training ~30-40 min after the
# last user input: this box sleeps after 30 min idle (AC STANDBYIDLE=0x708),
# and its idle session/display power transition tears down the console tree,
# sending the Fortran/MKL runtime under torch a CTRL_CLOSE_EVENT that
# hard-aborts training (same failure the OphirRebalance task hit in June --
# see ophir-bot/rebalance.ps1). Mirror that task's proven fix: hold
# ES_CONTINUOUS|ES_SYSTEM|ES_DISPLAY for the whole session; the hold releases
# automatically when this process exits.
#
# Usage (all arguments are forwarded verbatim to run_loop.sh):
#   powershell -NoProfile -File autoresearch/run_loop_awake.ps1 `
#     --session s3-20260709 --max-iters 3 --epsilon 0.049

if (-not ("Win32.Power" -as [type])) {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
}
$ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM = [uint32]1; $ES_DISPLAY = [uint32]2
[void][Win32.Power]::SetThreadExecutionState([uint32]($ES_CONTINUOUS -bor $ES_SYSTEM -bor $ES_DISPLAY))
Write-Host "keep-awake armed (no sleep / display-off while the session runs)"

# Git Bash, not WSL bash: run_loop.sh expects the repo's POSIX toolchain.
$gitBash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $gitBash)) { $gitBash = "bash" }
& $gitBash (Join-Path $PSScriptRoot "run_loop.sh") @args
exit $LASTEXITCODE
