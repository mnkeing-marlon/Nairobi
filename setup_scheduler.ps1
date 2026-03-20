# ──────────────────────────────────────────────────────────────
# setup_scheduler.ps1
# Register a Windows Task Scheduler job that runs the full
# pipeline every 30 minutes.
# Run this script ONCE with elevated privileges (Admin).
# ──────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────
$TaskName   = "NairobiAQ_Pipeline"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe  = Join-Path $ProjectDir ".prophet\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectDir "run_pipeline_full.py"

# Verify paths exist
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python not found at $PythonExe. Activate or create the venv first."
    exit 1
}
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found at $ScriptPath."
    exit 1
}

# ── Create the scheduled task ─────────────────────────────
$Action  = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Days 365)

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 25)

# Remove existing task if present
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'."
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Nairobi Air Quality: scrape + pipeline + retrain every 30 min" `
    -RunLevel Highest

Write-Host ""
Write-Host "Scheduled task '$TaskName' created successfully."
Write-Host "  Interval : every 30 minutes"
Write-Host "  Python   : $PythonExe"
Write-Host "  Script   : $ScriptPath"
Write-Host ""
Write-Host "To remove:  Unregister-ScheduledTask -TaskName '$TaskName'"
