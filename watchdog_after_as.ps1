# After Assamese finishes, stop the all-language --overwrite job so it
# cannot wipe Hindi/English/Tamil/Marathi built by the priority job.
$ErrorActionPreference = "Continue"
$root = "D:\projects\hhrag"
$asMeta = Join-Path $root "indexes\as\shard_meta.json"
$log = Join-Path $root "watchdog.log"

function Write-Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line
}

Write-Log "watchdog started; waiting for $asMeta"
while (-not (Test-Path $asMeta)) {
    Start-Sleep -Seconds 30
}
Write-Log "Assamese shard complete; stopping all-language overwrite job"

Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $cmd = $_.CommandLine
    if ($null -eq $cmd) { return }
    $isAll = ($cmd -like "*ingest.full_train*") -and ($cmd -like "*--batch-size 512*") -and ($cmd -notlike "*--languages*")
    if ($isAll) {
        Write-Log ("killing PID {0} {1}" -f $_.ProcessId, $cmd)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Seconds 5
Write-Log "starting remaining-language job"
Start-Process -FilePath (Join-Path $root "run_rest_train.bat") -WorkingDirectory $root -WindowStyle Hidden
Write-Log "watchdog done"
