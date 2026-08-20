Write-Host "starting remaining ingest"
$root = "D:\projects\hhrag"
Start-Process -FilePath (Join-Path $root "run_remaining.bat") -WorkingDirectory $root -WindowStyle Hidden
Start-Sleep -Seconds 3
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    if ($_.CommandLine -like "*ingest.remaining*") {
        Write-Host ("started PID {0}" -f $_.ProcessId)
    }
}
