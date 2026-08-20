import subprocess
cmd = [
    "powershell",
    "-NoProfile",
    "-Command",
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.ProcessId.ToString() + ' | ' + $_.CommandLine }",
]
print(subprocess.check_output(cmd, text=True, errors="replace"))
