import subprocess

out = subprocess.check_output(
    [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'curl|cmd' -and $_.CommandLine -match 'msmarco|fetch.bat|hhrag_parquets' } | ForEach-Object { $_.ProcessId.ToString() + ' | ' + $_.CommandLine }",
    ],
    text=True,
    errors="replace",
)
print(out or "(no curl/fetch procs)")
for line in out.splitlines():
    if not line.strip():
        continue
    pid = line.split("|", 1)[0].strip()
    if pid.isdigit():
        print("killing", pid)
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], check=False)
subprocess.run(["schtasks", "/End", "/TN", "hhrag_parquets"], check=False)
print("ended hhrag_parquets task")
