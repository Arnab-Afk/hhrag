import subprocess

out = subprocess.check_output(
    [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.ProcessId.ToString() + ' | ' + $_.CommandLine }",
    ],
    text=True,
    errors="replace",
)
for line in out.splitlines():
    if "ingest.remaining" in line:
        pid = int(line.split("|", 1)[0].strip())
        print("killing", pid, line[:120])
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
print("done kill")
