import subprocess

tr = (
    r'D:\projects\hhrag\venv\Scripts\python.exe -m http.server 8766 '
    r'--bind 0.0.0.0 --directory D:\projects\hhrag\indexes'
)
create = [
    "schtasks", "/Create", "/TN", "hhrag_idxhttp",
    "/TR", tr, "/SC", "ONSTART", "/F",
]
print(subprocess.check_output(create, text=True, errors="replace"))
print(subprocess.check_output(["schtasks", "/Run", "/TN", "hhrag_idxhttp"], text=True, errors="replace"))
print(subprocess.check_output(["schtasks", "/Query", "/TN", "hhrag_idxhttp", "/FO", "LIST"], text=True, errors="replace"))
