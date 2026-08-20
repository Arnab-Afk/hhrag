import json
import subprocess
import tempfile
from pathlib import Path

URL = "https://voiceraghhgoa.arnabbhowmik.in/api/ask"
QUERIES = [
    ("en-IN", "What is the best pizza topping in Naples?"),
    ("en-IN", "Explain photosynthesis in the Amazon canopy."),
    ("en-IN", "Who won the 1998 FIFA World Cup?"),
    ("en-IN", "How do I train a dragon in my backyard?"),
    ("en-IN", "What is the wifi password of this coffee shop?"),
    ("en-IN", "asdfgh qwerty zxcvbn mnbvcxz"),
    ("en-IN", "banana banana banana banana"),
    ("hi-IN", "नेपल्स में सबसे अच्छा पिज्जा टॉपिंग क्या है?"),
    ("hi-IN", "चंद्रयान-3 कब उतरा?"),
    ("en-IN", "When does the southwest monsoon reach Kerala?"),
    ("en-IN", "Who is the current CEO of OpenAI?"),
    ("en-IN", "What is 2 plus 2?"),
    ("en-IN", "Tell me a joke about octopuses."),
    ("en-IN", "Where is Hogwarts located?"),
    ("en-IN", "How many moons does Jupiter have?"),
]

out_dir = Path(r"E:\projects\hhrag\benchmark\results")
for i, (lang, query) in enumerate(QUERIES, 1):
    payload = json.dumps(
        {"query": query, "language": lang, "mode": "extractive"}
    ).encode()
    tmp = out_dir / f"_rand_{i}.json"
    tmp.write_bytes(payload)
    dest = out_dir / f"_rand_out_{i}.json"
    proc = subprocess.run(
        [
            "curl.exe", "-sS", "-o", str(dest),
            "-w", "%{http_code} %{time_total}",
            "--max-time", "30",
            "-X", "POST", URL,
            "-H", "Content-Type: application/json",
            "--data-binary", f"@{tmp}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    meta = proc.stdout.strip()
    try:
        body = json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        print(f"{i:02d} FAIL {meta} {proc.stderr[:120]}")
        continue
    ans = (body.get("answer") or "").replace("\n", " ")[:110]
    print(
        f"{i:02d} refused={body.get('refused')} reason={body.get('refuse_reason')} "
        f"retrieve={body.get('retrieval_ms')} under200={body.get('retrieval_under_200ms')} "
        f"score={body.get('confidence')} | {query[:48]}"
    )
    print(f"    {ans}")
