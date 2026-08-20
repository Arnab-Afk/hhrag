@echo off
cd /d D:\projects\hhrag
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
echo START-FINALIZE %DATE% %TIME%>> D:\projects\hhrag\finalize.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.finalize_jsonl --languages hi,en,ta --overwrite --batch-size 512 >> D:\projects\hhrag\finalize.log 2>&1
echo END-FINALIZE %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\finalize.log
