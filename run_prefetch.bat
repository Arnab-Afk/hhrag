@echo off
cd /d D:\projects\hhrag
set HF_HUB_OFFLINE=
set TRANSFORMERS_OFFLINE=
echo START-PREFETCH %DATE% %TIME%>> D:\projects\hhrag\prefetch.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.prefetch_parquets >> D:\projects\hhrag\prefetch.log 2>&1
echo END-PREFETCH %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\prefetch.log
