@echo off
cd /d D:\projects\hhrag
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set MSMARCO_PARQUET_DIR=D:\projects\hhrag\msmarco
echo START-REMAINING %DATE% %TIME%>> D:\projects\hhrag\remaining.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.remaining --batch-size 512 --out D:\projects\hhrag\indexes >> D:\projects\hhrag\remaining.log 2>&1
echo END-REMAINING %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\remaining.log
