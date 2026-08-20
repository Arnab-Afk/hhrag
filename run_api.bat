@echo off
cd /d D:\projects\hhrag
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
echo START-API %DATE% %TIME%>> D:\projects\hhrag\api.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m uvicorn app.main:app --host 0.0.0.0 --port 7860 >> D:\projects\hhrag\api.log 2>&1
echo END-API %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\api.log
