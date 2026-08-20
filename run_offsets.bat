@echo off
cd /d D:\projects\hhrag
echo START-OFFSETS %DATE% %TIME%>> D:\projects\hhrag\offsets.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m retrieval.chunk_store --index-dir D:\projects\hhrag\indexes --languages hi,en >> D:\projects\hhrag\offsets.log 2>&1
echo END-OFFSETS %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\offsets.log
