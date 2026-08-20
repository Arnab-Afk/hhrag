@echo off
cd /d D:\projects\hhrag
echo START-PRIORITY %DATE% %TIME%>> D:\projects\hhrag\priority_train.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.full_train --languages hi,ta,mr --overwrite --batch-size 256 >> D:\projects\hhrag\priority_train.log 2>&1
echo END-PRIORITY %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\priority_train.log
