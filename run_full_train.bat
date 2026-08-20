@echo off
cd /d D:\projects\hhrag
echo START %DATE% %TIME%>> D:\projects\hhrag\full_train.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.full_train --overwrite --batch-size 512 >> D:\projects\hhrag\full_train.log 2>&1
echo END %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\full_train.log
