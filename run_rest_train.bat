@echo off
cd /d D:\projects\hhrag
echo START-REST %DATE% %TIME%>> D:\projects\hhrag\rest_train.log
D:\projects\hhrag\venv\Scripts\python.exe -u -m ingest.full_train --languages bn,gu,kn,ml,ne,or,pa,sa,ur --batch-size 512 >> D:\projects\hhrag\rest_train.log 2>&1
echo END-REST %DATE% %TIME% exit=%ERRORLEVEL%>> D:\projects\hhrag\rest_train.log
