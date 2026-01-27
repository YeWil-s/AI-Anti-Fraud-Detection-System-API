@echo off
REM Celery Worker启动脚本

echo ========================================
echo   启动Celery Worker
echo ========================================
echo.

REM 激活虚拟环境
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [√] 虚拟环境已激活
) else (
    echo [×] 虚拟环境不存在,请先运行 setup.bat
    pause
    exit /b 1
)

echo.
echo [提示] 启动Celery Worker...
echo [提示] 确保Redis正在运行: docker-compose up -d redis
echo.

REM 启动Celery Worker
python -m celery -A app.tasks.celery_app worker --loglevel=info --pool=solo

pause
