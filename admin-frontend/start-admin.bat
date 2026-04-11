@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo [管理后台] 正在启动本地页面服务…
echo 若浏览器未自动打开，请访问: http://127.0.0.1:8765/
echo 关闭标题为 admin-frontend-http 的黑色窗口即停止服务。
echo.

start "admin-frontend-http" cmd /k cd /d "%~dp0" ^&^& python -m http.server 8765
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765/"

pause
