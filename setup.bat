@echo off
REM 项目初始化脚本 (Windows)

echo === AI Anti-Fraud Detection System - 项目初始化 ===

REM 1. 检查Python版本
echo 检查Python版本...
python --version

REM 2. 创建虚拟环境
echo 创建Python虚拟环境...
python -m venv venv

REM 3. 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 4. 安装依赖
echo 安装Python依赖...
pip install -r requirements.txt

REM 5. 复制环境变量
if not exist .env (
    echo 复制环境变量模板...
    copy .env.example .env
    echo 请编辑 .env 文件配置数据库等信息
)

REM 6. 启动Docker服务
echo 启动Docker服务...
docker-compose up -d postgres redis minio

REM 7. 等待数据库启动
echo 等待数据库启动...
timeout /t 10 /nobreak

REM 8. 初始化数据库
echo 初始化数据库...
alembic upgrade head

echo.
echo === 初始化完成! ===
echo 运行 'python main.py' 启动应用

pause
