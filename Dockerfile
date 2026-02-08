FROM python:3.11-slim

WORKDIR /app

# [修正] 安装系统依赖
# 新增: libsndfile1 (音频处理必备)
# 新增: libgl1, libglib2.0-0 (OpenCV GUI版必备，即使转headless也建议保留libglib)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]