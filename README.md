# AI 反诈检测系统 · 后端

基于 FastAPI 的后端服务（实时检测、Deepfake、话术分析等）。
接口文档在服务启动后访问：`http://127.0.0.1:8000/docs`。
---

## 部署方式

**仅通过 Docker 使用：拉取或导入已构建的 API 镜像，配合本仓库的 `docker-compose.yml` 启动 MySQL / Redis / MinIO 与 API 容器。** 依赖已打在镜像内，**无需**在宿主机安装 Python。
无公网 IP、需要给移动端访问时，在本机用 [natapp](https://natapp.cn/) 将隧道指向 **`127.0.0.1:8000`**。
---

## 1. 环境要求

| 软件 | 说明 |
|------|------|
| Windows 10/11（或 Linux / macOS） | 下文以 PowerShell 为例 |
| Docker Desktop（或兼容的 Docker 引擎与 Compose） | 拉取/导入镜像并编排容器；Windows 建议开启 WSL2 后端 |
| Git | 克隆本仓库时使用（若用源码压缩包则可省略） |

---

## 2. 获取项目文件

用 Git 克隆仓库，或解压源码包后，进入项目根目录（至少需包含 `docker-compose.yml`、`.env.example`、`alembic/` 等；镜像内已含应用与 `models/` 时可只带编排与配置）。示例：

```powershell
cd D:\00_frameFile
```

---

## 3. 环境变量 `.env`
### 3.1 准备配置文件
`.env` 被 `.gitignore` 忽略，不会出现在仓库里。部署时在**项目根目录**（与 `main.py` 同级）放置一份 `.env`：可从本机备份复制过来，或从模板生成（见 3.4）。文件名必须是 `.env`。

### 3.2 与 `docker-compose.yml` 对齐的项（改一处就要改另一处）

以下默认值与当前仓库 `docker-compose.yml` 中 MySQL / MinIO 一致。**若只改 `.env` 而不同步改 Compose，会导致连库失败。**

| 变量 | 默认含义（API 与 MySQL 均在同一 `docker compose` 中时） |
|------|----------|
| `DATABASE_URL` | `mysql+aiomysql://root:123456@mysql:3306/ai_fraud_detection`（服务名 `mysql`，容器内端口 `3306`） |
| `REDIS_URL` | `redis://redis:6379/0`（或由 Compose 的 `environment` 覆盖，见第 4 节） |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | 与 `.env.example` 或 Compose 中 `redis://redis:6379/1`、`/2` 一致 |
| `MINIO_ENDPOINT` | `minio:9000`（或由 Compose 的 `environment` 覆盖为 `minio:9000`） |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 与 Compose 中 MinIO 一致，默认 `minioadmin` |

若修改 MySQL root 密码，必须同时修改 **`docker-compose.yml` 里 `mysql` 服务**与 **`DATABASE_URL`**。

### 3.3 可自行替换的密钥与第三方配置（不影响依赖容器启动）

下列项在**新机器上可重新生成或填空**，一般不影响「先跑通 API + 文档」；涉及的功能若未使用可忽略。

| 变量 | 说明 |
|------|------|
| `SECRET_KEY` | 应用密钥，建议长随机串；更换后已签发的会话若依赖旧密钥会失效 |
| `JWT_SECRET_KEY` | JWT 签名密钥；更换后需重新登录 |
| `LLM_API_KEY` / `LLM_BASE_URL` | 大模型接口；不配置时相关能力不可用 |
| `ZHIPU_API_KEY` | 智谱 API；同上 |
| `SMS_ACCESS_KEY` / `SMS_SECRET_KEY` | 短信；不用可留空 |
| `SMTP_*` / `EMAIL_*` | 邮件；不用可保留占位 |

### 3.4 无模板时

仓库内有 `.env.example`，可复制为 `.env` 再按上表填写：

```powershell
copy .env.example .env
```

---

## 4. 镜像与运行

镜像内已包含 Python 依赖与应用代码，**不要在宿主机或容器内再执行** `pip install -r requirements.txt`。

### 4.1 本机上的镜像在哪里？有没有「下载网站」？

`docker build` 成功后，镜像只保存在**你这台电脑**的 Docker 本地存储里，**不会**自动出现在任何网站或网盘，别人也**无法**通过浏览器去「下载」它，除非你另行分发。

在本机查看是否已有镜像（名称、标签以你构建时为准，例如 `verify`）：

```powershell
docker images ai-fraud-detection-api
```

若列表为空，说明本机还没构建过或已删除，需要在项目根目录执行：

```powershell
docker build -t ai-fraud-detection-api:latest .
```

### 4.2 别人怎么获取这个镜像？（必须有人「交付」）

Docker 镜像**没有**统一的公开下载地址。别人要拿到和你一致的镜像，只能通过下面**三种方式之一**：

| 方式 | 谁做 | 别人怎么做 |
|------|------|------------|
| **离线文件** | 你在本机把镜像打成 tar | 把 `ai-fraud-detection-api.tar` 拷给对方；对方执行 `docker load -i ai-fraud-detection-api.tar`（见 4.7） |
| **镜像仓库** | 你在 Docker Hub、阿里云 ACR、腾讯云等**创建仓库并 `docker push`** | 对方执行 `docker pull <你提供的完整地址，含仓库名/标签>` |
| **对方自己构建** | 对方有本仓库源码与网络 | 对方在项目根目录执行 `docker build -t ai-fraud-detection-api:latest .`（耗时长、镜像体积大） |

总结：**「验证构建成功」只代表你本机已有镜像；要让阅卷方/同事使用，你必须明确二选一——发 tar，或 push 后把 `docker pull` 的地址写进说明。**

### 4.3 对方导入或拉取镜像之后

- 若收到 **tar**：`docker load -i ai-fraud-detection-api.tar`，再用 `docker images` 确认标签（可 `docker tag` 成下文用的名字）。
- 若使用 **`docker pull`**：按你文档里写的仓库地址执行。

以下以本地镜像名 **`ai-fraud-detection-api:latest`** 为例继续配置。

### 4.4 修改 `docker-compose.yml` 中的 `api` 服务

使用 **`image`** 指向上述镜像，**去掉或注释 `build:`**，避免本机重新构建。`api` 段要点如下：

```yaml
  api:
    profiles: ["api"]
    image: ai-fraud-detection-api:latest
    container_name: ai_fraud_api
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      MINIO_ENDPOINT: minio:9000
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_started
    networks:
      - ai_fraud_network
    restart: unless-stopped
```

在项目根目录准备好 **`.env`**（见第 3 节），其中 **`DATABASE_URL`** 使用 Compose 内主机名，例如：

`mysql+aiomysql://root:123456@mysql:3306/ai_fraud_detection`

### 4.5 启动与验证

```powershell
docker compose --profile api up -d
docker compose ps
```

首次启动 MySQL 可能需要数十秒。浏览器访问：`http://127.0.0.1:8000/docs`；健康检查：`http://127.0.0.1:8000/health`。

### 4.6 数据库迁移（首次）

```powershell
docker compose --profile api run --rm api alembic upgrade head
```

若 `api` 已在运行，也可：`docker compose exec api alembic upgrade head`。

### 4.7 导出镜像供他人使用（离线交付时）

在本机已有镜像的前提下（例如标签为 `verify` 或 `latest`）：

```powershell
docker tag ai-fraud-detection-api:verify ai-fraud-detection-api:latest
docker save -o ai-fraud-detection-api.tar ai-fraud-detection-api:latest
```

将 **`ai-fraud-detection-api.tar`**（体积可能很大）与本仓库中的 `docker-compose.yml`、`.env.example`、以及 `.env` 填写说明一并交给对方；对方用 **4.3 节**的 `docker load` 导入即可。

---

## 5. natapp 内网穿透（无公网服务器时）

1. 在 [natapp.cn](https://natapp.cn/) 注册，购买或申请**免费/付费隧道**，得到 **authtoken**。
2. 下载对应系统的客户端，在本地配置该 authtoken，将隧道指向本机 **`127.0.0.1:8000`**（与后端端口一致）。
3. 启动 natapp 后，控制台会显示公网访问地址，一般为 **`https://xxxx.natapp1.cc`** 等形式（以实际为准）。

### 5.1 移动端 App 如何填地址

- HTTP API 基址指向穿透后的根地址即可，具体路径仍以项目为准（例如用户相关为 `/api/users/...`，完整列表见 `http://127.0.0.1:8000/docs`）。
- WebSocket 使用 **`wss://`** + 同一域名 + 项目中的 WS 路径（例如 `/api/detection/ws/...`）。穿透地址为 **HTTPS** 时，请用 **`wss://`**，不要用 `ws://`。
- 免费隧道若更换域名或端口，需在 App 的配置里同步更新。

### 5.2 说明

穿透与业务代码无关。运行时要同时保持：**后端已监听本机 `8000`**，且 **natapp 客户端进程未退出**。

---

## 6. 常见问题

1. **数据库连接失败**  
   确认 `mysql` 容器已 healthy；`DATABASE_URL` 中主机名为 **`mysql`**、端口为 **`3306`**（与 API 同 Compose 网络）。

2. **迁移报错**  
   使用第 4.6 节命令；确保 `.env` 可被容器读取且 `DATABASE_URL` 正确。

3. **App 能打开文档但业务异常**  
   检查 `LLM_API_KEY` 等可选配置是否在使用场景下已填写。

4. **Celery 异步任务**  
   若需异步检测队列，在容器或宿主机按仓库内说明启动 Celery Worker；仅验证 HTTP/WebSocket 可先不启。

---

## 7. 项目结构（摘要）

```
├── app/              # 应用代码（API、服务、模型等）
├── alembic/          # 数据库迁移
├── models/           # AI 模型权重（体积大，需随部署一并准备）
├── main.py           # 入口
├── requirements.txt  # 构建镜像时使用；仅拉取现成镜像运行则不需要在宿主机安装
├── docker-compose.yml
├── .env.example      # 环境变量模板（勿提交真实 .env）
└── Dockerfile        # 构建 API 镜像时使用
```

API 列表以运行后的 **Swagger**（`/docs`）为准。
