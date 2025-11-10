# AI伪造检测与诈骗预警系统 - 后端

## 项目简介
基于FastAPI的AI伪造检测与诈骗预警系统后端服务，提供实时通话检测、Deepfake识别、诈骗话术分析等功能。

## 技术栈
- **Web框架**: FastAPI
- **数据库**: PostgreSQL
- **缓存**: Redis
- **对象存储**: MinIO
- **异步任务**: Celery
- **AI框架**: PyTorch / TensorFlow
- **容器化**: Docker

## 快速开始

### 1. 环境准备
```bash
# 创建Python虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
# 复制环境变量模板
copy .env.example .env

# 编辑.env文件，填入实际配置
```

### 3. 启动数据库和Redis (使用Docker)
```bash
docker-compose up -d postgres redis minio
```

### 4. 初始化数据库
```bash
# 创建数据库迁移
alembic revision --autogenerate -m "Initial migration"

# 执行迁移
alembic upgrade head
```

### 5. 启动应用
```bash
# 开发模式
python main.py

# 或使用uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 访问API文档
打开浏览器访问: http://localhost:8000/docs

## 项目结构
```
d:/00_firstFile/
├── app/
│   ├── __init__.py
│   ├── api/              # API路由
│   │   ├── __init__.py
│   │   └── users.py      # 用户管理接口
│   ├── core/             # 核心配置
│   │   ├── __init__.py
│   │   ├── config.py     # 配置管理
│   │   ├── security.py   # JWT认证
│   │   ├── redis.py      # Redis工具
│   │   └── storage.py    # MinIO存储
│   ├── db/               # 数据库
│   │   ├── __init__.py
│   │   └── database.py   # 数据库连接
│   ├── models/           # 数据模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── call_record.py
│   │   ├── ai_detection_log.py
│   │   ├── risk_rule.py
│   │   └── blacklist.py
│   └── schemas/          # Pydantic模型
│       └── __init__.py
├── alembic/              # 数据库迁移
│   ├── env.py
│   └── script.py.mako
├── models/               # AI模型文件目录
├── main.py               # 应用入口
├── requirements.txt      # 依赖列表
├── .env.example          # 环境变量模板
├── .gitignore
├── alembic.ini           # Alembic配置
├── docker-compose.yml    # Docker编排
└── Dockerfile            # Docker镜像
```

## API接口

### 用户管理
- `POST /api/users/register` - 用户注册
- `POST /api/users/login` - 用户登录
- `GET /api/users/me` - 获取当前用户信息
- `PUT /api/users/family/{family_id}` - 绑定家庭组

### 系统接口
- `GET /` - 系统信息
- `GET /health` - 健康检查

## 数据库表结构

### users (用户表)
- user_id: 用户ID
- phone: 手机号
- name: 姓名
- password_hash: 密码哈希
- family_id: 家庭组ID
- is_active: 是否激活
- created_at: 创建时间

### call_records (通话记录表)
- call_id: 通话ID
- user_id: 用户ID
- caller_number: 来电号码
- start_time: 开始时间
- end_time: 结束时间
- duration: 通话时长
- detected_result: 检测结果
- audio_url: 录音URL

### ai_detection_logs (AI检测日志表)
- log_id: 日志ID
- call_id: 通话ID
- voice_confidence: 声音置信度
- video_confidence: 视频置信度
- text_confidence: 文本置信度
- overall_score: 综合评分

### risk_rules (风险规则表)
- rule_id: 规则ID
- keyword: 关键词
- action: 动作
- risk_level: 风险等级

### number_blacklist (号码黑名单表)
- id: ID
- number: 电话号码
- source: 来源
- report_count: 举报次数

## 开发指南

### 添加新的API路由
1. 在 `app/api/` 目录下创建新的路由文件
2. 在 `app/api/__init__.py` 中导出路由
3. 在 `main.py` 中注册路由

### 添加新的数据模型
1. 在 `app/models/` 目录下创建模型文件
2. 继承 `Base` 类定义表结构
3. 创建对应的 Pydantic Schema
4. 运行 `alembic revision --autogenerate` 生成迁移

## 部署

### Docker部署
```bash
# 构建镜像
docker build -t ai-fraud-detection-api .

# 运行容器
docker-compose up -d
```

## 下一步开发计划
- [ ] 实现JWT认证中间件
- [ ] 集成短信验证码服务
- [ ] 开发通话检测API
- [ ] 集成AI模型服务
- [ ] 实现WebSocket实时推送
- [ ] 添加Celery异步任务
- [ ] 完善家庭组功能
- [ ] 添加单元测试

## 许可证
MIT License
