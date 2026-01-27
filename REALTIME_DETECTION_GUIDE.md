# 实时检测服务使用指南

## 📋 功能概览

### 已完成功能 ✅

#### 1. 音视频流处理
- ✅ WebSocket连接管理
- ✅ 音频流接收与切片处理  
- ✅ 视频帧提取接口
- ✅ 文件上传与MinIO存储

#### 2. AI模型集成
- ✅ 模型服务层架构
- ✅ ONNXRuntime配置
- ✅ 模型加载与预测接口
- ✅ PyTorch/TensorFlow环境准备

#### 3. Celery异步任务队列
- ✅ Celery应用配置
- ✅ 任务调度与分发
- ✅ Redis消息代理
- ✅ 任务状态监控

---

## 🚀 快速开始

### 1. 启动所需服务

```bash
# 启动MySQL + Redis + MinIO
docker-compose up -d

# 启动FastAPI应用
python main.py

# 启动Celery Worker (新终端)
start_celery.bat
```

### 2. WebSocket实时检测

#### 连接WebSocket
```javascript
// 前端连接示例
const ws = new WebSocket('ws://localhost:8000/api/detection/ws/1');

ws.onopen = () => {
    console.log('WebSocket连接成功');
};

// 发送音频数据
ws.send(JSON.stringify({
    type: 'audio',
    data: audioBase64Data
}));

// 发送视频帧
ws.send(JSON.stringify({
    type: 'video',
    data: frameBase64Data
}));

// 接收检测结果
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log('检测结果:', result);
};
```

### 3. 文件上传API

#### 上传音频
```bash
curl -X POST "http://localhost:8000/api/detection/upload/audio" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@audio.mp3"
```

#### 上传视频
```bash
curl -X POST "http://localhost:8000/api/detection/upload/video" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@video.mp4"
```

#### 提取视频帧
```bash
curl -X POST "http://localhost:8000/api/detection/extract-frames" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@video.mp4" \
  -F "frame_rate=1"
```

### 4. 异步任务API

#### 提交音频检测任务
```bash
curl -X POST "http://localhost:8000/api/tasks/audio/detect" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_features": [[0.1, 0.2, ...], ...],
    "call_id": 1
  }'
```

#### 查询任务状态
```bash
curl "http://localhost:8000/api/tasks/status/TASK_ID"
```

---

## 📁 项目结构

```
app/
├── api/
│   ├── detection.py      # 实时检测API (WebSocket + 文件上传)
│   ├── tasks.py          # 任务管理API
│   └── users.py          # 用户管理API
├── services/
│   ├── websocket_manager.py   # WebSocket连接管理
│   ├── audio_processor.py     # 音频处理器
│   ├── video_processor.py     # 视频处理器
│   └── model_service.py       # AI模型服务
├── tasks/
│   ├── celery_app.py          # Celery配置
│   └── detection_tasks.py     # 检测异步任务
└── core/
    └── storage.py             # MinIO存储服务
```

---

## 🔧 配置说明

### 环境变量 (.env)

```ini
# AI模型路径
VOICE_MODEL_PATH=./models/voice_detection.onnx
VIDEO_MODEL_PATH=./models/video_detection.onnx
TEXT_MODEL_PATH=./models/text_detection

# Celery配置
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# WebSocket配置
WS_HEARTBEAT_INTERVAL=30
```

### 模型文件准备

1. 创建模型目录:
```bash
mkdir models
```

2. 放置ONNX模型文件:
   - `models/voice_detection.onnx` - 语音检测模型
   - `models/video_detection.onnx` - 视频检测模型

---

## 🎯 API接口列表

### WebSocket接口
| 端点 | 描述 |
|------|------|
| `ws://localhost:8000/api/detection/ws/{user_id}` | WebSocket实时检测连接 |

### 文件上传接口
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/detection/upload/audio` | POST | 上传音频文件 |
| `/api/detection/upload/video` | POST | 上传视频文件 |
| `/api/detection/extract-frames` | POST | 提取视频关键帧 |

### 任务管理接口
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/tasks/audio/detect` | POST | 提交音频检测任务 |
| `/api/tasks/video/detect` | POST | 提交视频检测任务 |
| `/api/tasks/text/detect` | POST | 提交文本检测任务 |
| `/api/tasks/status/{task_id}` | GET | 查询任务状态 |

---

## ⚡ 性能优化

### 1. 使用GPU加速
修改 `app/services/model_service.py`:
```python
self.voice_session = ort.InferenceSession(
    settings.VOICE_MODEL_PATH,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### 2. 调整Celery并发
```bash
celery -A app.tasks.celery_app worker --concurrency=4
```

### 3. Redis性能调优
```ini
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

---

## 🐛 常见问题

### 1. Celery Worker无法启动
**解决**: 确保Redis正在运行
```bash
docker-compose up -d redis
```

### 2. WebSocket连接失败
**解决**: 检查防火墙设置,确保端口8000开放

### 3. 模型加载失败
**解决**: 
- 检查模型文件路径是否正确
- 确认ONNX Runtime已安装: `pip install onnxruntime`

---

## 📊 监控与日志

### Celery任务监控
```bash
# 启动Flower监控面板
celery -A app.tasks.celery_app flower
# 访问: http://localhost:5555
```

### 查看应用日志
```bash
tail -f logs/app.log
```

---

## 🔐 安全建议

1. **生产环境**: 修改JWT密钥和MinIO密钥
2. **WebSocket认证**: 添加token验证
3. **文件上传限制**: 设置文件大小和类型限制
4. **Rate Limiting**: 添加请求频率限制

---

## 📝 下一步开发

- [ ] 添加模型热更新功能
- [ ] 实现分布式任务队列
- [ ] 添加任务优先级机制
- [ ] 集成Prometheus监控
- [ ] 添加WebSocket断线重连

---

## 📞 技术支持

如有问题请提Issue或查看API文档: http://localhost:8000/docs
