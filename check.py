import onnxruntime as ort
import os

model_path = "./models/text_fraud_model.onnx"

print(f"检查文件: {model_path}")
if not os.path.exists(model_path):
    print("❌ 文件不存在！")
    exit()

print(f"文件大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

try:
    session = ort.InferenceSession(model_path)
    inputs = [i.name for i in session.get_inputs()]
    print(f"🔍 模型入口参数名: {inputs}")

    if "input_ids" in inputs:
        print("✅ 这是正确的 BERT 模型！(请立即重启 Celery)")
    elif "input" in inputs:
        print("❌ 这是错误的视频/音频模型！(请重新复制 400MB 的那个文件)")
    else:
        print("⚠️ 未知模型结构")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")