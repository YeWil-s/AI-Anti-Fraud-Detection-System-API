import os
import asyncio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from app.services.model_service import ModelService
from app.core.config import settings

EVAL_DIR = Path("dataset/audio")
REAL_DIR = EVAL_DIR / "real"
FAKE_DIR = EVAL_DIR / "fake"

async def run_evaluation():
    print(f" 开始模型评估 (阈值: {settings.VOICE_DETECTION_THRESHOLD}, DL权重: {0.7}, ML权重: {0.3})")
    print(" 正在加载模型到内存...")
    model_service = ModelService()
    y_true = []  # 真实标签: 0=真, 1=假
    y_pred = []  # 预测标签: 0=真, 1=假
    y_scores = [] # 预测得分 (用于画ROC曲线或后续分析)
    real_files = list(REAL_DIR.glob("*.wav")) + list(REAL_DIR.glob("*.mp3"))
    fake_files = list(FAKE_DIR.glob("*.wav")) + list(FAKE_DIR.glob("*.mp3"))
    
    if not real_files and not fake_files:
        print(" 找不到测试音频")
        return

    test_cases = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]

    print(f" 共找到 {len(test_cases)} 个测试样本，开始推理...")
    for file_path, label in tqdm(test_cases, desc="Evaluating"):
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            result = await model_service.predict_voice(audio_bytes)
            
            y_true.append(label)
            y_pred.append(1 if result["is_fake"] else 0)
            y_scores.append(result["confidence"])
            
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
    print("\n" + "="*40)
    print("模 型 评 估 报 告")
    print("="*40)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f" 准确率 (Accuracy):  {acc:.4f}  ")
    print(f" 精确率 (Precision): {prec:.4f}  ")
    print(f" 召回率 (Recall):    {rec:.4f}  ")
    print(f" F1 分数 (F1-Score): {f1:.4f}  ")
    print("-" * 40)
    print("🧠 混淆矩阵 (Confusion Matrix):")
    print(f"  [真声音] 预测为真(TN): {cm[0][0]:<5} | 误报为假(FP): {cm[0][1]}")
    print(f"  [假声音] 漏报为真(FN): {cm[1][0]:<5} | 预测为假(TP): {cm[1][1]}")
    print("="*40)
if __name__ == "__main__":
    asyncio.run(run_evaluation())