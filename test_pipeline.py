# test_pipeline.py
import time
import redis
from app.tasks.detection_tasks import detect_text_task
from app.core.config import settings

def run_simulation():
    user_id = 3
    call_id = 9999  # 模拟一通真实的电话通话
    
    # 连接 Redis，用于模拟底层音视频向"黑板"写入数据
    r = redis.from_url(settings.REDIS_URL)

    print("=== 🎬 开始模拟【三级漏斗多模态风控】测试 ===")

    # 初始化/清空当前通话的黑板数据
    r.delete(f"call:{call_id}:latest_audio_conf")
    r.delete(f"call:{call_id}:latest_video_conf")
    r.setex(f"call:{call_id}:latest_audio_conf", 3600, "0.05") # 初始非常安全
    r.setex(f"call:{call_id}:latest_video_conf", 3600, "0.02")

    # ---------------------------------------------------------
    # 测试 1：正常的寒暄（期望：被本地 ONNX 极速拦截，根本不呼叫大模型）
    # ---------------------------------------------------------
    print("\n[时间 00:01] 说话：喂，小王啊，晚上打算吃什么？")
    res1 = detect_text_task.delay(text="喂，小王啊，晚上打算吃什么？", user_id=user_id, call_id=call_id)
    time.sleep(3) 
    print(f"-> 任务ID: {res1.id} ")
    print("   👉 预期 Celery 日志: [ONNX 判定为安全闲聊，跳过大模型处理]")

    # ---------------------------------------------------------
    # 测试 2：开始暴露意图（期望：ONNX 觉得可疑(>0.3)，提交给 LLM 研判）
    # ---------------------------------------------------------
    print("\n[时间 00:15] 说话：我是你领导，现在在外面办点急事。")
    res2 = detect_text_task.delay(text="我是你领导，现在在外面办点急事。", user_id=user_id, call_id=call_id)
    time.sleep(5)
    print(f"-> 任务ID: {res2.id} ")
    print("   👉 预期 Celery 日志: LLM 会结合上下文，可能判定为 suspicious(中危)")

    # ---------------------------------------------------------
    # 测试 3：连环套收网，并伴随高仿音视频（期望：LLM 结合黑板数据做出终极制裁）
    # ---------------------------------------------------------
    print("\n[时间 00:30] 说话：恭喜你中奖了！")
    print("   🚨 (底层系统模拟：音频和视频雷达突然报警，写入 Redis 黑板)")
    r.setex(f"call:{call_id}:latest_audio_conf", 3600, "0.85") # 极高的语音伪造率
    r.setex(f"call:{call_id}:latest_video_conf", 3600, "0.92") # 极高的换脸概率
    
    # 只发文本任务，它会自动去 Redis 拉取上面的高危分数
    res3 = detect_text_task.delay(text="恭喜你中奖了！", user_id=user_id, call_id=call_id)
    
    time.sleep(5)
    print(f"-> 任务ID: {res3.id}")
    print("   👉 预期 Celery 日志: LLM 拿到 ONNX 高分 + 历史聊天 + 音视频高危分数，直接触发 upgrade_level 和强阻断！")

if __name__ == "__main__":
    run_simulation()