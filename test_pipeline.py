# test_pipeline.py
import time
import base64
from app.tasks.detection_tasks import detect_text_task, multi_modal_fusion_task

def run_simulation():
    user_id = 3
    call_id = 9999  # 模拟一通真实的电话通话

    print("=== 🎬 开始模拟诈骗通话 ===")

    # ---------------------------------------------------------
    # 测试 1：正常的寒暄（测试记忆池写入）
    # ---------------------------------------------------------
    print("\n[时间 00:01] 骗子发话：喂，小王啊，最近工作怎么样？")
    res1 = detect_text_task.delay(text="喂，小王啊，最近工作怎么样？", user_id=user_id, call_id=call_id)
    # 等待几秒钟让 Celery 处理完
    time.sleep(3) 
    print(f"-> 任务ID: {res1.id} (请去终端2查看大模型判定结果，应该是 safe)")

    # ---------------------------------------------------------
    # 测试 2：开始暴露意图（测试大模型结合上一句的上下文理解）
    # ---------------------------------------------------------
    print("\n[时间 00:15] 骗子发话：我是你领导，现在在外面办点急事。")
    res2 = detect_text_task.delay(text="我是你领导，现在在外面办点急事。", user_id=user_id, call_id=call_id)
    time.sleep(3)
    print(f"-> 任务ID: {res2.id} (请去终端2查看，可能会判定为 suspicious)")

    # ---------------------------------------------------------
    # 测试 3：连环套收网，并伴随高仿音视频（测试多模态融合引擎！）
    # ---------------------------------------------------------
    print("\n[时间 00:30] 骗子发话：赶紧给我转两万块钱应急！")
    print("           (底层 AI 探针报警：发现音频克隆特征 0.85，视频唇形不自然 0.92)")
    
    # 触发最高级别的多模态融合任务
    res3 = multi_modal_fusion_task.delay(
        text="赶紧给我转两万块钱应急！", 
        audio_conf=0.85, 
        video_conf=0.92, 
        user_id=user_id, 
        call_id=call_id
    )
    time.sleep(5)
    print(f"-> 任务ID: {res3.id}")
    print("\n=== 🎯 请观察 Celery 终端（终端2）的日志 ===")
    print("你应该能看到大模型：")
    print("1. 成功读取了前面的聊天记录（喂、我是你领导、转钱）")
    print("2. 结合极高的音视频置信度，给出了 critical/fake 的终极拦截指令！")

if __name__ == "__main__":
    run_simulation()