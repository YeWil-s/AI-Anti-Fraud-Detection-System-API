import asyncio
from langchain_openai import ChatOpenAI
from app.core.config import settings

async def test_deepseek():
    print("正在连接 DeepSeek 大模型...")
    try:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            max_tokens=100
        )
        # 发送一句简单的测试问候
        response = await llm.ainvoke("你好，请做个简短的自我介绍。")
        print("\n✅ 连接成功！大模型回复如下：")
        print(response.content)
    except Exception as e:
        print(f"\n❌ 连接失败，请检查 API_KEY 是否正确，或者网络是否连通。错误信息: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepseek())