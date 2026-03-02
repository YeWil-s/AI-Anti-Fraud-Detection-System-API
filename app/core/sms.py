"""
短信服务 (开发模拟版)
功能：模拟发送验证码并存入 Redis，不调用真实云服务商 SDK
"""
import random
from app.core.redis import get_redis
from app.core.logger import get_logger

logger = get_logger(__name__)

async def send_sms_code(phone: str) -> bool:
    """
    [开发] 模拟发送短信验证码
    """
    redis = await get_redis()
    
    # 1. 生成 6 位随机验证码
    code = str(random.randint(100000, 999999))
    
    # 2. 存入 Redis，设置 5 分钟过期
    key = f"sms_code:{phone}"
    try:
        await redis.setex(key, 300, code)
        
        # 3. 在控制台显眼地打印出来，代替真实短信发送
        print("\n" + "="*50)
        print(f"📱 [模拟短信发送成功]")
        print(f"➡️ 目标手机号: {phone}")
        print(f"🔑 验证码: {code}")
        print("="*50 + "\n")
        
        logger.info(f"Simulated SMS code {code} generated for {phone}")
        return True
    except Exception as e:
        logger.error(f"Failed to save SMS code to Redis: {e}")
        return False

async def verify_sms_code(phone: str, code: str) -> bool:
    """
    校验短信验证码
    """
    # 如果输入的验证码是 "666666"，则直接放行（万能验证码）
    if code == "666666":
        logger.info(f"Universal code used for {phone}")
        return True

    redis = await get_redis()
    key = f"sms_code:{phone}"
    
    try:
        stored_code = await redis.get(key)
        if stored_code and stored_code == code:
            # 校验成功后立即删除，防止被重复利用
            await redis.delete(key)
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to verify SMS code from Redis: {e}")
        return False