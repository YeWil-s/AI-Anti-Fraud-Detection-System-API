"""
短信验证码服务
"""
import random
import redis
from typing import Optional
from app.core.config import settings
from app.core.logger import get_logger

# 创建Redis客户端(用于存储验证码)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)  # type: ignore[assignment]


def generate_sms_code(length: int = 6) -> str:
    """
    生成随机验证码
    
    Args:
        length: 验证码长度,默认6位
        
    Returns:
        验证码字符串
    """
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])


def send_sms_code(phone: str) -> bool:
    """
    发送短信验证码
    
    Args:
        phone: 手机号
        
    Returns:
        是否发送成功
    """
    # 生成验证码
    code = generate_sms_code()
    
    try:
    # 存储到Redis,有效期5分钟
        redis_key = f"sms_code:{phone}"
        redis_client.setex(redis_key, 300, code)
    
    # TODO: 接入真实短信服务商API (阿里云/腾讯云)
    # 示例: 阿里云短信SDK
    # try:
    #     from aliyunsdkcore.client import AcsClient
    #     from aliyunsdkcore.request import CommonRequest
    #     
    #     client = AcsClient(settings.SMS_ACCESS_KEY, settings.SMS_SECRET_KEY, 'cn-hangzhou')
    #     request = CommonRequest()
    #     request.set_domain('dysmsapi.aliyuncs.com')
    #     request.set_version('2017-05-25')
    #     request.set_action_name('SendSms')
    #     request.set_method('POST')
    #     request.add_query_param('PhoneNumbers', phone)
    #     request.add_query_param('SignName', '你的签名')
    #     request.add_query_param('TemplateCode', 'SMS_xxxxx')
    #     request.add_query_param('TemplateParam', f'{{"code":"{code}"}}')
    #     
    #     response = client.do_action_with_exception(request)
    #     return True
    # except Exception as e:
    #     print(f"短信发送失败: {e}")
    #     return False
    
        logger.info(f"[DEV_MODE] 手机号 {phone} 的验证码是: {code}")
        return True

    except Exception as e:
        # 记录 Redis 连接失败或短信 API 异常
        logger.error(f"Failed to send SMS to {phone}: {e}", exc_info=True)
        return False


def verify_sms_code(phone: str, code: str) -> bool:
    """
    验证短信验证码
    
    Args:
        phone: 手机号
        code: 用户输入的验证码
        
    Returns:
        验证是否通过
    """
    redis_key = f"sms_code:{phone}"
    stored_code = redis_client.get(redis_key)
    
    if not stored_code:
        return False  # 验证码不存在或已过期
    
    # 验证通过后删除验证码(一次性使用)
    if stored_code == code:
        redis_client.delete(redis_key)
        return True
    
    return False


def get_remaining_time(phone: str) -> Optional[int]:
    """
    获取验证码剩余有效时间
    
    Args:
        phone: 手机号
        
    Returns:
        剩余秒数,不存在则返回None
    """
    redis_key = f"sms_code:{phone}"
    ttl = redis_client.ttl(redis_key)
    return ttl if ttl > 0 else None

def send_fraud_alert_sms(phone: str, name: str, risk_level: str, time_str: str) -> bool:
    """
    [新增] 发送诈骗预警短信给家庭管理员
    
    Args:
        phone: 接收者手机号
        name: 正在通话的用户姓名
        risk_level: 风险等级
        time_str: 发生时间
    """
    try:
        # 模拟短信模板: 
        # "【反诈预警】您的家人{name}正在进行一笔{risk_level}风险通话，请立即确认安全！时间:{time_str}"
        msg = f"【反诈预警】您的家人 {name} 正在进行 {risk_level} 风险通话，系统判定为高危，请立即干预！"
        
        # 实际对接阿里云/腾讯云短信API
        # ... API 调用代码 ...
        
        # 开发环境直接打日志
        logger.critical(f"📨 [SMS SENT] To: {phone} | Content: {msg}")
        return True
    except Exception as e:
        logger.error(f"Failed to send alert SMS to {phone}: {e}")
        return False