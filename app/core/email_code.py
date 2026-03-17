"""
邮箱验证码服务
功能：发送邮箱验证码并存储到 Redis
"""
import random
from app.core.redis import get_redis
from app.core.logger import get_logger
from app.services.email_service import email_service

logger = get_logger(__name__)


async def send_email_code(email: str) -> bool:
    """
    发送邮箱验证码
    
    Args:
        email: 目标邮箱地址
        
    Returns:
        bool: 发送是否成功
    """
    redis = await get_redis()
    
    # 1. 生成 6 位随机验证码
    code = str(random.randint(100000, 999999))
    
    # 2. 存入 Redis，设置 5 分钟过期
    key = f"email_code:{email}"
    try:
        await redis.setex(key, 300, code)
        
        # 3. 发送邮件
        subject = "【AI反诈系统】验证码"
        body = f"""
尊敬的用户：

您的验证码是：{code}

该验证码5分钟内有效，请勿泄露给他人。

如非本人操作，请忽略此邮件。

---
AI反诈系统
        """
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 500px; margin: 0 auto; padding: 20px; }}
        .code-box {{ 
            background-color: #f0f0f0; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center;
            margin: 20px 0;
        }}
        .code {{ font-size: 32px; font-weight: bold; color: #dc3545; letter-spacing: 5px; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>AI反诈系统 - 验证码</h2>
        <p>尊敬的用户：</p>
        <p>您的验证码是：</p>
        <div class="code-box">
            <div class="code">{code}</div>
        </div>
        <p>该验证码 <strong>5分钟内</strong> 有效，请勿泄露给他人。</p>
        <p>如非本人操作，请忽略此邮件。</p>
        <div class="footer">
            <p>---</p>
            <p>AI反诈系统</p>
        </div>
    </div>
</body>
</html>
        """
        
        success = await email_service.send_email(email, subject, body, html_body)
        
        if success:
            # 4. 在控制台打印（方便开发测试）
            print("\n" + "="*50)
            print(f"📧 [邮箱验证码发送成功]")
            print(f"➡️ 目标邮箱: {email}")
            print(f"🔑 验证码: {code}")
            print("="*50 + "\n")
            
            logger.info(f"Email code {code} sent to {email}")
            return True
        else:
            logger.error(f"Failed to send email to {email}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send email code: {e}")
        return False


async def verify_email_code(email: str, code: str) -> bool:
    """
    校验邮箱验证码
    
    Args:
        email: 邮箱地址
        code: 用户输入的验证码
        
    Returns:
        bool: 验证是否通过
    """
    # 如果输入的验证码是 "666666"，则直接放行（万能验证码）
    if code == "666666":
        logger.info(f"Universal code used for {email}")
        return True

    redis = await get_redis()
    key = f"email_code:{email}"
    
    try:
        stored_code = await redis.get(key)
        if stored_code and stored_code == code:
            # 校验成功后立即删除，防止被重复利用
            await redis.delete(key)
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to verify email code from Redis: {e}")
        return False
