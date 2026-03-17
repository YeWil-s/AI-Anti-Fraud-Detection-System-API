"""
初始化反诈法律法规向量库
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.services.vector_db_service import vector_db
from app.core.logger import get_logger

logger = get_logger(__name__)

# 模拟整理好的法律法规数据
LAW_DATA = [
    {
        "id": "law_001",
        "title": "《中华人民共和国反电信网络诈骗法》第十二条",
        "content": "银行业金融机构、非银行支付机构应当为客户提供交易限额管理、防范电信网络诈骗等服务，发现异常开户或者可疑交易的，应当采取核实交易情况、限制交易等必要防范措施。",
        "tag": "银行转账"
    },
    {
        "id": "law_002",
        "title": "《中华人民共和国反电信网络诈骗法》第三十一条",
        "content": "任何单位和个人不得非法买卖、出租、出借电话卡、物联网卡、电信线路、短信端口、银行账户、支付账户、互联网账号等，不得提供实名核验帮助；不得假冒他人身份或者虚构代理关系开立上述卡、账户、账号等。",
        "tag": "买卖银行卡,帮信罪"
    },
    {
        "id": "law_003",
        "title": "最高法最高检公安部关于办理电信网络诈骗等刑事案件适用法律若干问题的意见",
        "content": "冒充国家机关工作人员拨打电话、发送短信进行诈骗，构成诈骗罪的，应当依照刑法第二百六十六条的规定定罪处罚。同时构成招摇撞骗罪的，依照处罚较重的规定定罪处罚。",
        "tag": "冒充公检法"
    }
]

def init_laws():
    print("====== 开始初始化反诈法律法规库 ======")
    
    documents = []
    metadatas = []
    ids = []
    
    for item in LAW_DATA:
        ids.append(item["id"])
        # 将标题和内容合并，有利于被向量化和检索
        documents.append(f"{item['title']}\n{item['content']}")
        metadatas.append({
            "title": item["title"],
            "fraud_type": item["tag"], # 作为标签关联
            "item_type": "law"
        })
        
    try:
        vector_db.add_data("anti_fraud_laws", documents, metadatas, ids)
        print(f"成功将 {len(ids)} 条法律法规灌入向量数据库！")
    except Exception as e:
        print(f"写入失败: {e}")

    # 简单测试一下检索
    print("\n====== 检索测试 ======")
    query = "刚才有个自称公安局的让我把钱转到安全账户"
    print(f"用户遇到情况: {query}")
    
    results = vector_db.search_similar("anti_fraud_laws", query, top_k=1)
    if results:
        print("\n【匹配到的法律】:")
        print(f"标题: {results[0]['metadata']['title']}")
        print(f"内容: {results[0]['content']}")
        print(f"相似度(越小越好): {results[0]['distance']}")

if __name__ == "__main__":
    init_laws()