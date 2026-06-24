"""
全局配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """应用配置"""
    
    # 模型配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
    
    # 模型参数
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # Neo4j配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
    
    # 应用配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # 知识库配置
    ENABLE_PUBMED = os.getenv("ENABLE_PUBMED", "false").lower() == "true"
    KNOWLEDGE_CACHE_DIR = os.getenv("KNOWLEDGE_CACHE_DIR", "./data/cache")


# 打印配置信息（调试用）
if __name__ == "__main__":
    print("=== 配置信息 ===")
    for key, value in Config.__dict__.items():
        if not key.startswith("_") and not callable(value):
            # 隐藏敏感信息
            if "KEY" in key and value:
                value = "***已设置***"
            print(f"{key}: {value}")