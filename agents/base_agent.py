"""
智能体基类
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BaseMedicalAgent(ABC):
    """所有医学智能体的基类"""
    
    def __init__(self,
                 model_name: str = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000):
        if model_name is None:
            model_name = os.getenv("DEEPSEEK_MODEL", os.getenv("MODEL_NAME", "deepseek-chat"))
        """
        初始化基础Agent
        
        Args:
            model_name: 模型名称
            temperature: 温度参数（0-1），越低越确定
            max_tokens: 最大输出token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化LLM
        self._init_llm()
    
    def _init_llm(self):
        """初始化大语言模型"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if api_key and base_url:
            # 使用OpenAI或兼容API
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"🤖 初始化LLM: {self.model_name}")
        else:
            # 尝试使用智谱GLM
            api_key = os.getenv("ZHIPU_API_KEY")
            if api_key:
                from langchain_community.chat_models import ChatZhipuAI
                self.llm = ChatZhipuAI(
                    model="glm-4-plus",
                    temperature=self.temperature,
                    api_key=api_key
                )
                logger.info("🤖 初始化LLM: 智谱GLM-4")
            else:
                raise ValueError("请设置 OPENAI_API_KEY 或 ZHIPU_API_KEY")
    
    def create_prompt(self, template: str) -> ChatPromptTemplate:
        """创建提示词模板"""
        return ChatPromptTemplate.from_template(template)
    
    @abstractmethod
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行Agent的核心逻辑"""
        pass
    
    def _parse_json_output(self, content: str) -> Dict:
        """解析JSON输出"""
        import json
        try:
            # 尝试提取JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # 如果解析失败，返回空字典
        logger.warning(f"⚠️ JSON解析失败: {content[:100]}...")
        return {}