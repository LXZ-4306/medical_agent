"""
意图解析器 - 分解复合医学问题
"""
from typing import Dict, List, Any
import json
import logging
from .base_agent import BaseMedicalAgent

logger = logging.getLogger(__name__)


class IntentDecomposer(BaseMedicalAgent):
    """医学意图解析器 - 将复杂问题分解为子任务"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.prompt_template = """
        你是一位医学信息学专家。请分析以下用户问题的复合意图，将其分解为独立的医学子任务。
        
        ## 用户问题
        {query}
        
        ## 解析要求
        1. 识别所有隐含的医学意图
        2. 判断问题的紧急程度
        3. 判断是否需要多学科协作
        
        ## 输出格式（严格JSON）
        {{
            "intents": [
                {{
                    "type": "症状描述|疾病查询|用药咨询|检查解读|就医建议|健康咨询",
                    "content": "具体的意图内容",
                    "priority": 1
                }}
            ],
            "urgency": "紧急|一般|可择期",
            "requires_multidisciplinary": true/false,
            "suggested_departments": ["科室1", "科室2"],
            "query_summary": "对用户问题的简要总结"
        }}
        
        请严格按照JSON格式输出。
        """
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """实现抽象方法 - 调用decompose"""
        query = state.get("user_query", "")
        intent_info = self.decompose(query)
        state["intent_info"] = intent_info
        state["query_summary"] = intent_info.get("query_summary", query[:50])
        return state

    def decompose(self, query: str) -> Dict:
        """
        分解用户问题
        
        Args:
            query: 用户输入的医学问题
            
        Returns:
            分解后的意图结构
        """
        if not query:
            return self._empty_result()
        
        try:
            prompt = self.create_prompt(self.prompt_template)
            chain = prompt | self.llm
            
            response = chain.invoke({"query": query})
            result = self._parse_json_output(response.content)
            
            if result and "intents" in result:
                logger.info(f"🔍 意图解析完成: {len(result['intents'])}个意图")
                return result
            else:
                logger.warning("⚠️ 意图解析失败，使用默认值")
                return self._default_result(query)
        
        except Exception as e:
            logger.error(f"❌ 意图解析错误: {e}")
            return self._default_result(query)
    
    def _default_result(self, query: str) -> Dict:
        """默认解析结果"""
        return {
            "intents": [
                {"type": "健康咨询", "content": query, "priority": 1}
            ],
            "urgency": "一般",
            "requires_multidisciplinary": False,
            "suggested_departments": ["内科"],
            "query_summary": query[:50]
        }
    
    def _empty_result(self) -> Dict:
        """空查询结果"""
        return {
            "intents": [],
            "urgency": "一般",
            "requires_multidisciplinary": False,
            "suggested_departments": [],
            "query_summary": ""
        }