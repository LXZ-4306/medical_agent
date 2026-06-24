"""
分诊智能体 - 负责初步分诊和科室推荐
"""
from typing import Dict, Any
import json
import logging
from .base_agent import BaseMedicalAgent

logger = logging.getLogger(__name__)


class TriageAgent(BaseMedicalAgent):
    """分诊Agent - 根据用户问题推荐科室"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.prompt_template = """
        你是一位经验丰富的三甲医院分诊医生。请根据用户的健康问题，判断应该咨询哪些科室。
        
        ## 可用科室列表
        内科、外科、妇产科、儿科、眼科、耳鼻喉科、皮肤科、精神科、肿瘤科、
        心血管科、神经科、内分泌科、风湿免疫科、骨科、康复科、临床药学、
        急诊科、感染科、消化科、呼吸科、肾内科、血液科、老年医学科
        
        ## 用户问题
        {query}
        
        ## 分诊规则
        1. 如有紧急情况（胸痛、呼吸困难、大出血等），急诊科必须优先
        2. 复杂病例可能需要多个科室会诊
        3. 儿科专门针对14岁以下儿童
        4. 妇产科针对女性生殖系统相关疾病
        
        ## 输出格式（严格JSON）
        {{
            "primary_department": "主要科室名称",
            "secondary_departments": ["辅助科室1", "辅助科室2"],
            "urgency_level": "紧急/一般/可择期",
            "reasoning": "分诊依据的简要说明",
            "suggested_action": "建议用户下一步采取的行动"
        }}
        
        请严格按照JSON格式输出，不要添加额外内容。
        """
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分诊
        
        Args:
            state: 包含user_query的状态字典
            
        Returns:
            更新后的状态字典
        """
        query = state.get("user_query", "")
        if not query:
            logger.error("❌ 分诊Agent: 未收到用户问题")
            state["triage_result"] = "{}"
            return state
        
        try:
            # 创建提示词
            prompt = self.create_prompt(self.prompt_template)
            chain = prompt | self.llm
            
            # 调用LLM
            response = chain.invoke({"query": query})
            content = response.content
            
            # 解析JSON
            result = self._parse_json_output(content)
            
            # 验证结果
            if result and "primary_department" in result:
                state["triage_result"] = json.dumps(result, ensure_ascii=False)
                logger.info(f"🏥 分诊结果: {result.get('primary_department')}")
                logger.debug(f"   理由: {result.get('reasoning')}")
            else:
                # 默认分诊
                default_result = {
                    "primary_department": "内科",
                    "secondary_departments": [],
                    "urgency_level": "一般",
                    "reasoning": "未能明确判断，建议咨询内科",
                    "suggested_action": "建议咨询全科或内科医生"
                }
                state["triage_result"] = json.dumps(default_result, ensure_ascii=False)
                logger.warning("⚠️ 分诊结果解析失败，使用默认值")
        
        except Exception as e:
            logger.error(f"❌ 分诊Agent错误: {e}")
            state["triage_result"] = "{}"
        
        return state