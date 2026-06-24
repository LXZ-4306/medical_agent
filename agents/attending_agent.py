"""
主诊智能体 - 模拟MDT多学科会诊整合
"""
from typing import Dict, Any, List
import json
import logging
from .base_agent import BaseMedicalAgent

logger = logging.getLogger(__name__)


class AttendingPhysicianAgent(BaseMedicalAgent):
    """主诊医生Agent - 整合各专科意见，给出最终建议"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.prompt_template = """
        你是三甲医院MDT（多学科会诊）的主诊医生。请综合分析各专科意见，给出最终的诊疗建议。
        
        ## 患者问题
        {query}
        
        ## 分诊结果
        {triage_result}
        
        ## 各专科会诊意见
        {specialist_opinions}
        
        ## 输出要求
        请以清晰、结构化的方式呈现综合会诊结论：
        
        ### 📋 综合诊断结论
        (基于多学科讨论的最终判断)
        
        ### 💊 推荐治疗方案（按优先级排序）
        1. 首选方案:
        2. 备选方案:
        3. 需进一步评估:
        
        ### 🏥 就医建议
        - 建议挂什么科室
        - 是否需要立即就医
        - 就医前准备
        
        ### ⚠️ 注意事项
        - 需要关注的症状变化
        - 禁忌事项
        - 需要复查的指标
        
        ### 📊 会诊置信度
        - 综合置信度: [0-100%]
        - 主要不确定性因素:
        
        ---
        ## 🔔 重要免责声明
        *本建议仅基于AI模拟的MDT会诊，仅供参考，不能替代专业医疗诊断。*
        *如有不适，请及时前往正规医疗机构就诊。*
        """
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        整合各专科意见
        
        Args:
            state: 包含所有专科意见的状态
            
        Returns:
            更新后的状态字典
        """
        query = state.get("user_query", "")
        triage_result = state.get("triage_result", "{}")
        opinions = state.get("specialist_opinions", [])
        
        if not opinions:
            logger.warning("⚠️ 未收到任何专科意见，直接调用LLM")
            opinions = [{
                "specialty": "通用医学",
                "opinion": "请提供详细的症状描述以便进一步分析。",
                "references": ""
            }]
        
        try:
            # 格式化专科意见
            formatted_opinions = self._format_opinions(opinions)
            
            # 创建提示词
            prompt = self.create_prompt(self.prompt_template)
            chain = prompt | self.llm
            
            # 调用LLM
            response = chain.invoke({
                "query": query,
                "triage_result": triage_result,
                "specialist_opinions": formatted_opinions
            })
            
            # 存储最终诊断
            state["final_diagnosis"] = response.content
            
            # 设置置信度（从回答中提取或默认）
            state["confidence_score"] = self._extract_confidence(response.content)
            
            # 设置是否需要人工审核
            state["need_human_review"] = state.get("confidence_score", 0.7) < 0.7
            
            logger.info(f"✅ MDT会诊完成，置信度: {state.get('confidence_score')}")
            
        except Exception as e:
            logger.error(f"❌ 主诊Agent错误: {e}")
            state["final_diagnosis"] = "抱歉，MDT会诊过程中出现错误，请重新尝试。"
        
        return state
    
    def _format_opinions(self, opinions: List[Dict]) -> str:
        """格式化专科意见供LLM阅读"""
        formatted = []
        for idx, opinion in enumerate(opinions, 1):
            specialty = opinion.get("specialty", "未知科室")
            content = opinion.get("opinion", "无意见")
            formatted.append(f"### 专科{idx}: {specialty}\n{content}\n")
        return "\n".join(formatted)
    
    def _extract_confidence(self, text: str) -> float:
        """从回答中提取置信度分数"""
        import re
        
        # 搜索置信度模式
        patterns = [
            r'置信度[：:]\s*(\d+)%',
            r'综合置信度[：:]\s*(\d+)%',
            r'confidence[：:]\s*(\d+)%'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1)) / 100
                return min(max(score, 0), 1)  # 限制在0-1之间
        
        # 默认置信度
        return 0.7