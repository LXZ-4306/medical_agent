"""
专科智能体 - 提供各专科的专业意见
"""
from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseMedicalAgent

logger = logging.getLogger(__name__)


class SpecialistAgent(BaseMedicalAgent):
    """专科Agent - 提供特定科室的专业诊断意见"""
    
    def __init__(self, 
                 specialty: str, 
                 knowledge_graph_client = None,
                 **kwargs):
        """
        初始化专科Agent
        
        Args:
            specialty: 专科名称（如"心血管科"）
            knowledge_graph_client: 知识图谱客户端
        """
        super().__init__(**kwargs)
        self.specialty = specialty
        self.kg_client = knowledge_graph_client
        
        # 专科特定提示词
        self.prompt_template = """
        你是一位{specialty}的资深主任医师。请根据以下信息提供专业的诊断意见。
        
        ## 患者主诉
        {query}
        
        ## 分诊信息
        {triage_info}
        
        ## 知识库参考（来自权威医学数据库）
        {knowledge_refs}
        
        ## 输出要求
        请以结构化的方式输出你的专业意见：
        
        ### 1. 初步判断
        - 可能的诊断方向（列出2-3个可能性）
        - 每个可能性的依据
        
        ### 2. 建议检查
        - 必要的检查项目
        - 检查的目的
        
        ### 3. 初步治疗建议
        - 生活建议
        - 药物考虑（如有）
        - 需要立即就医的指征
        
        ### 4. 风险评估
        - 潜在风险
        - 警示信号
        
        ### 5. 置信度
        - 对诊断的置信度 (0-100%)
        - 限制和不确定性说明
        
        ---
        请以专业、严谨的态度回答，同时确保患者能够理解。
        """
    
    def _get_knowledge_refs(self, query: str) -> str:
        """
        从知识图谱获取相关知识，未命中时自动查询中国药典
        不依赖任何硬编码词表，完全动态
        """
        if not self.kg_client:
            return "暂无知识库参考数据"

        try:
            refs = []
            # 直接用完整查询搜索KG（Neo4j CONTAINS模糊匹配）
            results = self.kg_client.query_related(query, limit=5)
            for result in results:
                ref_text = self._format_kg_result(result)
                if ref_text:
                    refs.append(ref_text)

            if not refs:
                # KG无匹配，用LLM识别查询中的药物名，从中国药典获取
                drug_names = self._detect_drug_names(query)
                for drug in drug_names[:2]:
                    try:
                        from knowledge_graph.graph_builder import MedicalGraphBuilder
                        builder = MedicalGraphBuilder()
                        ingested = builder.ingest_chinese_drug(drug)
                        if ingested > 0:
                            logger.info(f"Auto-ingested drug from ChP: {drug}")
                            results = self.kg_client.query_related(drug, limit=3)
                            for result in results:
                                ref_text = self._format_kg_result(result)
                                if ref_text:
                                    refs.append(ref_text)
                    except Exception as e:
                        logger.debug(f"Auto-ingest failed for {drug}: {e}")

            return "\n".join(refs[:5]) if refs else "未找到相关知识库参考"

        except Exception as e:
            logger.warning(f"知识图谱查询失败: {e}")
            return "知识库查询暂时不可用"

    def _detect_drug_names(self, query: str) -> List[str]:
        """用LLM识别查询中的药物名称，无硬编码词表"""
        import json
        try:
            prompt = self.create_prompt(
                "从以下文本中识别所有药物名称（西药或中成药），如果没有则返回空数组。"
                "只输出JSON数组，如: [\"阿莫西林\", \"布洛芬\"]\n\n文本: {text}"
            )
            chain = prompt | self.llm
            response = chain.invoke({"text": query[:200]})
            content = response.content.strip()
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        return []
    
    def _format_kg_result(self, result: Dict) -> str:
        """格式化知识图谱查询结果"""
        try:
            # 提取节点名称和关系
            parts = []
            if "a" in result and "name" in result["a"]:
                parts.append(f"实体: {result['a']['name']}")
            if "b" in result and "name" in result["b"]:
                parts.append(f"关联: {result['b']['name']}")
            if "r" in result and "type" in result["r"]:
                parts.append(f"关系: {result['r']['type']}")
            
            return " - ".join(parts) if parts else ""
        except:
            return ""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行专科诊断
        
        Args:
            state: 包含用户问题和分诊结果的状态
            
        Returns:
            更新后的状态字典
        """
        query = state.get("user_query", "")
        triage_info = state.get("triage_result", "{}")
        
        if not query:
            logger.error(f"❌ {self.specialty}Agent: 未收到用户问题")
            return state
        
        try:
            # 获取知识库参考
            knowledge_refs = self._get_knowledge_refs(query)
            
            # 创建提示词
            prompt = self.create_prompt(self.prompt_template)
            chain = prompt | self.llm
            
            # 调用LLM
            response = chain.invoke({
                "specialty": self.specialty,
                "query": query,
                "triage_info": triage_info,
                "knowledge_refs": knowledge_refs
            })
            
            # 构建意见
            opinion = {
                "specialty": self.specialty,
                "opinion": response.content,
                "references": knowledge_refs,
                "timestamp": "2026-06-22"  # 可以替换为实际时间
            }
            
            # 存储到状态
            if "specialist_opinions" not in state:
                state["specialist_opinions"] = []
            state["specialist_opinions"].append(opinion)
            
            logger.info(f"🧑‍⚕️ {self.specialty}Agent 诊断完成")
            
        except Exception as e:
            logger.error(f"❌ {self.specialty}Agent错误: {e}")
        
        return state