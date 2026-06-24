"""
LangGraph工作流定义 - 构建多Agent协同流程
"""
from typing import Dict, Any
import logging
from langgraph.graph import StateGraph, END

from .state import MedicalState
from agents import (
    TriageAgent,
    SpecialistAgent,
    AttendingPhysicianAgent,
    IntentDecomposer
)
from knowledge_graph import Neo4jClient

logger = logging.getLogger(__name__)


def build_medical_workflow(use_kg: bool = True) -> StateGraph:
    """
    构建MDT多学科会诊工作流
    
    Args:
        use_kg: 是否使用知识图谱
        
    Returns:
        编译好的LangGraph工作流
    """
    logger.info("🏗️ 构建MDT医学工作流...")
    
    # 初始化组件
    intent_decomposer = IntentDecomposer()
    triage_agent = TriageAgent()
    
    # 知识图谱客户端
    kg_client = Neo4jClient() if use_kg else None
    
    # 初始化各专科Agent
    specialist_agents = {
        "心血管科": SpecialistAgent("心血管科", kg_client),
        "内科": SpecialistAgent("内科", kg_client),
        "临床药学": SpecialistAgent("临床药学", kg_client),
        "神经科": SpecialistAgent("神经科", kg_client),
        "消化科": SpecialistAgent("消化科", kg_client),
        "呼吸科": SpecialistAgent("呼吸科", kg_client),
    }
    
    attending_agent = AttendingPhysicianAgent()
    
    # 创建状态图
    workflow = StateGraph(MedicalState)
    
    # 定义节点
    workflow.add_node("intent_decompose", 
                      lambda s: _decompose_intent(s, intent_decomposer))
    workflow.add_node("triage", 
                      lambda s: _triage(s, triage_agent))
    workflow.add_node("specialist_consult", 
                      lambda s: _consult_specialists(s, specialist_agents))
    workflow.add_node("attending_synthesis", 
                      lambda s: _synthesize(s, attending_agent))
    
    # 定义流程
    workflow.set_entry_point("intent_decompose")
    workflow.add_edge("intent_decompose", "triage")
    workflow.add_edge("triage", "specialist_consult")
    workflow.add_edge("specialist_consult", "attending_synthesis")
    workflow.add_edge("attending_synthesis", END)
    
    logger.info("✅ 工作流构建完成")
    return workflow.compile()


def _decompose_intent(state: MedicalState, decomposer: IntentDecomposer) -> MedicalState:
    """意图解析节点"""
    return decomposer.invoke(state)


def _triage(state: MedicalState, triage_agent: TriageAgent) -> MedicalState:
    """分诊节点"""
    return triage_agent.invoke(state)


def _consult_specialists(state: MedicalState, 
                         specialist_agents: Dict) -> MedicalState:
    """
    专科会诊节点 - 根据分诊结果调用对应专科
    """
    import json
    
    triage_str = state.get("triage_result", "{}")
    try:
        triage_data = json.loads(triage_str) if triage_str else {}
    except:
        triage_data = {}
    
    # 获取需要会诊的科室
    departments = []
    if triage_data.get("primary_department"):
        departments.append(triage_data["primary_department"])
    departments.extend(triage_data.get("secondary_departments", []))
    
    # 如果指定了科室列表，优先使用
    specified = state.get("suggested_departments", [])
    if specified:
        departments = specified + departments
    
    # 去重并限制数量
    departments = list(dict.fromkeys(departments))[:3]
    
    # 如果没有任何科室，默认使用内科
    if not departments:
        departments = ["内科"]
    
    logger.info(f"🏥 启动会诊: {departments}")
    
    # 调用各专科Agent
    state["specialist_opinions"] = []
    
    for dept in departments:
        if dept in specialist_agents:
            agent = specialist_agents[dept]
            state = agent.invoke(state)
        else:
            # 如果科室不存在，记录警告
            logger.warning(f"⚠️ 未知科室: {dept}")
    
    return state


def _synthesize(state: MedicalState, attending_agent: AttendingPhysicianAgent) -> MedicalState:
    """主诊整合节点"""
    return attending_agent.invoke(state)