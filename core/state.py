"""
状态管理 - 定义多Agent协作的状态结构
"""
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Literal


class MedicalState(TypedDict, total=False):
    """
    MDT多学科会诊的状态定义
    
    用于LangGraph工作流中传递数据
    """
    # 用户输入
    user_query: str                     # 用户原始问题
    
    # 意图解析
    intent_info: Dict[str, Any]         # 意图解析结果
    query_summary: str                  # 查询摘要
    
    # 分诊结果
    triage_result: str                  # JSON格式的分诊结果
    primary_department: str             # 主要科室
    secondary_departments: List[str]    # 辅助科室
    urgency_level: Literal["紧急", "一般", "可择期"]  # 紧急程度
    
    # 专科会诊
    specialist_opinions: List[Dict[str, Any]]  # 各专科意见列表
    
    # 最终结果
    final_diagnosis: str                # 最终诊断建议
    confidence_score: float             # 置信度 (0-1)
    need_human_review: bool             # 是否需要人工审核
    
    # 元信息
    timestamp: str                      # 时间戳
    error: Optional[str]                # 错误信息