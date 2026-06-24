"""
核心模块 - LangGraph工作流和状态管理
"""
from .state import MedicalState
from .langgraph_workflow import build_medical_workflow

__all__ = [
    'MedicalState',
    'build_medical_workflow'
]