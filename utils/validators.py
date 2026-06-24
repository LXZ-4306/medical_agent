"""
输入验证工具
"""
import re
from typing import Tuple, Optional


def validate_medical_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    验证医学查询是否有效
    
    Args:
        query: 用户查询文本
        
    Returns:
        (是否有效, 错误信息)
    """
    if not query or not query.strip():
        return False, "请输入健康问题"
    
    # 检查长度
    if len(query.strip()) < 2:
        return False, "请更详细地描述您的健康问题"
    
    if len(query.strip()) > 2000:
        return False, "问题描述过长，请精简到2000字以内"
    
    # 检查是否包含敏感或危险内容（简单过滤）
    dangerous_patterns = [
        r"自杀",
        r"自残",
        r"杀人",
        r"毒品",
        r"违禁",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "您的问题涉及敏感内容，请联系专业心理咨询或紧急热线"
    
    return True, None


def sanitize_input(text: str) -> str:
    """
    净化输入，移除潜在危险字符
    """
    if not text:
        return ""
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    
    # 限制长度
    return text.strip()[:2000]