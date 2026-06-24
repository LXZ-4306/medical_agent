#!/usr/bin/env python3
"""
医学智能体C端应用 - 主入口
多学科AI会诊助手 (MDT AI Assistant)
"""
import streamlit as st
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入核心模块
from core.langgraph_workflow import build_medical_workflow
from agents.intent_decomposer import IntentDecomposer
from knowledge_graph.graph_builder import MedicalGraphBuilder

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="MDT医学助手 - 多学科AI会诊",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 自定义CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #2e86c1;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fdedec;
        border-left: 5px solid #e74c3c;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .opinion-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .specialty-tag {
        display: inline-block;
        background-color: #2e86c1;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .confidence-high {
        color: #27ae60;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    .stChatMessage {
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 应用标题
# ============================================================
st.markdown('<div class="main-header">🏥 多学科AI医学助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于知识图谱 + 多智能体协同（模拟MDT会诊）</div>', unsafe_allow_html=True)

# ============================================================
# 免责声明
# ============================================================
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ 免责声明：</strong><br>
    本AI助手为信息参考工具，所有建议<strong>不构成医疗诊断或治疗方案</strong>。
    生成内容基于权威医学知识库模拟，但<strong>不能替代专业医疗</strong>。
    如有不适，请及时前往正规医疗机构就诊。
</div>
""", unsafe_allow_html=True)

# ============================================================
# 初始化缓存资源
# ============================================================
@st.cache_resource(show_spinner="🏗️ 初始化医学智能体...")
def load_workflow():
    """加载LangGraph工作流"""
    logger.info("🔄 加载医学工作流...")
    try:
        workflow = build_medical_workflow(use_kg=True)
        logger.info("✅ 工作流加载成功")
        return workflow
    except Exception as e:
        logger.error(f"❌ 工作流加载失败: {e}")
        st.error(f"智能体初始化失败: {e}")
        return None

@st.cache_resource(show_spinner="🧠 加载意图解析器...")
def load_intent_decomposer():
    """加载意图解析器"""
    try:
        return IntentDecomposer()
    except Exception as e:
        logger.error(f"❌ 意图解析器加载失败: {e}")
        return None

@st.cache_resource(show_spinner="Initializing knowledge graph...")
def initialize_knowledge_graph():
    """初始化知识图谱（仅建立schema，不清除已有数据）"""
    try:
        builder = MedicalGraphBuilder()
        stats = builder.build(include_pubmed=False)
        logger.info(f"Knowledge graph ready: {stats}")
        return stats
    except Exception as e:
        logger.warning(f"Knowledge graph init failed: {e}")
        return None

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("🧠 核心能力")
    st.markdown("""
    - ✅ **多学科会诊**：模拟MDT讨论，综合多专科意见
    - ✅ **知识图谱驱动**：基于DrugBank、Orphanet等开源数据
    - ✅ **复合意图解析**：自动分解复杂医学问题
    - ✅ **置信度评估**：显示各环节可信度
    """)
    
    st.header("📚 知识增长")
    st.markdown("""
    - 🧠 **LLM实体提取**：从每次对话中提取医学实体
    - 📈 **增量学习**：图谱随使用不断丰富
    - 🔗 **关系发现**：自动识别实体间的医学关系
    - 📄 PubMed文献（可选）
    """)
    
    # 知识图谱状态
    st.divider()
    if st.button("Refresh graph stats", type="secondary"):
        builder = MedicalGraphBuilder()
        stats = builder.get_statistics()
        st.session_state.kg_stats = stats
        st.rerun()
    
    # 配置选项
    st.divider()
    st.header("⚙️ 配置")
    temperature = st.slider(
        "模型温度 (越低越确定)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    show_details = st.checkbox("显示会诊详情", value=True)
    
    st.divider()
    st.caption("v1.0.0 | 仅供研究参考")

# ============================================================
# 初始化会话状态
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "您好！我是多学科AI医学助手。\n\n请描述您的健康问题（症状、不适或用药疑问），我会调动各专科智能体为您提供参考建议。"
    })

if "workflow" not in st.session_state:
    st.session_state.workflow = load_workflow()

if "kg_stats" not in st.session_state:
    st.session_state.kg_stats = initialize_knowledge_graph()

if "waiting_response" not in st.session_state:
    st.session_state.waiting_response = False

# ============================================================
# 加载资源
# ============================================================
workflow = st.session_state.workflow
kg_stats = st.session_state.kg_stats

if workflow is None:
    st.error("❌ 智能体未就绪，请检查配置和依赖")
    st.stop()

# ============================================================
# 显示知识图谱状态
# ============================================================
if kg_stats:
    labels = [("💊 药物", "drug_count"), ("🏥 疾病", "disease_count"),
              ("🤒 症状", "symptom_count"), ("🔗 关系", "relationship_count")]
    cols = st.columns(len(labels))
    for i, (label, key) in enumerate(labels):
        with cols[i]:
            st.metric(label, kg_stats.get(key, 0))
    st.divider()

# ============================================================
# 聊天界面
# ============================================================
# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# 用户输入处理
# ============================================================
if prompt := st.chat_input("请输入您的健康问题...", disabled=st.session_state.waiting_response):
    # 添加用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 设置等待状态
    st.session_state.waiting_response = True
    
    # 显示助手响应
    with st.chat_message("assistant"):
        # 创建状态容器
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            # 显示处理进度
            with status_placeholder.container():
                st.info("🧠 AI医生团队正在会诊...")
                
                # 第一步：意图解析
                st.write("📋 正在解析医学意图...")
                decomposer = load_intent_decomposer()
                if decomposer:
                    intent_info = decomposer.decompose(prompt)
                    st.write(f"  - 识别到 {len(intent_info.get('intents', []))} 个核心意图")
                    st.write(f"  - 紧急程度: {intent_info.get('urgency', '一般')}")
                
                # 第二步：执行工作流
                st.write("🏥 启动MDT多学科会诊...")
                st.write("  - 正在进行分诊...")
                st.write("  - 各专科医生正在讨论...")
                
                # 构建初始状态
                initial_state = {
                    "user_query": prompt,
                    "triage_result": "",
                    "specialist_opinions": [],
                    "final_diagnosis": "",
                    "need_human_review": False,
                    "confidence_score": 0.0
                }
                
                # 执行工作流
                result = workflow.invoke(initial_state)

                # 增量写入知识图谱
                final_text = result.get("final_diagnosis", "")
                if final_text:
                    try:
                        builder = MedicalGraphBuilder()
                        ingested = builder.ingest_conversation(prompt, final_text)
                        if ingested > 0:
                            logger.info(f"KG updated with {ingested} new entities")
                            st.session_state.kg_stats = builder.get_statistics()
                    except Exception as e:
                        logger.debug(f"Ingestion non-critical: {e}")

                # 获取最终结果
                final_diagnosis = result.get("final_diagnosis", "")
                confidence = result.get("confidence_score", 0.5)
                need_review = result.get("need_human_review", False)
                
                # 更新状态
                status_placeholder.success("✅ 会诊完成")
            
            # 显示最终结果
            with response_placeholder.container():
                # 置信度指示器
                if confidence >= 0.8:
                    conf_class = "confidence-high"
                    conf_text = "高置信度"
                elif confidence >= 0.6:
                    conf_class = "confidence-medium"
                    conf_text = "中等置信度"
                else:
                    conf_class = "confidence-low"
                    conf_text = "低置信度（建议人工复核）"
                
                st.markdown(f"**置信度**: <span class='{conf_class}'>{conf_text} ({confidence:.0%})</span>", 
                           unsafe_allow_html=True)
                
                if need_review:
                    st.warning("⚠️ 此病例较为复杂，建议进一步咨询专业医生")
                
                # 显示诊断结果
                st.markdown("---")
                st.markdown(final_diagnosis)
                
                # 显示会诊详情（折叠）
                if show_details:
                    with st.expander("🔍 查看会诊详情"):
                        # 显示分诊结果
                        triage_str = result.get("triage_result", "{}")
                        try:
                            triage_data = json.loads(triage_str)
                            st.subheader("🏥 分诊结果")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**主要科室**: {triage_data.get('primary_department', '未指定')}")
                                st.write(f"**紧急程度**: {triage_data.get('urgency_level', '一般')}")
                            with col2:
                                st.write(f"**辅助科室**: {', '.join(triage_data.get('secondary_departments', []))}")
                                st.write(f"**建议行动**: {triage_data.get('suggested_action', '')}")
                        except:
                            pass
                        
                        # 显示各专科意见
                        opinions = result.get("specialist_opinions", [])
                        if opinions:
                            st.subheader("🧑‍⚕️ 专科会诊意见")
                            for opinion in opinions:
                                with st.container():
                                    specialty = opinion.get("specialty", "未知科室")
                                    content = opinion.get("opinion", "")
                                    st.markdown(f"**{specialty}**")
                                    st.write(content[:500] + "..." if len(content) > 500 else content)
                                    st.divider()
                
                # 添加免责声明
                st.markdown("---")
                st.caption("⚠️ 本内容由AI生成，仅供参考，不构成医疗建议。如有不适请及时就医。")
            
            # 保存到会话
            full_response = final_diagnosis
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"❌ 处理过程中出现错误: {str(e)}"
            status_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            logger.error(f"工作流执行错误: {e}", exc_info=True)
        
        finally:
            st.session_state.waiting_response = False

# ============================================================
# 页脚
# ============================================================
st.divider()
st.caption("🏥 MDT医学助手 | 基于LangGraph + 知识图谱 | 仅供研究参考")