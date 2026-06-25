# 🏥 MDT 医学助手 — 多学科 AI 会诊系统

基于 **LangGraph + Neo4j 知识图谱** 的多智能体协同医学助手，模拟临床 MDT（Multi-Disciplinary Team）会诊流程，提供医学信息参考。

> ⚠️ **免责声明**：本系统仅供研究参考，所有 AI 生成内容不构成医疗诊断或治疗方案，不能替代专业医师判断。

## 核心架构

```
用户提问 → 意图解析 → 智能分诊 → 多专科会诊 → 主诊整合 → 输出建议
           │                         │
           └── LLM 实体提取 ──────────┴──→ Neo4j 知识图谱（增量学习）
```

### 智能体角色

| Agent | 职责 |
|-------|------|
| **IntentDecomposer** | 解析用户复杂医学问题，拆分为多个子意图 |
| **TriageAgent** | 根据症状和意图进行智能分诊，确定主/辅科室 |
| **SpecialistAgent** ×6 | 心血管科、内科、临床药学、神经科、消化科、呼吸科专科会诊 |
| **AttendingPhysicianAgent** | 综合各专科意见，生成最终建议和置信度评估 |

## 技术栈

- **前端框架**：Streamlit
- **工作流引擎**：LangGraph（状态图编排多 Agent 协作）
- **LLM 接口**：LangChain（OpenAI API 兼容，默认 DeepSeek，可选智谱 GLM）
- **知识图谱**：Neo4j（存储实体与关系） + ChromaDB（向量检索）
- **LLM 驱动的知识提取**：从对话中自动抽取医学实体（药物、疾病、症状等）并写入图谱
- **外部知识源**：PubMed 文献检索、中国药典 (ChP) 药物查询、开源医学数据库（DrugBank、Orphanet）

## 项目结构

```
medical_agent/
├── app.py                     # Streamlit 主入口
├── config.py                  # 全局配置
├── requirements.txt           # Python 依赖
├── agents/                    # 智能体实现
│   ├── base_agent.py          # Agent 基类（LLM 初始化、JSON 解析）
│   ├── intent_decomposer.py   # 意图分解器
│   ├── triage_agent.py        # 分诊 Agent
│   ├── specialist_agent.py    # 专科 Agent
│   └── attending_agent.py     # 主诊整合 Agent
├── core/
│   ├── state.py               # 工作流状态定义（TypedDict）
│   └── langgraph_workflow.py  # LangGraph 工作流编排
├── knowledge_graph/
│   ├── neo4j_client.py        # Neo4j 客户端封装
│   ├── graph_builder.py       # 图谱构建与对话摄入
│   └── open_sources.py        # LLM 实体提取 + 中国药典 + PubMed
├── utils/                     # 日志、校验工具
├── tests/                     # 测试用例
└── data/                      # 数据缓存目录
```

## 快速开始

### 1. 环境准备

- Python ≥ 3.10
- Neo4j 数据库（本地或远程）
- LLM API Key（DeepSeek / OpenAI / 智谱）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env` 文件并填入你的 API Key：

```env
# LLM API（OpenAI 兼容接口，示例为 DeepSeek）
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.deepseek.com/v1

# Neo4j 图数据库
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# 模型配置
DEEPSEEK_MODEL=deepseek-chat
TEMPERATURE=0.3
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

## 工作流说明

1. **意图解析**：将用户问题拆解为多个医学子意图（如症状查询、用药咨询、疾病科普等）
2. **智能分诊**：根据意图确定主要科室和辅助科室，评估紧急程度
3. **专科会诊**：调用 2-3 个相关专科 Agent 并行分析，各 Agent 结合知识图谱上下文给出专业意见
4. **主诊整合**：Attending 综合各专科意见，输出结构化建议和置信度评分
5. **增量学习**：每次对话后通过 LLM 提取医学实体，自动写入 Neo4j 知识图谱

## 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | LLM 模型 | `deepseek-chat` |
| `TEMPERATURE` | 生成温度 (0-1) | `0.3` |
| `MAX_TOKENS` | 最大输出 token | `2000` |
| `ENABLE_PUBMED` | 启用 PubMed 文献检索 | `false` |
| `NEO4J_URI` | Neo4j 连接地址 | `bolt://localhost:7687` |

## License

仅供学术研究使用。
