"""
开源医学知识 + LLM驱动的实体提取
"""
import json
import os
from typing import List, Dict
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MedicalEntityExtractor:
    """使用LLM从对话中提取医学实体和关系"""

    EXTRACTION_PROMPT = """你是一位医学知识工程师。从以下医患对话中提取所有医学实体及其关系。

## 对话内容
患者问题: {query}

AI回复摘要: {response}

## 提取规则
1. 识别医学实体，分类为: Drug(药物), Disease(疾病), Symptom(症状), Procedure(检查/治疗), BodyPart(身体部位)
2. 每个实体必须有 name 字段，可选 description 和 category
3. 识别实体间关系: TREATS(治疗), HAS_SYMPTOM(有症状), DIAGNOSES(诊断), CAUSES(引起), CONTRAINDICATES(禁忌), INTERACTS_WITH(相互作用)
4. 仅提取对话中明确提到的实体，不要编造

## 输出格式 (严格JSON)
{{
    "entities": [
        {{"type": "Drug", "name": "阿司匹林", "properties": {{"description": "解热镇痛药", "category": "小分子药物"}}}},
        {{"type": "Disease", "name": "高血压", "properties": {{"description": "血压持续升高", "category": "心血管疾病"}}}}
    ],
    "relationships": [
        {{"from_type": "Drug", "from_name": "阿司匹林", "to_type": "Disease", "to_name": "高血压", "rel_type": "TREATS"}}
    ]
}}

只输出JSON，不要有其他内容。如无可提取实体，返回空的entities和relationships数组。"""

    def __init__(self):
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=2000,
            api_key=api_key,
            base_url=base_url
        )

    def extract_from_conversation(self, query: str, response: str) -> dict:
        """从一次对话中提取医学实体和关系"""
        try:
            from langchain.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(self.EXTRACTION_PROMPT)
            chain = prompt | self.llm
            result = chain.invoke({
                "query": query[:500],
                "response": response[:1000]
            })

            content = result.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                logger.info(f"Extracted {len(parsed.get('entities', []))} entities, "
                            f"{len(parsed.get('relationships', []))} relationships")
                return parsed
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        return {"entities": [], "relationships": []}


class ChineseDrugQuery:
    """基于LLM的中国药典药物查询接口（LLM训练数据含药典知识）"""

    DRUG_QUERY_PROMPT = """你是一位中国药典专家。请根据中国药典(ChP)标准，提供以下药物的结构化信息。

## 查询药物
{drug_name}

## 输出格式 (严格JSON)
{{
    "drug": {{
        "name": "药物通用名（中文）",
        "english_name": "英文名/化学名",
        "category": "药物分类（如：抗生素类、心血管类、中成药等）",
        "dosage_form": "剂型（片剂/注射剂/胶囊等）",
        "indications": ["适应症1", "适应症2"],
        "contraindications": ["禁忌1", "禁忌2"],
        "adverse_reactions": ["不良反应1", "不良反应2"],
        "dosage_adult": "成人常规剂量",
        "dosage_elderly": "老年人剂量调整",
        "drug_interactions": [
            {{"drug": "相互作用药物名", "effect": "相互作用效果", "severity": "轻度/中度/重度"}}
        ],
        "pharmacology": "药理作用简述",
        "pregnancy_category": "孕期安全分级（A/B/C/D/X）",
        "storage": "储存条件"
    }}
}}

只输出JSON，不要其他内容。如果药物不在中国药典中，返回 {{"error": "未收录"}}。"""

    def __init__(self):
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=2000,
            api_key=api_key,
            base_url=base_url
        )

    def query_drug(self, drug_name: str) -> dict:
        """查询单个药物的药典信息，返回结构化数据"""
        try:
            from langchain.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(self.DRUG_QUERY_PROMPT)
            chain = prompt | self.llm
            result = chain.invoke({"drug_name": drug_name})

            content = result.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if "error" not in parsed:
                    logger.info(f"Chinese drug queried: {drug_name}")
                return parsed
        except Exception as e:
            logger.warning(f"Chinese drug query failed ({drug_name}): {e}")

        return {"error": "查询失败"}

    def query_and_extract(self, drug_name: str) -> dict:
        """查询药物并转换为图谱实体格式，可直接用于ingest"""
        info = self.query_drug(drug_name)
        if "error" in info or "drug" not in info:
            return {"entities": [], "relationships": []}

        drug = info["drug"]
        entities = [{
            "type": "Drug",
            "name": drug.get("name", drug_name),
            "properties": {
                "description": drug.get("pharmacology", ""),
                "category": drug.get("category", ""),
                "english_name": drug.get("english_name", ""),
                "dosage_form": drug.get("dosage_form", ""),
                "dosage_adult": drug.get("dosage_adult", ""),
                "pregnancy_category": drug.get("pregnancy_category", ""),
                "source": "中国药典(ChP)"
            }
        }]

        relationships = []
        for indication in drug.get("indications", []):
            entities.append({
                "type": "Disease",
                "name": indication,
                "properties": {"description": f"{drug.get('name', drug_name)}的适应症"}
            })
            relationships.append({
                "from_type": "Drug", "from_name": drug.get("name", drug_name),
                "to_type": "Disease", "to_name": indication,
                "rel_type": "TREATS"
            })
        for interaction in drug.get("drug_interactions", []):
            rel_drug = interaction.get("drug", "")
            if rel_drug:
                entities.append({
                    "type": "Drug",
                    "name": rel_drug,
                    "properties": {"category": "相互作用药物"}
                })
                relationships.append({
                    "from_type": "Drug", "from_name": drug.get("name", drug_name),
                    "to_type": "Drug", "to_name": rel_drug,
                    "rel_type": "INTERACTS_WITH"
                })

        logger.info(f"Chinese drug extracted: {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}


class OpenMedicalSources:
    """外部医学知识源接口（保留PubMed等真实API）"""

    def fetch_pubmed_abstracts(self, keywords: List[str], max_results: int = 20) -> List[Dict]:
        """通过PubMed API获取文献摘要（需要biopython）"""
        try:
            from Bio import Entrez
            Entrez.email = os.getenv("PUBMED_EMAIL", "user@example.com")

            papers = []
            for keyword in keywords[:3]:
                try:
                    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results)
                    record = Entrez.read(handle)
                    handle.close()

                    ids = record["IdList"]
                    if ids:
                        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
                        records = Entrez.read(handle)
                        handle.close()

                        for article in records.get("PubmedArticle", [])[:5]:
                            try:
                                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                                abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", "")
                                if abstract:
                                    papers.append({
                                        "title": title,
                                        "abstract": abstract[0] if isinstance(abstract, list) else abstract,
                                        "keyword": keyword,
                                        "source": "PubMed"
                                    })
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning(f"PubMed search failed ({keyword}): {e}")
                    continue

            logger.info(f"PubMed: fetched {len(papers)} abstracts")
            return papers

        except ImportError:
            logger.warning("BioPython not installed, skipping PubMed")
            return []
        except Exception as e:
            logger.error(f"PubMed API error: {e}")
            return []
