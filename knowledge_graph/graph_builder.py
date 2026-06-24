"""
医学知识图谱构建器 — 增量模式
"""
from typing import Dict
import logging
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class MedicalGraphBuilder:
    """增量构建医学知识图谱，随对话不断增长"""

    def __init__(self):
        self.client = Neo4jClient()

    def build(self, include_pubmed: bool = False):
        """初始化图谱schema（约束和索引），不清空已有数据，不加载假数据"""
        logger.info("Initializing knowledge graph schema (incremental mode)...")
        self.client.create_schema()
        stats = self.get_statistics()
        logger.info(f"Schema ready. Current graph: {stats}")
        return stats

    def ingest_conversation(self, query: str, response: str) -> int:
        """从一次对话中提取实体并增量写入图谱，返回新增实体数"""
        from .open_sources import MedicalEntityExtractor

        try:
            extractor = MedicalEntityExtractor()
            extracted = extractor.extract_from_conversation(query, response)

            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            if not entities:
                logger.debug("No entities extracted from this conversation")
                return 0

            for entity in entities:
                etype = entity.get("type", "")
                name = entity.get("name", "")
                props = entity.get("properties", {})
                if name and etype:
                    self.client.merge_node(etype, {"name": name}, props)

            for rel in relationships:
                from_type = rel.get("from_type", "")
                from_name = rel.get("from_name", "")
                to_type = rel.get("to_type", "")
                to_name = rel.get("to_name", "")
                rel_type = rel.get("rel_type", "")
                if all([from_type, from_name, to_type, to_name, rel_type]):
                    self.client.merge_relationship(
                        from_type, from_name, to_type, to_name, rel_type
                    )

            logger.info(f"Ingested {len(entities)} entities, {len(relationships)} relationships")
            return len(entities)

        except Exception as e:
            logger.warning(f"Knowledge ingestion skipped: {e}")
            return 0

    def ingest_chinese_drug(self, drug_name: str) -> int:
        """查询中国药典并摄入单个药物到图谱，返回新增实体数"""
        from .open_sources import ChineseDrugQuery

        try:
            query = ChineseDrugQuery()
            extracted = query.query_and_extract(drug_name)

            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            if not entities:
                return 0

            for entity in entities:
                etype = entity.get("type", "")
                name = entity.get("name", "")
                props = entity.get("properties", {})
                if name and etype:
                    self.client.merge_node(etype, {"name": name}, props)

            for rel in relationships:
                from_type = rel.get("from_type", "")
                from_name = rel.get("from_name", "")
                to_type = rel.get("to_type", "")
                to_name = rel.get("to_name", "")
                rel_type = rel.get("rel_type", "")
                if all([from_type, from_name, to_type, to_name, rel_type]):
                    self.client.merge_relationship(
                        from_type, from_name, to_type, to_name, rel_type
                    )

            logger.info(f"Chinese drug ingested: {drug_name} -> {len(entities)} entities")
            return len(entities)

        except Exception as e:
            logger.warning(f"Chinese drug ingestion failed ({drug_name}): {e}")
            return 0

    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        stats = {}
        for label in ["Drug", "Disease", "Symptom", "Procedure", "BodyPart"]:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = self.client.execute_query(query)
            stats[f"{label.lower()}_count"] = result[0]["count"] if result else 0

        query = "MATCH ()-[r]->() RETURN count(r) as count"
        result = self.client.execute_query(query)
        stats["relationship_count"] = result[0]["count"] if result else 0

        return stats

    def test_queries(self):
        """测试查询功能"""
        test_cases = ["头痛", "糖尿病", "阿司匹林", "高血压"]

        logger.info("Testing knowledge graph queries...")
        for query in test_cases:
            results = self.client.query_related(query)
            logger.info(f"  Query '{query}': {len(results)} results")
            for i, result in enumerate(results[:2]):
                logger.debug(f"    Result {i+1}: {result}")

        return True
