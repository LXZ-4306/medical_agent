"""
Neo4j图数据库客户端
"""
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j数据库客户端单例"""
    
    _instance = None
    _driver: Optional[Driver] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._driver is None:
            self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4j")
        
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"✅ Neo4j连接成功: {uri}")
        except Exception as e:
            logger.error(f"❌ Neo4j连接失败: {e}")
            raise
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        if self._driver is None:
            self._connect()
        return self._driver.session()
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        执行Cypher查询并返回结果
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def create_node(self, label: str, properties: Dict) -> Dict:
        """创建节点"""
        query = f"""
        CREATE (n:{label} $properties)
        RETURN n
        """
        result = self.execute_query(query, {"properties": properties})
        return result[0] if result else {}
    
    def create_relationship(self, 
                           from_label: str, 
                           from_id: str,
                           to_label: str, 
                           to_id: str,
                           rel_type: str,
                           properties: Dict = None) -> bool:
        """创建关系"""
        query = f"""
        MATCH (a:{from_label} {{id: $from_id}})
        MATCH (b:{to_label} {{id: $to_id}})
        CREATE (a)-[r:{rel_type} $properties]->(b)
        RETURN r
        """
        params = {
            "from_id": from_id,
            "to_id": to_id,
            "properties": properties or {}
        }
        result = self.execute_query(query, params)
        return len(result) > 0
    
    def query_related(self, entity_name: str, relationship_type: str = None, 
                     limit: int = 10) -> List[Dict]:
        """
        查询与实体相关的知识
        """
        rel_clause = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
        MATCH (a)-[r{rel_clause}]-(b)
        WHERE a.name CONTAINS $entity_name OR b.name CONTAINS $entity_name
        RETURN a, r, b
        LIMIT $limit
        """
        return self.execute_query(query, {
            "entity_name": entity_name,
            "limit": limit
        })
    
    def get_all_nodes(self, label: str = None) -> List[Dict]:
        """获取所有节点"""
        label_clause = f":{label}" if label else ""
        query = f"MATCH (n{label_clause}) RETURN n"
        return self.execute_query(query)
    
    def merge_node(self, label: str, match_props: dict, extra_props: dict = None) -> dict:
        """幂等创建/更新节点，按name字段匹配"""
        all_props = {**match_props, **(extra_props or {})}
        query = f"""
        MERGE (n:{label} {{name: $match_name}})
        SET n += $properties
        RETURN n
        """
        result = self.execute_query(query, {
            "match_name": match_props.get("name", ""),
            "properties": all_props
        })
        return result[0] if result else {}

    def merge_relationship(self, from_label: str, from_name: str,
                           to_label: str, to_name: str, rel_type: str) -> bool:
        """幂等创建关系，已存在则跳过"""
        query = f"""
        MATCH (a:{from_label} {{name: $from_name}})
        MATCH (b:{to_label} {{name: $to_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET r.occurrences = 1, r.first_seen = timestamp()
        ON MATCH SET r.occurrences = r.occurrences + 1, r.last_seen = timestamp()
        RETURN r
        """
        result = self.execute_query(query, {
            "from_name": from_name,
            "to_name": to_name
        })
        return len(result) > 0

    def create_schema(self):
        """创建唯一性约束和索引，幂等（已存在则忽略）"""
        constraints = [
            "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT procedure_name_unique IF NOT EXISTS FOR (n:Procedure) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT bodypart_name_unique IF NOT EXISTS FOR (n:BodyPart) REQUIRE n.name IS UNIQUE",
        ]
        for cypher in constraints:
            try:
                self.execute_query(cypher)
                logger.info(f"  Schema: {cypher[:60]}...")
            except Exception as e:
                logger.debug(f"  Constraint may already exist: {e}")
        logger.info("Schema constraints initialized")

    def clear_database(self):
        """清空数据库（危险操作）"""
        self.execute_query("MATCH (n) DETACH DELETE n")
        logger.warning("🗑️ 数据库已清空")