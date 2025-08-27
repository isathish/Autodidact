"""
Phase 3, Milestone M3.2 — Knowledge Graph Population
Populates Neo4j with grounded concepts and relationships.
"""

from neo4j import GraphDatabase


class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_concept(self, name, concept_type="entity"):
        with self.driver.session() as session:
            session.run(
                "MERGE (c:Concept {name: $name, type: $type})",
                name=name,
                type=concept_type
            )

    def create_relationship(self, from_concept, to_concept, rel_type):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Concept {name: $from_name})
                MATCH (b:Concept {name: $to_name})
                MERGE (a)-[r:%s]->(b)
                """ % rel_type,
                from_name=from_concept,
                to_name=to_concept
            )

    def get_concepts(self):
        with self.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN c.name AS name, c.type AS type")
            return [(record["name"], record["type"]) for record in result]


if __name__ == "__main__":
    kg = KnowledgeGraph(password="test")
    kg.create_concept("cat", "animal")
    kg.create_concept("animal", "category")
    kg.create_relationship("cat", "animal", "IS_A")
    print("Concepts:", kg.get_concepts())
    kg.close()
