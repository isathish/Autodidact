"""
Phase 3, Milestone M3.3 — Relationship Learning
Extracts relationships from text and actions to populate the knowledge graph.
"""

import re
from memory.graph_builder import KnowledgeGraph


class RelationshipExtractor:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def extract_from_text(self, text: str):
        """
        Very basic pattern-based relationship extraction.
        Example: "A cat is an animal" -> (cat)-[IS_A]->(animal)
        """
        patterns = [
            (r"(\w+)\s+is\s+an?\s+(\w+)", "IS_A"),
            (r"(\w+)\s+causes\s+(\w+)", "CAUSES")
        ]
        for pattern, rel_type in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subj, obj = match.groups()
                self.kg.create_concept(subj.lower())
                self.kg.create_concept(obj.lower())
                self.kg.create_relationship(subj.lower(), obj.lower(), rel_type)

    def extract_from_action(self, action_desc: str, result_desc: str):
        """
        Example: CLICK 'play video' -> video plays
        Creates (play video button)-[CAUSES]->(video playing)
        """
        self.kg.create_concept(action_desc.lower(), "action")
        self.kg.create_concept(result_desc.lower(), "event")
        self.kg.create_relationship(action_desc.lower(), result_desc.lower(), "CAUSES")


if __name__ == "__main__":
    kg = KnowledgeGraph(password="test")
    extractor = RelationshipExtractor(kg)
    extractor.extract_from_text("A cat is an animal")
    extractor.extract_from_action("play video button", "video playing")
    print("Concepts:", kg.get_concepts())
    kg.close()
