"""
Phase 4, Milestone M4.1 — Goal Generation System
Generates curiosity, information, and skill-based goals for the agent.
"""

import random


class GoalGenerator:
    def __init__(self):
        self.goal_types = ["curiosity", "information", "skill"]

    def generate_goal(self, world_model=None, knowledge_graph=None):
        goal_type = random.choice(self.goal_types)
        if goal_type == "curiosity":
            return self._generate_curiosity_goal(world_model)
        elif goal_type == "information":
            return self._generate_information_goal(knowledge_graph)
        elif goal_type == "skill":
            return self._generate_skill_goal()
        return None

    def _generate_curiosity_goal(self, world_model):
        return "Navigate to a page that will surprise my world model."

    def _generate_information_goal(self, knowledge_graph):
        if knowledge_graph:
            concepts = knowledge_graph.get_concepts()
            if concepts:
                concept = random.choice(concepts)[0]
                return f"Find out what '{concept}' means."
        return "Discover a new concept."

    def _generate_skill_goal(self):
        skills = [
            "Learn how to use a search engine.",
            "Learn how to fill out a form.",
            "Learn how to navigate multi-page content."
        ]
        return random.choice(skills)


if __name__ == "__main__":
    generator = GoalGenerator()
    for _ in range(5):
        print("Generated goal:", generator.generate_goal())
