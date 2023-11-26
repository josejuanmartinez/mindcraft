NL = '\n-'


class Answer:
    @staticmethod
    def create(stm: list[str],
               ltm: list[str],
               world_knowledge: list[str],
               character_id: str,
               world_id: str,
               topic: str,
               personality: list[str],
               motivations: list[str]) -> str:

        prompt = f"""You are {character_id}, a character from the world of {world_id}. You need to react to an 
        interaction about a TOPIC, given that you know some details about that topic:"""
        if len(world_knowledge) > 0:
            prompt += f"""1) From books, chronicles, and stories known by you and other people about the world, you know
             that: {NL.join(world_knowledge)}"""
        if len(ltm) > 0:
            prompt += f"""2) Also, you have some memories about this topic which happened personally to you:
            {NL.join(ltm)}"""
        if len(stm) > 0:
            prompt += f"""3) Recently you have had these conversations with other people, which may or may not be 
            relevant: {NL.join(stm)}"""
        if len(personality) > 0:
            prompt += f"""You are a {",".join(personality)} character, so answer accordingly."""
        if len(motivations) > 0:
            prompt += f"""Also, you have several goals and motivations in life, namely: {",".join(motivations)}"""

        prompt += f"""
        TOPIC:
        {topic}
        """

        return prompt
