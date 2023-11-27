from collections import deque

from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate

NL = '\n-'


class Answer:
    @staticmethod
    def create(stm: deque[str],
               ltm: list[str],
               world_knowledge: list[str],
               character_id: str,
               world_id: str,
               topic: str,
               personality: list[str],
               motivations: list[str],
               prompt_template: PromptTemplate = PromptTemplate.ALPACA) -> str:

        system = f"You are {character_id}, a character from the world of {world_id}. " \
                 f"Answer to the question of another character of your own world, given that you know some " \
                 f"details about that topic.\n"
        if len(personality) > 0:
            system += f"\n\nCreate your answers taking into account that you are a {','.join(personality)} character." \
                      f" Reformulate your answer adding that features of your personality into them."""
        if len(motivations) > 0:
            system += f"\n\nAlso, you have several goals and motivations in life, namely: {','.join(motivations)}." \
                      f" Take them into account to reformulate your answers guiding them towards your motivations."
        if len(world_knowledge) > 0:
            system += f"\n1) From books, chronicles, and stories known by you and other people about the world," \
                      f" you know that: {NL.join(world_knowledge)[:500]}"
        if len(ltm) > 0:
            system += f"\n2) Also, you have some memories about this topic which happened personally to you " \
                      f"{NL.join(ltm)}"
        if len(stm) > 0:
            system += f"\n3) Recently you have had these conversations with other people, which may or may not be " \
                      f"relevant: {NL.join(stm)}"

        system += "\n\nRemember you are a character talking to another character. You are not aware of the author " \
                  "/ writer of the book or lore. Always answer as a character of a book talking to another character."

        return prompt_template.value.replace("{system}", system).replace("{prompt}", topic)

