from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate

NL = '\n-'


class Prompt:
    def __init__(self):
        """
        Static class that manages the creation of the prompts, gathering all the information from the world,
        past interactions, personalities, motivation, mood, conversational styles, etc. to query the LLM
        """
        pass

    @staticmethod
    def create(ltm: list[str],
               world_knowledge: list[str],
               character_name: str,
               world_name: str,
               topic: str,
               personality: list[str],
               motivations: list[str],
               conversational_style: list[str],
               mood: str = None,
               prompt_template: PromptTemplate = PromptTemplate.ALPACA) -> str:
        """
        Static method that creates the prompt to send to the LLM, gathering all the information from the world,
        past interactions, personalities, motivation, mood, conversational styles, etc.
        :param ltm: A list of past interactions with a specific character about this topic
        :param world_knowledge: Pieces of lore/knowledge in the world about this topic
        :param character_name: The name of the character
        :param world_name: The name of the world
        :param topic: The topic you are asking about
        :param personality: A list of personalities of the NPC who is answering. For example: `wise`, `intelligent`
        :param motivations: A list of motivations seeked by the NPC who is answering. For example:
        `protecting the nature`
        :param conversational_style: A list of examples of a conversation which happened when the NPC was in a similar
        mood
        :param mood: The current mood of the NPC
        :param prompt_template: One of the PromptTemplate types to use depending on your LLM.
        :return: the prompt
        """

        system = f"You are {character_name}, a character from the world of {world_name}. " \
                 f"Answer to the question of another character of your own world, given that you know some " \
                 f"details about that topic.\n"
        if mood is not None:
            system += f"\n\nYou are right now very {mood}! Your answer should clearly show that feeling!"

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

        if len(conversational_style) > 0:
            system += f"\n\nAn example of how you talk with your actual mood about other topics is the following " \
                      f"(mimic the style but ignore the content as it is for other topic):" \
                      f" {NL.join(conversational_style)[:500]}"

        system += "\n\nRemember you are a character talking to another character. You are not aware of the author " \
                  "/ writer of the book or lore. Always answer as a character of a book talking to another character."

        return prompt_template.value['prompt'].replace("{system}", system).replace("{prompt}", topic)

