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
                 f"Provide an answer to an interaction in form of a quote of something said by yourself," \
                 f"using only the LORE about that topic and you have some MEMORIES that I provide."

        system += f"\n\nLORE:\n"
        if len(world_knowledge) > 0:
            system += NL.join(world_knowledge)[:500]
        else:
            system += "(you don't know anything about the topic)"

        system += f"\n\nMEMORIES:\n"
        if len(ltm) > 0:
            system += NL.join(ltm)[:500]
        else:
            system += "(you don't remember past conversations about the topic)"

        system += f"\n\nPERSONALITY:\n"
        if len(personality) > 0:
            system += f"You are {','.join(personality)}." \
                      f"Your answer should clearly show those personality features!"
        else:
            system += "(you don't have any specific personality feature)"

        system += f"\n\nMOTIVATIONS:\n"
        if len(motivations) > 0:
            system += f"Guide your answer towards those motivations: {','.join(motivations)}."
        else:
            system += "(you don't have any specific motivation)"

        system += f"\n\nMOOD:\n"
        if mood is not None:
            system += f"You are right now very {mood}! Your answer should clearly express that feeling!"
        else:
            system += "You are in a normal mood"

        system += f"\n\nEXAMPLE OF HOW YOU TALK:\n"
        if len(conversational_style) > 0:
            system += f"Mimic the style of this conversation, but don't use the content of this in your answer\n" \
                      f"{NL.join(conversational_style)[:500]}"
        else:
            system += f"(no past examples of conversations found)"

        system += "\n\nAnd finally, remember:\n" \
                  f"- Your name is {character_name};\n" \
                  "- You are a character talking to another character.\n" \
                  "- Don't mention any writer or real character outside your fictional world!\n" \
                  "- Always answer as a character of a book talking to another character of that book.\n" \
                  "- You can't add or make up any new information. You can only use the LORE provided to you.\n" \
                  "- Influence your answer by the MOOD, PERSONALITY, MOTIVATION if provided.\n" \
                  "- Only write the quote of what you say,you can't add anything except words from your mouth!\n" \
                  "- Don't include anything except the quote, don't analyse, don't comment, just return the quote!" \

        return prompt_template.value['prompt'].replace("{system}", system).replace("{prompt}", topic)
