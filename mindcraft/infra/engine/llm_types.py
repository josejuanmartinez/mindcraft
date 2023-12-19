from enum import Enum

from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate


class LLMType(Enum):
    MISTRAL7B_AWQ = {
        "name": "TheBloke/mistral_7b_norobots-AWQ",
        "template": PromptTemplate.NO_ROBOTS
    }
    ZEPHYR7B_AWQ = {
        "name": "TheBloke/zephyr-7B-beta-AWQ",
        "template": PromptTemplate.NO_ROBOTS
    }
    NOTUS7B_AWQ = {
        "name": "TheBloke/notus-7B-v1-AWQ",
        "template": PromptTemplate.NO_ROBOTS
    }
    STARLING_AWQ = {
        "name": "TheBloke/Starling-LM-7B-alpha-AWQ",
        "template": PromptTemplate.OPENCHAT
    }


