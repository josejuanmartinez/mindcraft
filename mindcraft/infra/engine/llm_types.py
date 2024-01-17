from enum import Enum

from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate


class LLMType(Enum):
    MISTRAL7B_AWQ = {
        "name": "TheBloke/mistral_7b_norobots-AWQ",
        "template": PromptTemplate.NO_ROBOTS,
        "quantization": "awq"
    }
    ZEPHYR7B_AWQ = {
        "name": "TheBloke/zephyr-7B-beta-AWQ",
        "template": PromptTemplate.NO_ROBOTS,
        "quantization": "awq"
    }
    NOTUS7B_AWQ = {
        "name": "TheBloke/notus-7B-v1-AWQ",
        "template": PromptTemplate.NO_ROBOTS,
        "quantization": "awq"
    }
    STARLING7B_AWQ = {
        "name": "TheBloke/Starling-LM-7B-alpha-AWQ",
        "template": PromptTemplate.OPENCHAT,
        "quantization": "awq"
    }
    YI_6B_AWQ = {
        "name": "TheBloke/Yi-6B-AWQ",
        "template": PromptTemplate.ONLY_PROMPT,
        "quantization": "awq"
    }
    PHI2_3B = {
        "name": "microsoft/phi-2",
        "template": PromptTemplate.INSTRUCT_PROMPT_OUTPUT,
        "quantization": None
    }
    STABLELM_ZEPHYR_3B = {
        "name": "stabilityai/stablelm-zephyr-3b",
        "template": PromptTemplate.NO_ROBOTS,
        "quantization": None
    }


