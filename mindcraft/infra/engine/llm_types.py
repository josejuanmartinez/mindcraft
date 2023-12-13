from enum import Enum


class LLMType(Enum):
    MISTRAL7B_AWQ = "TheBloke/openinstruct-mistral-7B-AWQ"
    ZEPHYR7B = "HuggingFaceH4/zephyr-7b-beta"
    ZEPHYR7B_AWQ = "TheBloke/zephyr-7B-beta-AWQ"
    PHI15 = "microsoft/phi-1_5"


