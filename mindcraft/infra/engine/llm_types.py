from enum import Enum


class LLMType(Enum):
    MISTRAL7B_AWQ = "TheBloke/openinstruct-mistral-7B-AWQ"
    ZEPHYR7B_AWQ = "TheBloke/zephyr-7B-beta-AWQ"
    YI6B_AWQ = "TheBloke/dragon-yi-6B-v0-AWQ"


