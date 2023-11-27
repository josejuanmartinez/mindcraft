from enum import Enum


class PromptTemplate(Enum):
    NO_ROBOTS = "<|system|> {system} </s> <|user|> {prompt} </s> <|assistant|> {{response}} </s>"
    ALPACA = """{system}\n\n### Instruction:\n{prompt}\n\n### Response:"""
