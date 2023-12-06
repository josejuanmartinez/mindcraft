from enum import Enum


class PromptTemplate(Enum):
    NO_ROBOTS = {
        "prompt": "<|system|> {system} </s> <|user|> {prompt} </s> <|assistant|> {{response}} </s>",
        "response": "<|assistant|>"
    }

    ALPACA = {
        "prompt": """{system}\n\n### Instruction:\n{prompt}\n\n### Response:""",
        "response": "### Response:"
    }
