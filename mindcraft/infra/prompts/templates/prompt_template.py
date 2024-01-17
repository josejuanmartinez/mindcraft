from enum import Enum


class PromptTemplate(Enum):
    NO_ROBOTS = {
        "prompt": "<|system|>\n{system}\n</s>\n<|user|>\n{prompt}\n</s>\n<|assistant|>",
        "response": "<|assistant|>"
    }

    ALPACA = {
        "prompt": """{system}\n\n### Instruction:\n{prompt}\n\n### Response:""",
        "response": "### Response:"
    }

    OPENCHAT = {
        "prompt": """{system}\n{prompt}\n<|end_of_turn|>\nGPT4 Assistant:""",
        "response": """GPT4 Assistant:"""
    }

    ONLY_PROMPT = {
        "prompt": """{system}\n{prompt}\nYOUR RESPONSE:""",
        "response": """YOUR RESPONSE:"""
    }

    INSTRUCT_PROMPT_OUTPUT = {
        "prompt": """{system}\nInstruct: {prompt}\nOutput:""",
        "response": """Output:"""
    }
