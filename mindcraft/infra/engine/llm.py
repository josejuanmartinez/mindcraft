import re
import torch

from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate
from mindcraft.infra.engine.llm_types import LLMType


class LLM:
    def __init__(self,
                 llm_type: LLMType = LLMType.MISTRAL7B_AWQ,
                 temperature: float = 0.8):
        """
        Large Language Model class, in charge of executing a prompt and retrieving an answer for the LLM. Used to
        generate the answers of the NPCs.
        :param llm_type: one of the LLMType engines to use.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llm_type = llm_type
        self.temperature = temperature

    def __call__(self,
                 prompt: str,
                 max_tokens: int = 100,
                 do_sample: bool = True) -> str:
        """
        Sends a prompt to the LLM. You can specify the max. number of tokens to retrieve and if you do sampling when
        generating the text.
        :param prompt: the prompt to use
        :param max_tokens: max tokens to receive
        :param do_sample: apply stochastic selection of tokens to prevent always generating the same wording.
        Default: true
        :return: the answer
        """
        raise NotImplementedError()

    def retrieve_answer(self,
                        prompt: str,
                        max_tokens: int = 100,
                        do_sample: bool = True) -> str:
        """
        Sends a prompt to the LLM. You can specify the max. number of tokens to retrieve and if you do sampling when
        generating the text.
        :param prompt: the prompt to use
        :param max_tokens: max tokens to receive
        :param do_sample: apply stochastic selection of tokens to prevent always generating the same wording.
        :return: the answer
        """
        raise NotImplementedError()

    def clean(self, answer: str, response_placeholder: str):
        prompt_with_answer = answer
        index = prompt_with_answer.find(response_placeholder)
        if index != -1:
            prompt_with_answer = prompt_with_answer[index + len(response_placeholder):]
        prompt_with_answer = prompt_with_answer.replace("</s>", "")
        prompt_with_answer = prompt_with_answer.replace("<s>", "")
        prompt_with_answer = prompt_with_answer.replace("</s>", "")
        prompt_with_answer = prompt_with_answer.replace("\"", "")
        prompt_with_answer = re.sub(r"\(.*\)", "", prompt_with_answer)
        return prompt_with_answer
