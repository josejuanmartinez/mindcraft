import torch

from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate
from mindcraft.infra.engine.llm_types import LLMType


class LLM:
    def __init__(self,
                 model_name: LLMType = LLMType.MISTRAL7B_AWQ):
        """
        Large Language Model class, in charge of executing a prompt and retrieving an answer for the LLM. Used to
        generate the answers of the NPCs.
        :param model_name: one of the LLMType engines to use.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.engine = model_name

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
                        do_sample: bool = True,
                        prompt_template: PromptTemplate = PromptTemplate.ALPACA) -> str:
        """
        Sends a prompt to the LLM. You can specify the max. number of tokens to retrieve and if you do sampling when
        generating the text.
        :param prompt: the prompt to use
        :param max_tokens: max tokens to receive
        :param do_sample: apply stochastic selection of tokens to prevent always generating the same wording.
        :param prompt_template: the answer usually comes inside the prompt itself, so we need to parse it, for which
        we need the reference to the template used
        :return: the answer
        """
        raise NotImplementedError()

