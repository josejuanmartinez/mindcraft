from typing import Union, Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer

from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate
from mindcraft.infra.engine.llm import LLM
from mindcraft.infra.engine.llm_types import LLMType

import logging

from mindcraft.settings import LOGGER_FORMAT, DATE_FORMAT

logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLM(LLM):
    def __init__(self,
                 llm_type: LLMType = LLMType.ZEPHYR7B_AWQ,
                 temperature: float = 0.8):
        """
        Large Language Model class, in charge of executing a prompt and retrieving an answer for the LLM. Used to
        generate the answers of the NPCs.
        :param llm_type: one of the LLMType engines to use.
        :param temperature: temperature to use in generation
        """
        super().__init__(llm_type, temperature)
        self.model = AutoModelForCausalLM.from_pretrained(llm_type.value['name'],
                                                          device_map=self.device,
                                                          trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_type.value['name'],
                                                       device_map=self.device,
                                                       trust_remote_code=True)

    def __call__(self,
                 prompt: str,
                 max_tokens: int = 100,
                 do_sample: bool = True,
                 prompt_template: PromptTemplate = PromptTemplate.ALPACA,
                 streaming: bool = False) -> Union[Iterator[str], str]:
        """
        Sends a prompt to the LLM. You can specify the max. number of tokens to retrieve and if you do sampling when
        generating the text.
        :param prompt: the prompt to use
        :param max_tokens: max tokens to receive
        :param do_sample: apply stochastic selection of tokens to prevent always generating the same wording.
        :param streaming: apply streaming if available (a text iterator will be returned instead of the text)
        Default: true
        :return: an iterator to the text of the answer (streaming=True) or the answer (streaming=False)
        """

        if streaming:
            logging.info("Streaming was ignored as it's not available for non-fast LLM inference. Use `fast=True`.")

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        self.model.to(self.device)

        generated_ids = self.model.generate(**model_inputs,
                                            max_new_tokens=max_tokens,
                                            do_sample=do_sample)
        yield self.tokenizer.batch_decode(generated_ids)[0]

    def retrieve_answer(self,
                        prompt: str,
                        max_tokens: int = 100,
                        do_sample: bool = True,
                        prompt_template: PromptTemplate = PromptTemplate.ALPACA,
                        streaming: bool = False) -> Union[Iterator[str], str]:
        """
        Sends a prompt to the LLM. You can specify the max. number of tokens to retrieve and if you do sampling when
        generating the text.
        :param prompt: the prompt to use
        :param max_tokens: max tokens to receive
        :param do_sample: apply stochastic selection of tokens to prevent always generating the same wording.
        :param prompt_template: the answer usually comes inside the prompt itself, so we need to parse it, for which
        we need the reference to the template used
        :param streaming: apply streaming if available (a text iterator will be returned instead of the text)
        :return: the answer
        """
        response_placeholder = self.llm_type.value['template'].value['response']
        for chunk in self.__call__(prompt, max_tokens, do_sample):
            yield self.clean(chunk, response_placeholder)

