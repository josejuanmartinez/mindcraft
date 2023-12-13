import json

import requests

from mindcraft import settings
from mindcraft.infra.engine.fast_llm import FastLLM
from mindcraft.infra.engine.llm import LLM
from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate
from mindcraft.infra.engine.llm_types import LLMType


class RemoveFastLLM(FastLLM):
    def __init__(self,
                 engine: LLMType = LLMType.MISTRAL7B_AWQ):
        """
        Large Language Model class, in charge of executing a prompt and retrieving an answer for the LLM. Used to
        generate the answers of the NPCs.
        :param engine: one of the LLMType engines to use.
        """
        super().__init__(engine)

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
        :return: the answer in a streaming fashion
        """
        headers = {"User-Agent": "mindcraft"}
        request = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "use_beam_search": not do_sample
        }
        response = requests.post(settings.FAST_INFERENCE_URL,
                                 headers=headers,
                                 json=request,
                                 stream=False)

        chunks = []
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][0]
                chunks.append(output)
        return "".join(chunks)

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
        return self.__call__(prompt, max_tokens, do_sample)
        """for elem in self.__call__(prompt, max_tokens, do_sample):
            print(elem, end='')
            yield elem
        print()"""

