from mindcraft.infra.engine.llm import LLM
from mindcraft.infra.prompts.templates.prompt_template import PromptTemplate
from mindcraft.infra.engine.llm_types import LLMType


class FastLLM(LLM):
    def __init__(self,
                 engine: LLMType = LLMType.ZEPHYR7B_AWQ,
                 temperature: float = 0.8):
        """
        Large Language Model class, in charge of executing a prompt and retrieving an answer for the LLM. Used to
        generate the answers of the NPCs.
        :param engine: one of the LLMType engines to use.
        :param temperature: temperature to use in generation
        """
        super().__init__(engine, temperature)
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("`vllm` is required for Fast Inference. To install it, type:\n"
                              "`pip install vllm`")

        self.sampling_params = SamplingParams(temperature=self.temperature)
        self.llm = LLM(model=self.llm_type.value['name'],
                       trust_remote_code=True,
                       dtype='float16',
                       quantization='awq',
                       tokenizer_mode="auto")

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
        prompts = [prompt]
        response = self.llm.generate(prompts, self.sampling_params)

        return response[0].outputs[0].text

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
        prompt_with_answer = self.__call__(prompt, max_tokens, do_sample)
        response_placeholder = self.llm_type.value['template'].value['response']
        return self.clean(prompt_with_answer, response_placeholder)
