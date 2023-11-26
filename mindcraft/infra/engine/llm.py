from transformers import AutoModelForCausalLM, AutoTokenizer

from mindcraft.infra.engine.llm_types import LLMType


class LLM:
    def __init__(self,
                 engine: LLMType = LLMType.MISTRAL7B,
                 device: str = 'cuda'):
        """

        :param engine:
        :param device:
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(engine.value)
        self.tokenizer = AutoTokenizer.from_pretrained(engine.value)

    def __call__(self,
                 prompt: str,
                 max_tokens: int = 100,
                 do_sample: bool = True):
        """

        :param prompt:
        :return:
        """
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        self.model.to(self.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=do_sample)
        return self.tokenizer.batch_decode(generated_ids)[0]
