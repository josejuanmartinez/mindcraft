from transformers import AutoModelForCausalLM, AutoTokenizer
from mindcraft.infra.engine.llm_types import LLMType


class LLM:
    def __init__(self,
                 engine: LLMType = LLMType.MISTRAL7B,
                 device: str = 'cuda'):
        """
        Large Language Model class, in charge of executing a prompt and retrieving an answer for the LLM. Used to
        generate the answers of the NPCs.
        :param engine: one of the LLMType engines to use.
        :param device: device to use (default: `cuda`)
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(engine.value, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(engine.value, device_map=self.device)

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
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        self.model.to(self.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=do_sample)
        return self.tokenizer.batch_decode(generated_ids)[0]
