from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
MODEL_NAME = "Salesforce/codegen2-1B"
MODEL_CACHE = "cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and tokenizer into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision="main",
            cache_dir=MODEL_CACHE,
            local_files_only=True
        ).to(device)

    def predict(self, 
            prompt: str = Input(description="Instruction for the model"),
            max_new_tokens: int = Input(description="max tokens to generate", default=128),
    ) -> str:
        """Run a single prediction on the model"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
            )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output