# src/runtime/hf_adapter_client.py  (tiny tweak for CPU)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
    TemperatureLogitsWarper
)
from transformers import (
        LogitsProcessorList,
        NoRepeatNGramLogitsProcessor,
        TopPLogitsWarper,
        TopKLogitsWarper,
        # If your version has it; otherwise keep temperature=... in generate()
        TemperatureLogitsWarper,
    )

from peft import PeftModel

class HFAdapterClient:
    def __init__(self,
                 base_model="Qwen/Qwen2.5-3B-Instruct",
                 adapter_dir="outputs/alex-qlora",
                 use_4bit=False):             # ← leave False on CPU
        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()

    def chat(self, messages):
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt")

        # Only ONE penalty path: we will pass repetition_penalty=...
        # Keep no-repeat-ngrams + sampling as custom processors.
        logits_processors = LogitsProcessorList([
            NoRepeatNGramLogitsProcessor(3),
            TopKLogitsWarper(50),
            TopPLogitsWarper(0.9),
            TemperatureLogitsWarper(0.85),
        ])

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=420,
                do_sample=True,
                # Use argument-based repetition penalty (no custom processor)
                repetition_penalty=1.18,
                logits_processor=logits_processors,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
            )

        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text.split(messages[-1]["content"])[-1].strip()