# src/infer.py  (minimal CPU inference with LoRA adapter)
import argparse, torch, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)  # e.g. Qwen/Qwen2.5-3B-Instruct
    p.add_argument("--adapter_dir", required=True) # e.g. outputs/alex-qlora
    p.add_argument("--tone", default="direct")
    p.add_argument("--user", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load base on CPU
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    # Attach adapter
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    model.config.use_cache = False  # safer on CPU

    # Build a simple chat prompt; include your tone tag if you trained with it
    system = f"<tone:{args.tone}>\nYou are Alex. Respond directly."
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": args.user}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # print only the model's continuation
    completion = text.split(messages[-1]["content"])[-1].strip()
    print(completion)

if __name__ == "__main__":
    main()
