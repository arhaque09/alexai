import yaml
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model
import torch

@dataclass
class CFG:
    base_model: str; dataset_path: str; output_dir: str; adapter_name: str
    bf16: bool; per_device_train_batch_size: int; gradient_accumulation_steps: int
    learning_rate: float; num_train_epochs: int; logging_steps: int; save_steps: int
    warmup_ratio: float; lr_scheduler_type: str; packing: bool; max_seq_length: int
    lora_r: int; lora_alpha: int; lora_dropout: float; target_modules: list
    bnb_4bit: bool; bnb_4bit_quant_type: str; bnb_4bit_compute_dtype: str

def load_cfg(path="configs/qlora.yaml") -> CFG:
    with open(path) as f: d = yaml.safe_load(f)
    return CFG(**d)

def format_example(ex, tok):
    # Build a chat-style prompt, then append the gold output
    messages = [
        {"role":"system","content": ex["system"]},
        {"role":"user","content": ex["input"]}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"text": prompt + ex["output"]}

def main():
    cfg = load_cfg()

    # ---- Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Base model on CPU dont have CUDA so this is what i use
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # ---- LoRA (PEFT)
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    
    model.enable_input_require_grads()   # make embeddings require grad flow for LoRA
    model.config.use_cache = False       # avoid cache with training

    # ---- Dataset -> text column
    ds = load_dataset("json", data_files=cfg.dataset_path, split="train")
    ds = ds.map(lambda ex: format_example(ex, tok), remove_columns=ds.column_names)

    # ---- Tokenize to input_ids/labels
    def tok_map(batch):
        enc = tok(batch["text"], truncation=True, max_length=cfg.max_seq_length)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tok_map, batched=True, remove_columns=["text"])

    # ---- Collator (causal LM)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # ---- Training args (CPU: no fp16/bf16)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=False, bf16_full_eval=False,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=False   # ← set to False on CPU
)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    print("Saved adapter to", cfg.output_dir)

if __name__ == "__main__":
    main()
