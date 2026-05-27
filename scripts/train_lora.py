import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def build_prompt(example: dict) -> str:
    prompt = f"### Instruction:\n{example['instruction'].strip()}\n"
    if example.get("input", "").strip():
        prompt += f"\n### Context:\n{example['input'].strip()}\n"
    prompt += "\n### Response:\n"
    return prompt


def tokenize_example(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    prompt = build_prompt(example)
    response = example["output"].strip() + tokenizer.eos_token
    full_text = prompt + response

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full["input_ids"]
    labels = input_ids.copy()
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


class CausalLMCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on finance and tech instructions with LoRA.")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_file", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--validation_file", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/fintech-lora"))
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--use_4bit requires a CUDA GPU and bitsandbytes support.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
    )
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )
    tokenized = dataset.map(
        lambda example: tokenize_example(example, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments(args),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
