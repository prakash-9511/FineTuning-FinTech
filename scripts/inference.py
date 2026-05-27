import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(instruction: str, context: str) -> str:
    prompt = f"### Instruction:\n{instruction.strip()}\n"
    if context.strip():
        prompt += f"\n### Context:\n{context.strip()}\n"
    prompt += "\n### Response:\n"
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers with a fine-tuned LoRA adapter.")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    prompt = build_prompt(args.instruction, args.input)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
