# Fine-Tuning a Domain-Specific AI Model for Finance and Tech

This project demonstrates how to fine-tune an open-source Large Language Model for finance and technology vocabulary, professional response style, document summarization, and jargon-aware question answering.

The implementation uses parameter-efficient fine-tuning with LoRA. It can be run with a model such as `mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Meta-Llama-3-8B-Instruct`, or a smaller model for a local smoke test.

## Project Objective

Train a general-purpose LLM to better understand and respond to industry-specific language from:

- Finance: EBITDA, P/E ratio, derivatives, equity dilution, SIP investment, balance sheet analysis
- Technology: Kubernetes, microservices, CI/CD, Docker containers, REST APIs, vector databases

After fine-tuning, the model should be able to:

- Answer industry-specific questions accurately
- Understand common finance and technology jargon
- Generate professional business responses
- Summarize domain documents in a concise, structured way

## Repository Structure

```text
.
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
├── scripts/
│   ├── evaluate_keywords.py
│   ├── inference.py
│   ├── prepare_dataset.py
│   └── train_lora.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For large models, a CUDA GPU is strongly recommended. If you only need to verify that the code path works locally, use a small model such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## Open the UI

The project includes a static workflow UI under `ui/`.

Start a local server from the project root:

```powershell
python -m http.server 8080 -d ui
```

Open this URL in your browser:

```text
http://localhost:8080
```

The UI helps you review the dataset, configure LoRA training, generate PowerShell commands, build inference prompts, and run keyword evaluation.

## Dataset Format

Each line in `data/train.jsonl` and `data/validation.jsonl` is a JSON object:

```json
{
  "domain": "finance",
  "instruction": "Explain EBITDA to a startup founder.",
  "input": "",
  "output": "EBITDA means earnings before interest, taxes, depreciation, and amortization..."
}
```

This instruction-tuning format teaches the model both terminology and response style.

## Validate the Dataset

```powershell
python scripts/prepare_dataset.py --train data/train.jsonl --validation data/validation.jsonl
```

## Fine-Tune with LoRA

Example using Mistral:

```powershell
python scripts/train_lora.py `
  --model_name mistralai/Mistral-7B-Instruct-v0.3 `
  --train_file data/train.jsonl `
  --validation_file data/validation.jsonl `
  --output_dir outputs/fintech-lora `
  --num_train_epochs 3 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 2e-4 `
  --use_4bit
```

Local smoke test with a smaller model:

```powershell
python scripts/train_lora.py `
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --train_file data/train.jsonl `
  --validation_file data/validation.jsonl `
  --output_dir outputs/fintech-lora-smoke `
  --num_train_epochs 1 `
  --max_length 512
```

The script saves the LoRA adapter and tokenizer files under the selected output directory.

## Run Inference

```powershell
python scripts/inference.py `
  --model_name mistralai/Mistral-7B-Instruct-v0.3 `
  --adapter_path outputs/fintech-lora `
  --instruction "Explain how equity dilution affects existing shareholders." `
  --input ""
```

You can also ask technology questions:

```powershell
python scripts/inference.py `
  --model_name mistralai/Mistral-7B-Instruct-v0.3 `
  --adapter_path outputs/fintech-lora `
  --instruction "Summarize why Kubernetes is useful for microservices." `
  --input ""
```

## Evaluate Keyword Coverage

The evaluation script checks whether generated answers include expected domain concepts.

```powershell
python scripts/evaluate_keywords.py `
  --model_name mistralai/Mistral-7B-Instruct-v0.3 `
  --adapter_path outputs/fintech-lora
```

The score is not a replacement for expert review, but it provides a quick sanity check that the model is using relevant terminology.

## Expected Outcome

A successful fine-tuned model should produce answers like:

- Finance: "EBITDA is useful for comparing operating profitability before financing and accounting choices, but it should not be treated as free cash flow."
- Technology: "A CI/CD pipeline automates build, test, and deployment stages so teams can release smaller changes with lower operational risk."
- Summarization: "The balance sheet shows liquidity pressure because current liabilities exceed liquid assets, while retained earnings suggest historical profitability."

## Notes

- `meta-llama/Meta-Llama-3-8B-Instruct` may require accepting model terms on Hugging Face before downloading.
- 4-bit training depends on CUDA-compatible `bitsandbytes`. If unavailable, omit `--use_4bit`.
- This dataset is intentionally small for assignment demonstration. For production work, expand it with reviewed documents, question-answer pairs, glossary entries, and compliance-approved examples.
