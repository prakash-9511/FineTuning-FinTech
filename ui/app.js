const examples = [
  {
    domain: "finance",
    task: "explain",
    instruction: "Explain EBITDA to a startup founder.",
    output:
      "EBITDA estimates operating profitability before financing, taxes, and non-cash accounting charges.",
  },
  {
    domain: "finance",
    task: "explain",
    instruction: "What does a high P/E ratio usually indicate?",
    output:
      "It can indicate growth expectations, premium market confidence, or overvaluation compared with peers.",
  },
  {
    domain: "finance",
    task: "answer",
    instruction: "Explain equity dilution to an employee receiving stock options.",
    output:
      "New share issuance reduces ownership percentage, but may still create value if valuation increases.",
  },
  {
    domain: "finance",
    task: "summarize",
    instruction: "Summarize a balance sheet with rising receivables and short-term debt.",
    output:
      "Revenue growth needs review because collections, margins, and liquidity risk may be weakening.",
  },
  {
    domain: "tech",
    task: "explain",
    instruction: "Explain Kubernetes to a product manager.",
    output:
      "Kubernetes manages container deployment, scaling, service discovery, health checks, and recovery.",
  },
  {
    domain: "tech",
    task: "explain",
    instruction: "What are microservices?",
    output:
      "Microservices split an application into independently deployable services around focused capabilities.",
  },
  {
    domain: "tech",
    task: "answer",
    instruction: "Explain vector databases for an enterprise search project.",
    output:
      "Vector databases store embeddings so applications can retrieve semantically related content.",
  },
  {
    domain: "tech",
    task: "summarize",
    instruction: "Summarize a Docker, Kubernetes, and CI/CD migration.",
    output:
      "The service was modernized with containers, orchestration, automated testing, scanning, and staged deployment.",
  },
];

const navButtons = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");
const domainFilter = document.querySelector("#domain-filter");
const taskFilter = document.querySelector("#task-filter");
const exampleList = document.querySelector("#example-list");
const toast = document.querySelector("#toast");

const controls = {
  modelName: document.querySelector("#model-name"),
  outputDir: document.querySelector("#output-dir"),
  epochs: document.querySelector("#epochs"),
  learningRate: document.querySelector("#learning-rate"),
  batchSize: document.querySelector("#batch-size"),
  gradAccum: document.querySelector("#grad-accum"),
  maxLength: document.querySelector("#max-length"),
  use4bit: document.querySelector("#use-4bit"),
  adapterPath: document.querySelector("#adapter-path"),
  instructionInput: document.querySelector("#instruction-input"),
  contextInput: document.querySelector("#context-input"),
};

const trainingCommand = document.querySelector("#training-command");
const inferenceCommand = document.querySelector("#inference-command");
const evaluationCommand = document.querySelector("#evaluation-command");

function shellQuote(value) {
  return `"${String(value).replaceAll('"', '\\"')}"`;
}

function buildTrainingCommand() {
  const use4bit = controls.use4bit.checked ? " `\n  --use_4bit" : "";
  return `python scripts/train_lora.py \`
  --model_name ${controls.modelName.value} \`
  --train_file data/train.jsonl \`
  --validation_file data/validation.jsonl \`
  --output_dir ${controls.outputDir.value} \`
  --num_train_epochs ${controls.epochs.value} \`
  --per_device_train_batch_size ${controls.batchSize.value} \`
  --gradient_accumulation_steps ${controls.gradAccum.value} \`
  --learning_rate ${controls.learningRate.value} \`
  --max_length ${controls.maxLength.value}${use4bit}`;
}

function buildInferenceCommand() {
  return `python scripts/inference.py \`
  --model_name ${controls.modelName.value} \`
  --adapter_path ${controls.adapterPath.value} \`
  --instruction ${shellQuote(controls.instructionInput.value)} \`
  --input ${shellQuote(controls.contextInput.value)}`;
}

function buildEvaluationCommand() {
  return `python scripts/evaluate_keywords.py \`
  --model_name ${controls.modelName.value} \`
  --adapter_path ${controls.adapterPath.value}`;
}

function updateCommands() {
  trainingCommand.textContent = buildTrainingCommand();
  inferenceCommand.textContent = buildInferenceCommand();
  evaluationCommand.textContent = buildEvaluationCommand();
}

function renderExamples() {
  const domain = domainFilter.value;
  const task = taskFilter.value;
  const visibleExamples = examples.filter((example) => {
    const domainMatch = domain === "all" || example.domain === domain;
    const taskMatch = task === "all" || example.task === task;
    return domainMatch && taskMatch;
  });

  exampleList.innerHTML = visibleExamples
    .map(
      (example) => `
        <article class="example-card">
          <header>
            <strong>${example.instruction}</strong>
            <span class="pill ${example.domain}">${example.domain}</span>
          </header>
          <p>${example.output}</p>
        </article>
      `,
    )
    .join("");
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 1600);
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
  showToast("Copied command");
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.view;
    navButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    views.forEach((view) => view.classList.toggle("is-visible", view.id === target));
  });
});

document.querySelectorAll("input, select, textarea").forEach((field) => {
  field.addEventListener("input", updateCommands);
  field.addEventListener("change", updateCommands);
});

domainFilter.addEventListener("change", renderExamples);
taskFilter.addEventListener("change", renderExamples);

document.querySelectorAll(".copy-command").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.querySelector(`#${button.dataset.copyTarget}`);
    copyText(target.textContent);
  });
});

document.querySelector("#copy-active-command").addEventListener("click", () => {
  const visibleCommand = document.querySelector(".view.is-visible .active-command code");
  copyText(visibleCommand.textContent);
});

window.addEventListener("DOMContentLoaded", () => {
  updateCommands();
  renderExamples();
  if (window.lucide) {
    window.lucide.createIcons();
  }
});
