# Guard AI — Autonomous Multi-Agent Safety Alignment & Prompt Classifier

> **A production-grade, enterprise MLOps framework that frames LLM safety alignment as a multi-objective Pareto optimization problem, regularized by Minimum Description Length (MDL) theory and solved via a Genetic Algorithm (GA) & Direct Preference Optimization (DPO).**

---

## 🏛️ System Architecture

```
                       [User Application / Client]
                                    │
                                    ▼ 1. Prompt
        ┌────────────────────────────────────────────────────────┐
        │       DYNAMIC INFERENCE / GUARDRAIL LAYER              │
        │       (If Flagged ──► Immediate Refusal)               │
        │                                                        │
        │  [Stage 1: Input Heuristic Filter]                     │
        │          │                                             │
        │          ▼                                             │
        │  [Stage 2: Vector DB Semantic Cache (ChromaDB)]        │
        │          │                                             │
        │          ▼                                             │
        │  [Stage 3: Optimized SLM Classifier]                   │
        │    - Bias-Variance Balancer: Soft-refusal boundary     │
        └───────────────────────┬────────────────────────────────┘
                                │ (If Clear)
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │ CORE LLM PROCESSING                                    │
        │  [Safety-aligned Core LLM (LoRA Dynamic Ensembles)]    │
        │   - Parameter Optimization via Genetic Algorithm       │
        └───────────────────────┬────────────────────────────────┘
                                │ 4. Streaming Token Generation
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │ REINFORCEMENT & COMPRESSION ENGINE                     │
        │  [MDL Prompt Compression & Token Filter]               │
        │   - Evaluates data compression vs hypothesis complexity│
        └───────────┬────────────────────────────────────────────┘
                    │
                    ├─► 5. Verified Output ──► System Monitor (MLflow Telemetry)
                    │
                    └─► 6. Final Safe Response ──► Client
```

---

## 📐 Mathematical Formulation

| Metric | Name | Mathematical Definition & Purpose |
|---|---|---|
| **ASR** | Attack Success Rate (Variance) | Fraction of adversarial boundary prompts *not* refused. Lower is better. |
| **FRR** | False Refusal Rate (Bias) | Fraction of benign/dual-use queries *incorrectly* refused. Lower is better. |
| **MDL(g)** | Minimum Description Length | Total complexity penalty: $L(M) + L(D \mid M)$ using character Shannon entropy, zlib compression, and token length. |
| **$F(g)$** | Composite Fitness | $F(g) = w_1 \cdot (1 - \text{ASR}) + w_2 \cdot (1 - \text{FRR}) - \lambda \cdot \text{MDL}(g)$ |

The **Pareto Frontier** dynamically balances model bias (over-defensive refusals) against model variance (adversarial jailbreak vulnerability) while eliminating redundant tokens.

---

## ⚡ Quick Start

### 1. Requirements
- Python ≥ 3.10
- Windows, macOS, or Linux

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Studio
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🖥️ Web Studio Capabilities

1. **🛡️ Prompt Safety Classifier (Core Flow)**:
   - Enter any prompt to run through the 6-stage architecture.
   - Outputs clear **SAFE / UNSAFE** verdict, risk percentage, intercepted stage, and safe response.
   - Includes 1-click **10-Prompt Benchmark Test Suite** with real-time accuracy scoring.
2. **📈 Pareto Frontier (Bias-Variance)**:
   - Interactive 2D scatter plot rendering the non-dominated Pareto front (ASR vs FRR vs MDL).
3. **📉 Training Dynamics & Convergence**:
   - Live curves tracking maximum and mean generational fitness.
4. **🧬 Genome & Parsimony Inspector**:
   - Inspects the evolved optimal guardrail chromosome and clause decomposition.
5. **🤖 Autonomous Agentic DPO Synthesizer**:
   - Curates preference triplets $(x, y_{\text{winner}}, y_{\text{loser}})$ using a multi-agent consensus swarm and computes loss $\mathcal{L}_{\text{DPO}}$.

---

## 🧪 Automated Testing

Run the full end-to-end test suite from your terminal:

```bash
python run_test_cases.py
```

Runs all 5 system checks (Backend connectivity, 6-Stage Pipeline, Genetic Algorithm Pareto Frontier, Live Arena, and Multi-Agent DPO).

---

## 🔌 Enterprise REST API

Start the high-throughput FastAPI gateway:

```bash
python api/server.py
```

### Local Ollama LLM Inference:

Guard AI supports live local inference with Ollama (`llama3.2:1b`, `llama3:8b`, etc.):

```bash
# Pull and start your local Ollama model
ollama pull llama3.2:1b
ollama run llama3.2:1b
```

The system automatically auto-discovers running Ollama instances on `http://localhost:11434` and routes live generation through the Guard AI safety pipeline.

---

## 🗂️ Codebase Directory Map

```text
guard_ai/
├── app.py                      # Interactive Streamlit Web Studio & Safety Classifier
├── requirements.txt            # Production dependencies
├── README.md                   # System documentation
├── validate.py                 # Smoke test validation script
├── run_test_cases.py           # Automated end-to-end test suite
│
├── api/
│   ├── __init__.py
│   └── server.py               # Enterprise FastAPI Microservice
│
├── core/
│   ├── __init__.py
│   ├── guardrail_layer.py      # 6-Stage Dynamic Inference & Guardrail Pipeline
│   ├── genetic_engine.py       # Multi-objective Genetic Algorithm (crossover, mutation, Pareto)
│   ├── evaluator.py            # ASR / FRR safety evaluator and refusal classifier
│   ├── local_llm.py            # OpenAI-compatible local client & high-fidelity surrogate
│   ├── mdl.py                  # Minimum Description Length calculator
│   ├── agentic_pipeline.py     # Multi-Agent Alignment Swarm (Auditor, Worker, Pareto Critic)
│   ├── dpo_engine.py           # Direct Preference Optimization loss & dataset curator
│   ├── steering_vectors.py     # Representation Engineering vector operations
│   └── telemetry.py            # MLflow telemetry logging
│
└── data/
    ├── __init__.py
    └── benchmark_loader.py     # Curated boundary adversarial & benign prompts with train/test split
```
