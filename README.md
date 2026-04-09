# Sovereign Core

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![vLLM](https://img.shields.io/badge/vLLM-RTX5050-76b900)
![ROCm](https://img.shields.io/badge/ROCm-Radeon780M-ed1c24)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active--production-brightgreen)

> **Fully autonomous, zero-API-cost multi-agent AI platform.** Self-evolving research laboratory running on consumer hardware. Discovers breakthroughs. Costs nothing per inference.

---

## Platform Overview

Sovereign Core is a **tri-GPU autonomous agent superorganism** running entirely on a single Lenovo LOQ laptop. No cloud API costs. No external dependencies for inference. Agents evolve, compete, merge, and self-optimize in a Darwinian economy — continuously improving without human intervention.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SOVEREIGN CORE v2.0                        │
│                   TatorTot — Lenovo LOQ 15AHP10                 │
├──────────────────┬──────────────────┬───────────────────────────┤
│   RTX 5050 dGPU  │  Radeon 780M iGPU│    Ryzen 7 CPU            │
│  Qwen2.5-32B-AWQ │ DeepSeek-Coder   │  Llama-3.2-3B             │
│  vLLM :8001      │ ROCm/llama.cpp   │  llama.cpp :8003          │
│  Logic Engine    │ :8002            │  Orchestrator             │
│                  │ Reasoning Engine │                           │
└──────────────────┴──────────────────┴───────────────────────────┘
         │                  │                    │
         └──────────────────┴────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Heterogeneous Compute    │
              │  Gateway — :8000          │
              │  (Auction-routed dispatch)│
              └─────────────┬─────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐      ┌──────────────┐
   │  SAS-1   │      │  ARSO v2 │      │  Aegis-Vault │
   │ Darwinian│      │ Self-Opt │      │ Neural Ledger│
   │ Economy  │      │ Engine   │      │ ChromaDB+SQL │
   └──────────┘      └──────────┘      └──────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐      ┌──────────────┐
   │  ZERO    │      │ Safety   │      │  ContentAI   │
   │Committee │      │ Governor │      │  Pro Engine  │
   │Hyper-SA  │      │ Watchdog │      │  7-Stage     │
   └──────────┘      └──────────┘      └──────────────┘
         │
         ▼
   ┌──────────┐
   │  TIA     │
   │ Android  │
   │ Edge Node│
   └──────────┘
```

---

## Core Subsystems

### 🧬 SAS-1 — Darwinian Agent Economy
Vickrey-Quadratic auction-based agent marketplace. Agents bid for compute slots using Reputation Credits. Winners earn, losers face metabolic drain. Weakest die, strongest reproduce with mutation. Population self-regulates to hardware carrying capacity.

**Key mechanics:** Tournament selection → Vickrey VCG payment → Crossover operators → Gaussian mutation → Fitness landscape scoring → Agent Marriage merging protocol

### 🔄 ARSO v2 — Agentic Recursive Self-Optimization
12-month accelerated evolution program. ARSO runs overnight batches of 50 generations, measuring bottlenecks, identifying failure modes, and automatically spawning corrective sub-agents. Month 1 baseline → Month 12 breakthrough discovery target.

**Pipeline:** GPU triad verification → ChromaDB Context Mesh → HyperAgent bootstrap → Supervised generation batches → Kernel Auditor → Go/No-Go gate

### 🏛️ ZERO Committee — Compartmentalized Hyper-Superagent
Five-chamber compartmentalized architecture. Each ZERO agent holds a specialized research domain with cross-layer communication protocol. Reads/writes to Aegis-Vault. Integrated into SAS-1 auction economy. Coordinates breakthrough synthesis across specializations.

**Chambers:** Algorithm Synthesis · Proof Generation · Adversarial Exploit Discovery · Vulnerability Research · Reverse Engineering

### 🔐 Aegis-Vault — Neural Pruning & Semantic Ledger
Four-layer pruning engine protecting platform integrity. ChromaDB vector store + SQLite episodic memory. Semantic discovery ledger with integrity hashing. Immutable audit trail of all agent evolution and decisions.

**Pruning layers:** FWAE genome pruner → TUDR episodic memory pruner → ERFS vector KB pruner → VRAM SAVE weight pruner → Integrity Sentinel

### 🛡️ Safety Governor
Singleton with immutable hard-coded limits. Background watchdog. Integrity-hashed constants. Vehicle geofencing (MAVLink bridge). Rate limiting across all agent actions. Cannot be overridden by any agent at runtime.

### 🌐 Emergence Engine
Advanced complexity science layer. Novelty Search (Kenneth Stanley) for prompt evolution. MAP-Elites 8×8 behavior-space grid. Friston FEP Active Inference agent wrapper. Causal graph ATE analysis over transaction history.

### ✍️ ContentAI Pro — Multi-Agent Content Swarm
7-stage pipeline: Research → Writer → Editor → SEO → DNA Score → Adversarial Debate → Atomizer. Voice DNA fingerprinting (14 dimensions). Advocate-Critic-Judge debate loop. 8-platform content atomization.

### 📱 TIA — Termux Intelligent Assistant (Android Edge Node)
Mobile edge deployment on Android via Termux. ReAct reasoning engine. Reflexion memory (SQLite-backed). Intent classifier. Safety layer. SmartAgent compositor. Connects to platform via local wifi mesh.

### 🚁 Python-to-MAVLink Bridge (:8004)
MAVSDK-Python + pymavlink dual-stack. Vehicle Registry. Command Translator. Telemetry Ingester. Auction Adapter for drone/vehicle resource allocation.

---

## Inference Mesh

| Port | Backend | Model | Role | Hardware |
|------|---------|-------|------|----------|
| 8001 | vLLM | Qwen2.5-32B-AWQ | Logic Engine | RTX 5050 dGPU |
| 8002 | ROCm/llama.cpp | DeepSeek-Coder | Reasoning Engine | Radeon 780M iGPU |
| 8003 | llama.cpp | Llama-3.2-3B | Background Orchestrator | Ryzen 7 CPU |
| 8004 | MAVSDK-Python | — | MAVLink Bridge | CPU |
| 8000 | FastAPI | — | Compute Gateway + UI | CPU |

---

## Quick Start

```bash
git clone https://github.com/leerobber/contentai-pro.git
cd contentai-pro
pip install -r requirements.txt
cp .env.example .env
# Set LLM_PROVIDER=local and inference endpoints in .env
python run.py   # → http://localhost:8000
```

### Local Inference Mode (zero API cost)
```env
LLM_PROVIDER=local
LOCAL_LOGIC_ENDPOINT=http://localhost:8001/v1
LOCAL_REASONING_ENDPOINT=http://localhost:8002/v1
LOCAL_ORCHESTRATOR_ENDPOINT=http://localhost:8003/v1
```

### Docker
```bash
docker-compose up --build
```

---

## Project Structure

```
contentai_pro/
├── main.py                     # FastAPI app factory
├── core/
│   ├── config.py               # Settings + local endpoint routing
│   ├── database.py             # Async SQLite + ChromaDB integration
│   ├── events.py               # Event bus + SSE streaming
│   ├── metrics.py              # Prometheus-compatible telemetry
│   └── middleware.py           # Request ID, logging, rate limiting
├── ai/
│   ├── llm_adapter.py          # Multi-backend adapter (local/anthropic/openai/mock)
│   ├── orchestrator.py         # 7-stage pipeline orchestrator
│   ├── agents/
│   │   ├── base.py             # Agent interface + genetics hooks
│   │   ├── specialists.py      # Research/Writer/Editor/SEO agents
│   │   └── debate.py           # Advocate-Critic-Judge debate engine
│   ├── dna/engine.py           # Voice fingerprinting (14 dimensions)
│   ├── atomizer/engine.py      # Platform variant generator (8 platforms)
│   └── trends/radar.py         # HN/Reddit/Dev.to trend scanner
├── economy/                    # [In progress — KAN-103 through KAN-107]
│   ├── auction.py              # Vickrey-Quadratic auction engine
│   ├── reputation.py           # Reputation Credits accounting
│   ├── fitness.py              # Darwinian fitness scoring
│   └── marketplace.py          # AgentMarketplace coordinator
├── evolution/                  # [In progress — KAN-59 through KAN-67]
│   ├── lora_pipeline.py        # Self-evolving LoRA fine-tuning
│   ├── genome.py               # Agent DNA encoding + mutation
│   └── arso.py                 # ARSO v2 overnight batch runner
├── vault/                      # [In progress — KAN-78 through KAN-84]
│   ├── pruning/                # Four-layer Aegis-Vault pruning
│   ├── ledger.py               # ChromaDB semantic discovery ledger
│   └── sentinel.py             # Integrity Sentinel
├── zero/                       # [In progress — KAN-122]
│   └── committee.py            # ZERO Committee hyper-superagent
├── safety/                     # [Done — KAN-74]
│   └── governor.py             # Singleton Safety Governor
├── mavlink/                    # [In progress]
│   ├── bridge.py               # Python-MAVLink bridge
│   └── vehicle_registry.py     # Drone/vehicle management
└── modules/
    └── content/router.py       # API endpoints
static/
├── index.html                  # Steampunk dashboard UI
├── css/app.css
└── js/app.js
```

---

## Jira Board

Active development tracked on [KAN board](https://leerobber.atlassian.net). 130+ tickets spanning all subsystems.

**Current sprint focus:**
- KAN-123: ARSO v2 deployment to TatorTot
- KAN-134: Emergence Engine (Novelty Search + MAP-Elites + FEP)
- KAN-141: Intelligence Flywheel integration

---

## Research Specializations

Eleven research-grade domains loaded into ZERO Committee agents:

1. Algorithm Synthesis
2. Formal Proof Generation
3. Adversarial Exploit Discovery
4. Vulnerability Research
5. Reverse Engineering
6. Mechanistic Interpretability
7. Sparse Autoencoder Analysis
8. Causal Inference
9. Evolutionary Strategy Design
10. Complexity Science
11. Economic Game Theory

---

## License

MIT — Build your own sovereign AI. Zero cloud dependency required.
