# Qelys

&gt; Live-training LLM on **RTX 3050 4GB** (world's first)

## Status

- [x] Genesis (born on 2025-01-18)
- [x] v0.1: Qwen2.5-0.5B inference &lt; 500ms
- [ ] v0.2: Sync LoRU training without OOM
- [ ] v1.0: Async LoRU([AeloRU](https://github.com/jyimu/aeloru)) hot-reload (&lt; 5s)
- [ ] v2.0: Real-time Hebbian dual-expert
- [ ] v3.0: Production live streaming

----

## Research Question (v1.0/v2.0)

- **Q**: Can we achieve millisecond-level parameter updates on 4GB VRAM using AsyncLoRU?
- **Hypothesis**: Yes, by decoupling inference and training with dual-buffer LoRU vectors.
- **Method**: Implement, measure, iterate.

----

## Hardware (The Challenge)

- **GPU**: RTX 3050 4GB (extreme low VRAM) — because constraints breed innovation
- **RAM**: 16GB DDR4
- **OS**: Win10 25H2 + WSL2

----

## Architecture

```bash
Qelys (Application Layer)
↓ uses
AeloRu (Async Edge Low-Rank Update)
↓ enables
Real-time learning on consumer GPUs
```

- **Base Model**: Qwen2.5-0.5B (4-bit quantized)
- **Memory Engine**: [AeloRu](https://github.com/jyimu/aeloru) — async LoRU framework
- **Key Innovation**: Dual-buffer s-vector switching (microsecond latency)

----

## Quick Start

```bash
# Clone & install
git clone https://github.com/jyimu/Qelys.git
cd Qelys
pip install -r requirements.txt
```

## Roadmap

| Phase       | Goal                        |
| ----------- | --------------------------- |
| 2025 Winter | Sync LoRU validation (V0.2) |
| 2025 Spring | Async dual-buffer (V1.0)    |
| 2026 Summer | Hebbian real-time (V2.0)    |
| 2027+       | Production + Paper          |

## Paper Target

- **Title**: AsyncLoRU: Asynchronous Low-Rank Updates for Real-Time Edge Learning
- **Venue**: NeurIPS/ICML Workshop on Efficient ML
- **Claim**: First to achieve sub-100ms parameter updates on 4GB consumer GPUs.
- **Author**: JYimu ([@jyimu](https://github.com/jyimu))

## Author

jyimu ([@jyimu](https://github.com/jyimu)) - a high school student with a passion for pushing the limits of AI on consumer hardware. Always eager to learn and share knowledge.
