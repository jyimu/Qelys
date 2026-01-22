# Qelys

> Live-training LLM on **RTX 3050 4GB** (world's first)

## Status
- [x] Genesis (born on 2025-01-18)
- [ ] v0.1: Qwen2.5-0.5B inference < 500ms
- [ ] v0.2: LoRA fine-tuning on 4GB VRAM
- [ ] v1.0: Manual hot-reload (< 30s)
- [ ] v2.0: Automatic live update

## Research Question (v1.0/v2.0)
- **Q**: What is the minimum latency of parameter updates on a 4GB VRAM GPU? and live-training can be true in 4GB VRAM or not?
- **Hypothesis**: >30s due to memory fragmentation and load overhead. 
- **Method**: Measure, not implement.

## Hardware (The Challenge)
- **GPU**: RTX 3050 4GB (extreme low VRAM)(because I only have it)
- **RAM**: 16GB DDR4
- **OS**: Win10 25H2 (WSL2)

## Quick Start
```bash
# Clone & install
git clone https://github.com/jyimu/Qelys.git
cd Qelys
pip install -r requirements.txt

# Run Qwen2.5-0.5B with 4-bit quantization
python chat.py --model Qwen/Qwen2.5-0.5B-Instruct --quantization 4bit
```

## Roadmap
2025 Winter: 0.5B inference in 4GB (baseline)
2026 Spring: LoRA 1-step training in < 10s
2027 Summer: Hot-swapping without OOM
2028 Goal: Paper "Real-Time LLM on 4GB GPUs"
## Author
JYIMU(a high school student)
"Maybe I can't make Quick successfully but I will try my best!."
## License
MIT
