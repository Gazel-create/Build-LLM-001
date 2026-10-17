# How Far Can a Laptop Go? An Analysis of a From-Scratch GPT Implementation

**Author:** Satyam Rai (`@Gazel-create`)  
**Course:** COMP S333W — Advanced Programming and AI Algorithms  

## Overview

Most work with Large Language Models treats them as black-box APIs or assumes you need dedicated GPUs to train and test them. In this project, I build and analyse a from-scratch GPT implementation running entirely on a single consumer laptop CPU inside WSL2, measuring how model size, decoding strategies, and training-loop improvements affect text quality, throughput (`tokens/s`), and memory usage (`RES`).

## Code Attribution

* **`gpt_train.py` and `previous_chapters.py`:**  
  Adapted from Sebastian Raschka's [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) (*Build a Large Language Model (From Scratch)*, Manning, 2024). I added per-step wall-clock timing, `tokens/s` throughput logging, and CPU fallback tracking to benchmark the 124M configuration on my laptop.
* **`chapter2_dataloader.py` (and Chapter 2–5 modules):**  
  My own from-scratch implementation of the Chapter 2 BPE tokenizer pipeline (`tiktoken`), sliding-window `GPTDatasetV1`, and PyTorch `DataLoader`.

## Log Files

1. **`cpu_benchmark.log`**  
   Logs the preliminary CPU training run of the GPT-2 124M configuration (`context=256`, `emb=768`, `12 layers`, `12 heads`, `batch_size=2`, `lr=5e-4`, `seed=123`) on *The Verdict*. Records the CUDA driver fallback to CPU, per-step throughput (~95–125 tokens/s, 492.5 s total across 85 steps), the step-20 training loss spike (`15.77`), and validation overfitting after step 55 (min validation loss `6.16`, rising to `6.54` while training loss drops to `0.36`).
2. **`continuous_system_log.txt`**  
   Linux `top` resource log captured during the benchmark run inside WSL2 (7,845.7 MiB total visible memory). Shows Python resident memory (`RES`) peaking at ~4.5 GB (lowest free RAM ~636 MiB, zero swap used) and CPU utilization averaging ~361% (max ~394%, saturating 4 cores).
