# 🧠 Kbidu AI – 30B LLM (Cybersecurity & Code)

## ⚠️ Disclaimer / Use at Your Own Risk

**This model is provided "as is", without any warranty of any kind.**
- The model may produce incorrect, harmful, or biased outputs.
- You are **solely responsible** for how you use this model.
- The creators assume **no liability** for any damages, losses, or legal issues arising from its use.
- **Do not use this model** for illegal, unethical, or malicious purposes.
- This is an **experimental research project** – not a production‑grade system.

---

## 📌 Project Overview

**Kbidu AI** is a 30‑billion‑parameter decoder‑only transformer (causal language model) built from scratch. It is specialised for:
- **Cybersecurity** – vulnerabilities, exploits, cryptography, threat intelligence.
- **Coding** – Python, C, Java, C++, and more.
- **Advanced reasoning** – math, logic, step‑by‑step problem solving.

The model is designed to:
- **Think internally** before answering (`think_before_answer`).
- **Retrieve facts** from a knowledge base (RAG).
- **Reason step‑by‑step** (Chain‑of‑Thought, Tree‑of‑Thought).
- **Verify its answers** (self‑consistency).
- **Remember conversations** (short‑term memory).
- **Call tools** (agent mode with `[TOOL: ...]` syntax).
- **Express human‑like emotion** (empathetic, professional, humorous, supportive).

---

## 🏗️ Architecture

| Component | Implementation |
| :--- | :--- |
| **Normalisation** | RMSNorm |
| **Activation** | SwiGLU |
| **Positional Encoding** | Rotary Positional Embeddings (RoPE) |
| **Attention** | Grouped‑Query Attention (GQA) + Flash Attention |
| **Training** | FSDP (Fully Sharded Data Parallel) + Gradient Checkpointing |
| **Quantization** | 4‑bit (NF4) via `bitsandbytes` |
| **RAG** | FAISS + Sentence‑Transformers (`all-MiniLM-L6-v2`) |
| **Reasoning** | Chain‑of‑Thought, Tree‑of‑Thought, Self‑Consistency |
| **Agent** | Tool‑calling with `[TOOL: tool_name(params)]` |
| **Safety** | Rule‑based + classifier‑based content filtering |

---

## 📁 File Structure
