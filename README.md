
<p align="center">
<img src="https://raw.githubusercontent.com/warakitworayuthanakarnLDC/KengGPT/main/KengGPT.png" alt="KengGPT Logo" width="200" height="200">
</p>
<h1 align="center">KengGPT</h1>
<p align="center">
<b>An ultra-lightweight, bilingual (Thai/English) RWKV language model designed to run locally on vintage mobile hardware (Samsung Galaxy S5).</b>
</p>
<p align="center">
<img src="https://img.shields.io/badge/Architecture-RWKV--5-blue" alt="Architecture">
<img src="https://img.shields.io/badge/Parameters-22M-green" alt="Parameters">
<img src="https://img.shields.io/badge/Quantization-Q4__0-orange" alt="Quantization">
<img src="https://img.shields.io/badge/Target-Samsung%20Galaxy%20S5-red" alt="Target Hardware">
</p>
📖 About KengGPT
KengGPT is a proof-of-concept project exploring extreme model compression and efficiency. While modern LLMs require massive server-grade GPUs, KengGPT is engineered from the ground up to run smoothly on constrained, low-memory mobile devices without relying on cloud APIs.
📊 Training Results (KengGPT 1.0)
KengGPT 1.0 was trained from scratch using a high-speed FP16 pipeline in Google Colab utilizing mixed precision (torch.amp) and optimized data loaders.
Epoch	Steps	Time	Iteration Speed	Final Step Loss	Avg Epoch Loss
1 / 2	1,207 / 1,207	38m 28s	1.91s / it	3.4230	6.6453
2 / 2	1,207 / 1,207	38m 24s	1.91s / it	2.9745	3.1847
Final Checkpoint Saved: /content/kenggpt_rwkv_22m.pth
Observation: The dramatic drop in cross-entropy loss from 6.64 to 3.18 confirms the model successfully learned syntax, narrative flow, and bilingual patterns.
⚙️ Specifications: KengGPT 1.0 ("Kark-0")
Target Hardware: Samsung Galaxy S5 (Qualcomm Snapdragon 801 32-bit CPU, 2 GB LPDDR3 RAM, running unlocked LineageOS).
Model Architecture: ~22 Million parameter RWKV recurrent state space model (n_embd=384, n_layer=8).
Tokenizer: Custom 4,096-vocabulary Byte-Pair Encoding (BPE) tokenizer (keng_tokenizer.json, ~284 KB) designed to fit token lookup tables into <2 MB of RAM.
Quantization & Binary: 4-bit integer quantized weights (kenggpt_q4_0.bin, ~13 MB binary footprint).
Runtime Engine: Native C++ JNI bridge (librwkv.so) cross-compiled for armeabi-v7a with ARM NEON SIMD vectorization.
Dataset & Persona: Trained on a blended bilingual corpus consisting of Mr. Keng's personal Thai/English Discord logs (boosted 10x), TinyStories, Thai Wikipedia, SCB MT English-Thai, and DailyDialog.
🚀 Roadmap: KengGPT 1.5 ("Solaris")
Scaled Capacity: Expanding to 35M–45M parameters (n_embd=448, n_layer=12) with tied word embeddings to increase reasoning capability without exceeding a strict 25 MB RAM budget.
Supervised Fine-Tuning (SFT): Introducing special control tokens (<|system|>, <|user|>, <|assistant|>) paired with loss masking to allow structured query handling and system prompt adherence.
Runtime Guardrails: Implementing Finite State Machine (FSM) / Grammar-Constrained Decoding at the C++ sampling layer to eliminate subword fragmentation and out-of-distribution hallucinations.
🛠️ Usage & Inference
To run inference interactively with the Claude-Code style loading animation (·, ✻, ✽, ✶, ✳, ✢) in Google Colab:
code
Python

python keng_gpt_interactive.py
📜 License
This project is open-source under the MIT License.
