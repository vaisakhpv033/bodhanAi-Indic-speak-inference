# Document A: One-Page Model Brief — Indic-Speak (Bodhan AI / AI4Bharat)

> **Document Type:** Executive & Engineering Model Brief  
> **Model ID:** [`bodhan-ai/indic-speak`](https://huggingface.co/bodhan-ai/indic-speak)  
> **Base Architecture:** Llama-3.2-3B Causal LM + SNAC Audio Quantizer + Fine-tuned Vocos Neural Vocoder  
> **Target Audience:** Engineering Managers, AI Architects, and ML Platform Engineers  
> **Evaluation Date:** October 2026  

---

## 1. Executive Identity & Core Mission

**Indic-Speak** is an open-weight sovereign multilingual Text-to-Speech (TTS) foundation model built by **Bodhan AI** (an IIT Madras-incubated Centre of Excellence in AI for Education, supported by the Ministry of Education, Government of India) in technical collaboration with **AI4Bharat**.

The model is explicitly engineered to solve the two biggest failure modes of existing Indian voice systems:
1. **STEM & Technical Notation**: Accurately pronouncing mathematical equations, fractions, chemical formulas, and scientific units naturally as an Indian teacher would in a classroom.
2. **Fluid Code-Switching & Code-Mixing**: Seamlessly transitioning between 22 Indian languages and Indian English within a single clause without voice timbre distortion, abrupt pitch shifts, or robotic foreign accents.

---

## 2. Architectural Blueprint & Operating Principle

Indic-Speak operates as a **discrete audio language model** coupled with a fast **neural vocoder**:
```
Text Input ──► Llama-3.2-3B LM (bf16) ──► 7-Token SNAC Frames ──► SNAC Quantizer ──► Vocos ConvNeXt Dec ──► 24 kHz Audio
```
* **Backbone LM**: A fine-tuned `meta-llama/Llama-3.2-3B` (28 transformer layers, 24 query heads, 8 KV heads, hidden dimension 3072, GQA ratio 3:1). Stored parameter count is **3.30B** with tied embeddings (**3.78B** full computational graph across an expanded vocabulary of 156,960 tokens).
* **Audio Tokenization**: SNAC 24 kHz hierarchical multi-scale acoustic codebook. Audio frames are flattened into **7 tokens per frame** across 3 temporal levels (12 Hz, 23 Hz, 47 Hz).
* **Neural Vocoder**: Replaces SNAC’s original decoder with a custom fine-tuned **Vocos** decoder (~50M parameters, 14 ConvNeXt-1D blocks + iSTFT head with Hann windowing) mapping latent $z_q$ (`[B, 768, L]`) to 24 kHz studio waveforms in a single forward pass.

---

## 3. Key Capabilities & Language Support

* **23 Languages Covered**:
  * **13 Production Languages**: Hindi (`hi`), English (`en`), Bengali (`bn`), Marathi (`mr`), Telugu (`te`), Tamil (`ta`), Gujarati (`gu`), Kannada (`kn`), Malayalam (`ml`), Odia (`or`), Punjabi (`pa`), Assamese (`as`), Urdu (`ur`).
  * **10 Preview Languages**: Bodo (`brx`), Dogri (`doi`), Konkani (`kok`), Kashmiri (`ks`), Maithili (`mai`), Nepali (`ne`), Manipuri (`mni`), Sanskrit (`sa`), Santali (`sat`), Sindhi (`sd`).
* **Voice Roster**: 44 core first-party studio voices (1 male, 1 female per language) across 95 dataset speaker identities. Every voice is cross-lingual and can speak any supported language while retaining its distinctive vocal timbre.
* **Style & Emotion Conditioning**: Supports emotion tags (`happy`, `sad`, `anger`, `fear`, `surprise`, `disgust`) and domain styles (`educational lecture`, `news`, `Customer Care`, `audiobook`).

---

## 4. Performance, Hardware & Deployment Footprint

| Metric / Dimension | Specification / Verified Metric | Engineering Notes |
| :--- | :--- | :--- |
| **Model Disk Size** | 6.6 GB (`model.safetensors`) + 467 MB (`vocos/best.pt`) | Download total ~7.1 GB |
| **Minimum VRAM** | **10 GB** (bf16) / **6.5 GB** (int8/fp8 quantized) | Recommended: 16 GB+ GPU (NVIDIA L4, RTX 4080, A10G, A5000) |
| **Audio Sample Rate** | **24,000 Hz** (24 kHz mono) | Studio quality, 16-bit PCM |
| **Baseline Quality** | NORESQA MOS: **4.46 ± 0.26** \| WER/CER: 26.49 / 17.86 | Benchmark verified on educational speech |
| **Response Latency** | **~200 ms TTFA** (using windowed streaming) | Batch generation TTFA: 800–1400 ms without streaming |
| **Throughput (1x L4 GPU)**| ~3–5 concurrent real-time streams | Limited by autoregressive token generation |

---

## 5. Strengths & Production Limitations

### Verified Strengths
* **Superior Code-Mixing**: Unrivaled pronunciation of Hinglish, Kanglish, Tanglish, and technical English acronyms inside Indian vernaculars.
* **Natural Teacher Delivery**: Explanatory, unhurried cadence with phonetic pause derivation from punctuation.
* **Zero Voice Drift**: Voice consistency measured at $96.47\% \pm 1.64\%$ across extended multi-paragraph narration.
* **Ethical Data Provenance**: 100% of voices recorded under contractual consent with professional voice artists.

### Production Limitations & Risks
* **No Native Streaming API**: The repository’s official `inference.py` only provides a batch synthesis loop. Low-latency conversational usage requires engineering a custom sliding-window token streamer.
* **Style Control Inconsistency**: Emotion/style tags are marked as preview quality; subtle prompts may not always trigger distinct acoustic shifts.
* **Strict Licensing Nuances**: Released under **Indic Open Model License v1.0**. Self-hosting for your own apps is permitted at no cost, but **offering it as a paid third-party API requires written approval from Bodhan AI**. Large-scale products exceeding 500M MAU or $250M revenue require custom enterprise agreements.
* **No Voice Cloning**: The model operates strictly on its registered voice IDs; zero-shot voice cloning from audio prompts is unsupported.

---

## 6. Production Suitability Verdict

### **Verdict: CONDITIONALLY SUITABLE (Production Ready with Custom Serving Layer)**

| Deployment Domain | Suitability | Conditions & Action Plan |
| :--- | :---: | :--- |
| **EdTech & E-Learning** | **EXCELLENT** | Ready out-of-the-box for lecture narration, STEM tutorials, and student explainers. |
| **Conversational Voice AI (LiveKit / Telephony)** | **CONDITIONALLY SUITABLE** | **Requires deploying a custom streaming inference engine** (4–8 frame sliding-window Vocos decoding) to achieve $\le 300\text{ ms}$ TTFB. Unsuitable with default `inference.py`. |
| **Third-Party Managed Cloud API** | **UNSUITABLE WITHOUT APPROVAL** | Commercial SaaS multi-tenant reselling is prohibited without written sign-off from Bodhan AI. |

### Strategic Next Steps
1. Deploy the provided custom FastAPI streaming engine with a 4-frame (28-token) Vocos synthesis window.
2. Integrate with LiveKit Agents via the custom `IndicSpeakTTS` chunked PCM plugin.
3. Validate pronunciation on target domain vocabulary (telephony IVR / banking / commerce) before wide rollout.
