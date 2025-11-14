# 🧠 Machine Learning & AI Engineering Portfolio  
### by Darrell Mortalla (dmortalla)

This repository showcases a complete, production‑ready **ML & AI Engineering portfolio**, featuring:

- A **Flagship AI Assistant** (RAG + Memory + Tools + Streamlit)
- 7 fully interactive **Hero Apps**
- A multi‑app **Suite Launcher**
- 6 advanced **Phase II ML Engineering Projects**
- Clean architecture, modular design, and Google‑style docstrings

It is structured to demonstrate **modern AI engineering depth and breadth**, targeting roles in:
Machine Learning Engineer, AI Engineer, Applied Scientist, and LLM Engineer.

---

# ⭐ 1. Flagship AI Assistant (Crown Jewel Project)

**Directory:** `flagship-ai-assistant/`  
**Live Demo:** _Add Hostinger link when deployed_

A full personal AI assistant with:

### 🔹 Retrieval-Augmented Generation (RAG)
- Upload `.txt`, `.md`, `.pdf`
- TF‑IDF–based document search
- Context‑aware LLM responses

### 🔹 Long‑Term Memory
- JSON‑based persistent memory  
- Learns preferences (e.g., “I prefer short summaries”)

### 🔹 Tool Calling
- Safe AST‑based calculator  
- Extensible architecture for new tools

### 🔹 LLM‑powered Conversation
- OpenAI Chat Completions  
- Graceful fallback if no API key available

### 🔹 Clean Streamlit UI
- Chat interface  
- Document panel  
- Memory viewer  
- Tool call log

### 🔹 Docker‑ready Deployment
```
docker build -t flagship-ai-assistant .
docker run -p 8501:8501 flagship-ai-assistant
```

---

# ⚡ 2. Hero Apps (7 Interactive ML/AI Demos)

**Directory:** `hero/`

These apps display breadth and real‑time interactivity:

- **RAG Document Advisor**
- **Multimodal RAG Assistant**
- **Semantic Search (FAISS)**
- **ALS Recommender**
- **Time Series Forecaster**
- **Traffic Sign Classifier (CNN)**
- **ECG Autoencoder Anomaly Detector**

Each app includes Streamlit UI + clean backend design.

---

# 🛠️ 3. Suite (Multi‑App Launcher + CLI)

**Directory:** `suite/`

A unified dashboard that:
- Shows all Hero Apps
- Provides one‑click launches
- Includes a Python CLI utility

Useful as a recruiter demo hub.

---

# 🔬 4. Phase II ML Engineering Projects

**Directory:** `projects/`

These demonstrate deep ML knowledge, engineering practices, and classical+modern skills.

### **1. FastAPI Deployment (Model Serving)**
`fastapi-deployment/`  
REST API for inference, Dockerized, clean schema validation.

### **2. MLflow Tracking + Model Registry**
`mlflow-tracking/`  
Full experiment tracking pipeline.

### **3. Anomaly Ensemble (IsolationForest + LOF)**
`anomaly-ensemble/`  
Combined anomaly scoring for ops/fraud detection.

### **4. HAR LSTM Sequence Model**
`har-sequence-model/`  
LSTM classifier for human activity (PyTorch).

### **5. Transformer Time Series Forecaster**
`time-series-transformer/`  
Custom Transformer Encoder for forecasting.

### **6. Graph Neural Network (GCN on CORA)**
`graph-gnn-cora/`  
PyTorch Geometric GCN for node classification.

---

# 📁 Repository Structure

```text
mlai-portfolio/
│
├── flagship-ai-assistant/
│
├── hero/
│
├── suite/
│
├── projects/
│
├── README.md
└── .github/workflows/
```

---

# 🎯 Recruiter Summary

This portfolio demonstrates:

- LLM integration & prompt engineering  
- RAG systems & vector search  
- Streamlit UI development  
- API design with FastAPI  
- Experiment tracking (MLflow)  
- Deep learning models (LSTM, CNN, Transformers)  
- Graph ML with PyTorch Geometric  
- End‑to‑end deployment workflows  
- Clean, well‑documented engineering practices

Together, these projects form a complete, modern ML/AI Engineering portfolio.

---

# 🔗 Next Steps (for Deployment)

- Add Hostinger deployment links  
- Add GitHub Pages portfolio overview  
- Add screenshots to enhance visual appeal  
- Add CI/CD pipelines for rebuilding apps  

---

# © 2025 Darrell Mortalla  
**dmortalla.com** | Machine Learning & AI Engineering Portfolio
