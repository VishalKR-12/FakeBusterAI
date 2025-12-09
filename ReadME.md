🔥 Unified AI Content Shield
Real-Time Detection of Toxicity, Fake News, Deepfakes & Voice Cloning
📁 Project Structure

This repository follows a modular, scalable architecture for combining a Chrome Extension with a FastAPI-based AI backend.

Unified-AI-Content-Shield/
│
├── chrome-extension/
│   ├── manifest.json            # Chrome Extension manifest (MV3)
│   ├── popup.html               # UI for extension popup
│   ├── popup.js                 # Popup logic (fetch scores, UI updates)
│   ├── background.js            # Service worker (handles events, messaging)
│   ├── content_script.js        # Injected script to scan webpage content
│   ├── overlay.css              # Style for on-page trust score overlay
│   └── overlay.js               # Logic to render trust badges & warnings
│
├── backend/
│   ├── main.py                  # FastAPI entry point: mounts all routers
│   ├── requirements.txt         # Python dependencies
│   ├── config.py                # Environment configs, API keys, model paths
│   │
│   ├── routers/                 # API endpoints for each AI module
│   │   ├── text_router.py       # POST /analyze/text
│   │   ├── vision_router.py     # POST /analyze/vision
│   │   ├── audio_router.py      # POST /analyze/audio
│   │   └── fusion_router.py     # POST /analyze/fusion (multimodal score)
│   │
│   ├── models/                  # Model weights and model loaders
│   │   ├── text/                # BERT, RoBERTa, fake-news classifier
│   │   ├── vision/              # Deepfake CNN / ViT models
│   │   └── audio/               # Voice-clone and spoofing models
│   │
│   ├── services/                # Business logic for each AI domain
│   │   ├── text_service.py      # Toxicity, hate speech, fake news logic
│   │   ├── vision_service.py    # Deepfake and manipulation detection
│   │   └── audio_service.py     # Voice clone, frequency analysis
│   │
│   └── utils/
│       ├── preprocess.py        # Preprocessing for text, frames, audio
│       └── scoring.py           # Unified content trust score logic
│
└── README.md                    # Documentation

🧠 System Overview

The Unified AI Content Shield is a multimodal trust & safety system that detects:
✔ Toxic or harmful text
✔ Hate speech & harassment
✔ Fake news and misinformation
✔ Deepfake images/videos
✔ Voice cloning and manipulated audio

It works in real time as you browse the web through a Chrome Extension and communicates with the backend using REST APIs / WebSockets.

🚀 Key Components
### 🟦 1. Chrome Extension (Frontend Inference Layer)

Extracts text, images, and video frames from the current webpage

Sends them to backend AI models for analysis

Displays:

🔰 Content Trust Score

⚠️ Warnings for misinformation or toxicity

🧯 Red flags for deepfakes

Injects a live On-Screen Overlay Badge

🟩 2. Backend (FastAPI + ML Pipeline)

A complete microservice-style backend with 4 routers:

/analyze/text → Hate speech, toxicity, fake news

/analyze/vision → Deepfake and visual manipulation

/analyze/audio → Voice spoofing & cloning

/analyze/fusion → Final Trust Score (0–100)

Each module uses models loaded from models/ and logic defined in services/.

🟧 3. Unified Fusion Layer

All three detection outputs are merged to generate:

🔥 Content Trust Score (0–100)
Score	Meaning
80–100	Safe, verified
50–79	Caution — partial risk detected
20–49	Suspicious — limit reach
<20	Dangerous — block / review
📦 Installation & Setup
1. Clone the repository
git clone https://github.com/yourname/Unified-AI-Content-Shield.git
cd Unified-AI-Content-Shield

2. Setup Backend
Install dependencies
cd backend
pip install -r requirements.txt

Run the backend
uvicorn main:app --reload


API will be live at:
👉 http://localhost:8000

3. Load Chrome Extension

Open Chrome → Extensions → Developer Mode → Load Unpacked

Select the chrome-extension/ folder

Done!

📊 Roadmap
MVP

Text toxicity detection

Fake news classifier

Basic overlay badge

Chrome extension → backend API

V1.0

Deepfake image/video detection

Voice clone detection

Fusion scoring

Full dashboard for trust analytics

V2.0

Cross-browser support

Plugin support for YouTube, Instagram, X

Federated learning for privacy-first detection

Real-time streaming inference

✨ Future Enhancements

LLM-based context-aware misinformation checking

Memory-based pattern tracking of repeat offenders

Crowd-sourced moderation insights

Edge inference for low-latency deepfake detection

📜 License

MIT License — Free to use & modify.