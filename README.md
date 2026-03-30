#  JusticeAI - AI-Powered Legal Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-61DAFB) ![Flask](https://img.shields.io/badge/Flask-2.0-green) ![DistilBERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow) ![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple) ![Groq](https://img.shields.io/badge/Groq-LLM-orange) ![License](https://img.shields.io/badge/License-MIT-red)

> An intelligent legal assistance platform powered by Deep Learning,
> RAG Pipeline, and Large Language Models.

---

##  Overview

JusticeAI is a full-stack AI-powered legal assistant that provides
specialized legal guidance across 3 domains. It uses a fine-tuned
DistilBERT model for intent classification and a RAG pipeline
combining Pinecone vector search with Groq LLM for accurate
legal responses.

**Repository:** [github.com/NACHAMMAI-SN/JusticeAI](https://github.com/NACHAMMAI-SN/JusticeAI)

---

##  Deep Learning Architecture

```mermaid
graph TD
    A[ User Query] --> B[ Intent Classifier\nDistilBERT Fine-tuned\nLEDGAR Dataset]
    B --> C{Intent Type}
    C -->|Personal & Family| D[ RAG Pipeline]
    C -->|Business & Criminal| D
    C -->|Consultation| D
    D --> E[ Sentence Transformer\nparaphrase-multilingual-MiniLM-L12-v2\n384-dim embeddings]
    E --> F[ Pinecone Vector DB\nCosine Similarity Search]
    F --> G[ Retrieved Legal Context]
    G --> H[ Groq LLM\nllama-3.3-70b-versatile]
    A --> H
    H --> K[ Final Response to User]
```

---

## 🧠 Deep Learning Model Comparison

### BERT-based Models (Transformer Architecture)

| Model | Architecture | Val Accuracy | Val Loss |
|-------|--------------|--------------|----------|
| Legal-BERT + MLP | Legal-BERT → CLS token → 768→512→256→3 | 100.00% | 0.02 |
| DistilBERT + MLP | DistilBERT → CLS token → 768→256→3 | 100.00% | 0.03 |
| BERT-base + MLP | BERT → Mean Pooling → 768→512→128→7 | 100.00% | 0.02 |

### RNN-based Models (Sequential Architecture)

| Model | Optimizer | Val Accuracy | Val Loss | Epochs |
|-------|-----------|--------------|----------|--------|
| BidirectionalLSTM | Adam | 100.00% | 0.0002 | 20 |
| BidirectionalLSTM | AdamW | 100.00% | 0.0002 | 20 |
| LSTM | Adam | 98.89% | 0.0472 | 14 |
| LSTM | AdamW | 97.78% | 0.0761 | 15 |
| StackedLSTM | Adam | 97.78% | 0.1326 | 12 |
| StackedLSTM | AdamW | 97.78% | 0.1281 | 8 |
| BidirectionalLSTM | SGD | 96.67% | 0.0726 | 20 |
| SimpleRNN | AdamW | 91.11% | 0.3140 | 10 |
| SimpleRNN | Adam | 88.89% | 0.3551 | 10 |
| SimpleRNN | SGD | 83.33% | 0.5571 | 14 |
| LSTM | SGD | 78.89% | 0.5340 | 20 |
| StackedLSTM | SGD | 31.11% | 1.1002 | 6 |

### 🏆 Best Model

**BidirectionalLSTM with Adam optimizer** achieved **100% validation accuracy** matching the performance of BERT-based Transformer models.

### Key Findings

- BiLSTM reads text both forward AND backward, capturing full context
- Adam and AdamW optimizers significantly outperform SGD for all models
- SGD fails completely on StackedLSTM (31.11%) due to vanishing gradients
- LSTM family consistently outperforms SimpleRNN on legal text
- Transformer models (BERT) converge faster but BiLSTM matches final accuracy

---

## 📊 Training Results & Graphs

### Validation Accuracy per Epoch — All RNN Variants

![RNN Accuracy per Epoch](Server/models/rnn_accuracy_per_epoch.png)

### Validation Loss per Epoch — All RNN Variants

![RNN Loss per Epoch](Server/models/rnn_loss_per_epoch.png)

### Best Accuracy by Model and Optimizer

![RNN Best Accuracy Comparison](Server/models/rnn_best_accuracy_comparison.png)

### Final Model Comparison — RNN vs BERT

![Final Model Comparison](Server/models/final_model_comparison.png)

---

## 🔬 Running the RNN Comparison

To reproduce all RNN experiments:

```bash
cd Server
python rnn_comparison.py
```

This will:

- Load LEDGAR dataset from HuggingFace automatically
- Train 12 models (4 architectures × 3 optimizers)
- Save all results to Server/models/ folder
- Generate all comparison graphs
- Print best model at the end

Requirements:

```bash
pip install datasets torch matplotlib
```

Expected runtime: ~10 minutes on CPU

---

## 🏗️ Complete System Architecture

End-to-end flow from user query to response (production stack):

```mermaid
flowchart TD
    A[User Query - any Indian language] --> B[Language Detection - franc]
    B --> C[Intent Classification]
    C --> C1[Legal-BERT + MLP Head]
    C1 --> C2[fine-tuned on LEDGAR]
    C2 --> C3[3 classes: Personal & Family, Business & Criminal, Consultation]
    C3 --> D[RAG Pipeline]
    D --> D1[Sentence Transformer - multilingual MiniLM]
    D1 --> D2[384-dim embeddings]
    D2 --> D3[Pinecone Vector Search]
    D3 --> D4[top-3 relevant laws]
    D4 --> E[Response Generation]
    E --> E1[Groq LLaMA 3.3 70B]
    E1 --> E2[Retrieved Legal Context + Conversation History]
    E2 --> F[Final Response - user's language]
```

<details>
<summary>ASCII overview (same flow)</summary>

```
User Query (any Indian language)
        ↓
   Language Detection (franc)
        ↓
   Intent Classification
   ┌─────────────────────────────┐
   │  Legal-BERT + MLP Head      │
   │  (fine-tuned on LEDGAR)     │
   │  3 classes:                 │
   │  • Personal & Family        │
   │  • Business & Criminal      │
   │  • Consultation             │
   └─────────────────────────────┘
        ↓
   RAG Pipeline
   ┌─────────────────────────────┐
   │  Sentence Transformer       │
   │  (multilingual MiniLM)      │
   │  → 384-dim embeddings       │
   │        ↓                    │
   │  Pinecone Vector Search     │
   │  (top-3 relevant laws)      │
   └─────────────────────────────┘
        ↓
   Response Generation
   ┌─────────────────────────────┐
   │  Groq LLaMA 3.3 70B        │
   │  + Retrieved Legal Context  │
   │  + Conversation History     │
   └─────────────────────────────┘
        ↓
   Final Response (user's language)
```

</details>

---

##  Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Deep Learning | DistilBERT (distilbert-base-uncased) | Intent Classification fine-tuned on LEDGAR |
| Embeddings | Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2) | 384-dim text embeddings |
| Vector DB | Pinecone | Semantic search over legal documents |
| LLM | Groq (llama-3.3-70b-versatile) | Legal response generation |
| Frontend | React 18 + Vite | User interface |
| Backend | Flask (Python) | REST API server |
| Auth | Supabase | User authentication & chat history |
| Training | HuggingFace Transformers + PyTorch | DL model training |

---

##  Features

-  **3 Specialized Legal Chatbots** — Personal & Family, Business & Criminal, Consultation
-  **DL Intent Classification** — Fine-tuned DistilBERT on real LEDGAR legal dataset
-  **RAG Pipeline** — Pinecone vector search + Groq LLM for accurate responses
-  **Voice Input** — Speech to text support
-  **Text to Speech** — Audio response playback
-  **Multi-language Support** — Hindi, Tamil, and more
-  **Secure Authentication** — Supabase with email confirmation
-  **Persistent Chat History** — Per user, per service

---

## Screenshots

### Landing Page
![Landing Page](SS/SS_1.png)
> AI-powered legal assistant homepage with hero section and
> quick access to all legal services

###  Why JusticeAI
![Features Section](SS/SS_2.png)
> Specialized chatbots, Advanced AI integration, Multi-lingual
> support, Voice-to-Text, 24/7 availability and Security features

###  Legal Chatbot in Action
![Chatbot Interface](SS/SS_3.png)
> Real-time legal assistance powered by DistilBERT intent
> classification and RAG pipeline with Groq LLM

---

## Project Structure

```
JusticeAI/
├── App/
│   └── project/
│       ├── src/
│       │   ├── LandingPage.jsx
│       │   ├── ServiceChatPage.jsx
│       │   ├── ServicesPage.jsx
│       │   ├── ContactUs.jsx
│       │   ├── AboutUs.jsx
│       │   ├── Login.jsx
│       │   ├── SignUp.jsx
│       │   └── i18n.js
│       ├── vite.config.js
│       ├── package.json
│       └── .env
├── Server/
│   ├── app.py                        # Main Flask API
│   ├── train.py                      # DistilBERT training script
│   ├── dl_intent_classifier.py       # Intent classifier model definition
│   ├── dl_document_classifier.py     # Document classifier model definition
│   ├── dl_training_pipeline.py       # Training pipeline utilities
│   ├── populate_pinecone.py          # Pinecone data upload script
│   ├── models/
│   │   └── intent_classifier.pt      # Trained DistilBERT weights
│   ├── Dockerfile
│   ├── Procfile
│   └── requirements.txt
├── README.md
└── .gitignore
```

---

## DL Model Details

### Intent Classifier (DistilBERT)

- **Base Model:** distilbert-base-uncased
- **Dataset:** LEDGAR (real legal contracts dataset from HuggingFace)
- **Training:** 5 epochs
- **Validation Accuracy:** 100%
- **F1 Score:** 1.00
- **Classes:** personal-and-family | business-and-criminal | consultation
- **Framework:** PyTorch + HuggingFace Transformers

### RAG Pipeline

- **Embeddings:** paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
- **Vector DB:** Pinecone (cosine similarity metric)
- **LLM:** Groq llama-3.3-70b-versatile
- **Flow:** Query → Embed → Pinecone Search → Retrieve Context → Groq LLM → Response

---

## Results

| Metric | Value |
|--------|-------|
| Intent Classifier Accuracy | 100% |
| Intent Classifier F1 Score | 1.00 |
| Training Dataset | LEDGAR (real legal contracts) |
| Embedding Dimensions | 384 |
| Training Epochs | 5 |
| Number of Intent Classes | 3 |

### Training Curves

![Training Curves](Server/models/training_curves.png)

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Pinecone account (free tier)
- Groq API key (free)
- Supabase account (free)

### 1. Clone the repository

```bash
git clone https://github.com/NACHAMMAI-SN/JusticeAI.git
cd JusticeAI
```

### 2. Backend Setup

```bash
cd Server
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Backend Environment Variables

Create `Server/.env`:

```env
PINECONE_API=your_pinecone_api_key
GROQ_API=your_groq_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

Optional: `PINECONE_ENV` (e.g. `us-east-1`), `CREATE_PINECONE_INDEX=true` to build the index from Supabase PDFs on startup.

### 4. Train the model and populate Pinecone

```bash
python train.py
python populate_pinecone.py
```

### 5. Start Backend

```bash
python app.py
```

Backend runs on http://127.0.0.1:5000

### 6. Frontend Setup

```bash
cd App/project
npm install
```

### 7. Frontend Environment Variables

Create `App/project/.env`:

```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://127.0.0.1:5000
```

For local dev with the Vite proxy to avoid CORS, you can set `VITE_API_URL` to `/api` and ensure `vite.config.js` proxies `/api` to the Flask server.

### 8. Start Frontend

```bash
npm run dev
```

Frontend runs on http://localhost:5173

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/<service>/chat` | POST | Send message to legal chatbot |
| `/<service>/history` | GET | Get chat history for user |
| `/dl-info` | GET | Get DL architecture information |
| `/train-models` | POST | Trigger model training |
| `/submit-form` | POST | Submit contact form |

---

## Environment Variables

| Variable | Location | Description |
|----------|----------|-------------|
| `PINECONE_API` | Server/.env | Pinecone API key |
| `GROQ_API` | Server/.env | Groq API key |
| `SUPABASE_URL` | Server/.env | Supabase project URL (backend) |
| `SUPABASE_ANON_KEY` | Server/.env | Supabase anon key (backend) |
| `VITE_SUPABASE_URL` | App/project/.env | Supabase project URL (frontend) |
| `VITE_SUPABASE_ANON_KEY` | App/project/.env | Supabase anon key (frontend) |
| `VITE_API_URL` | App/project/.env | Backend API URL |

---

## Credits & Acknowledgements

- Original [LawPal](https://github.com/AaryanGole26/LawPal) repo by [@AaryanGole26](https://github.com/AaryanGole26)
- [HuggingFace](https://huggingface.co) for DistilBERT model
- [LEDGAR Dataset](https://huggingface.co/datasets/coastalcph/lex_glue) for intent classifier training
- [Groq](https://groq.com) for LLM API
- [Pinecone](https://pinecone.io) for vector database
- [Supabase](https://supabase.com) for authentication

---

##  License

This project is licensed under the MIT License.

---


