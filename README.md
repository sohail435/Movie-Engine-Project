# Advanced Movie Recommender & Sentiment Analytics Engine

An end-to-end Machine Learning web application combining personalized filtering mechanics with deep learning NLP feedback loop structures. The system utilizes an isolated microservices architecture to serve concurrent recommendation queries and parse real-time text reviews.

## 🚀 System Architecture & Flow

### Component Blueprint
The engine decouples storage, mathematical processing, and display layers to achieve high scannability and modular updates:

- **Frontend Interface:** Built with Streamlit to parse inputs dynamically.
- **Orchestration Layer (API Gateway):** Powered by FastAPI to expose operational endpoints.
- **Personalization Engine:** Uses NumPy vector dot-products to execute fast Cosine Similarity matrices.
- **NLP Engine:** Utilizes Hugging Face pipelines running an optimized DistilBERT model.

### Data Flow Execution
1. **Flow A (Online Query):** User requests recommendation -> FastAPI parses Relational Metadata -> Matches vectors in real time -> Computes similarity scores -> Renders sorted match percentages.
2. **Flow B (Real-time Feedback):** User inputs raw text review -> Transformer pipeline computes explicit sentiment probability weights -> Outputs profile vector shift factors.

## 🛠️ Tech Stack & Dependencies

| Component | Technology | Role |
| :--- | :--- | :--- |
| Core Language | Python 3.14 | Runtime Environment |
| Web Gateway | FastAPI & Uvicorn | Async Routing API Management |
| Machine Learning | Hugging Face Transformers | Deep Learning NLP Sentiment Models |
| Vector Operations | NumPy & Pandas | Matrix Analytics & Mathematical Core |
| Interface Framework | Streamlit | Responsive Web UI |

## 📦 Local Setup Instructions

Ensure your machine has Python installed, then execute these isolation steps:

1. **Clone the project & step inside:**
   ```bash
   cd Movie-Engine-Project