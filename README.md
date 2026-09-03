# ⚡ GridMind GR

**AI-Powered Greek Energy Intelligence Platform**

GridMind GR is an end-to-end Data Science and AI platform for analyzing, forecasting, and exploring the Greek electricity system using public energy data from **ADMIE/IPTO**.

The project combines **data engineering, machine learning, backend development, interactive visualization, and Retrieval-Augmented Generation (RAG)** into a production-oriented application.

> 🚧 **Project Status:** Under active development.

---

## 🎯 Project Goals

GridMind GR aims to transform publicly available Greek electricity data into an intelligent analytics platform capable of:

- Collecting and processing real energy-system data
- Exploring electricity demand and renewable energy production
- Forecasting future electricity demand
- Comparing ML forecasts with official IPTO forecasts
- Detecting unusual patterns and anomalies
- Visualizing the Greek electricity system through an interactive dashboard
- Answering energy-related questions through an AI assistant
- Retrieving information from official energy reports using RAG

---

## 🏗️ Planned Architecture

```text
                    ADMIE / IPTO
                  APIs & Reports
                        │
                        ▼
               ┌─────────────────┐
               │ Data Ingestion  │
               │    Pipeline     │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Data Validation │
               │   & Cleaning    │
               └────────┬────────┘
                        │
                        ▼
                 ┌────────────┐
                 │ PostgreSQL │
                 └─────┬──────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Analytics    ML Forecasting   FastAPI
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                    Streamlit             AI Assistant
                                             │
                                             ▼
                                      RAG + Vector DB
                                             │
                                             ▼
                                      Official Reports
```

---

## 📊 Data Sources

The primary data source is the **ADMIE/IPTO Operation & Market Files API**.

Planned datasets include:

| Dataset | Purpose |
|---|---|
| System Load | Electricity demand analysis and forecasting |
| RES Production | Renewable energy analysis and forecasting |
| Unit Production | Greek electricity generation mix |
| Interconnection Flows | Imports and exports analysis |
| IPTO Load Forecast | Benchmark against our ML forecast |
| IPTO RES Forecast | Renewable forecast benchmarking |
| Energy Reports | Knowledge base for RAG |

---

## 🤖 Machine Learning

The forecasting pipeline will evaluate multiple approaches, starting with simple baselines and progressing to machine-learning models.

Planned models include:

- Naive forecasting baseline
- Linear Regression
- Random Forest
- XGBoost
- Isolation Forest for anomaly detection

Forecast performance will be evaluated using:

- **MAE**
- **RMSE**
- **MAPE**

A key goal of the project is to compare the **GridMind GR electricity-demand forecast against the official IPTO forecast**.

---

## 🧠 AI Energy Analyst

GridMind GR will include an AI assistant capable of answering two types of questions.

### Structured Data Questions

Examples:

> What was the maximum electricity demand last month?

> How much renewable energy was produced yesterday?

These questions will be answered using structured energy data.

### Energy Knowledge Questions

Examples:

> What does System Load mean?

> How does ADMIE forecast electricity demand?

These questions will use **Retrieval-Augmented Generation (RAG)** over official energy reports and documentation.

The RAG pipeline will use:

```text
Official Reports
      ↓
Document Processing
      ↓
Embeddings
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
LLM
      ↓
Answer + Sources
```

---

## 📈 Interactive Dashboard

The Streamlit dashboard is planned to include:

- Electricity demand time series
- Renewable energy production
- Energy generation mix
- Imports and exports
- Actual vs ML forecast
- ML forecast vs official IPTO forecast
- Detected anomalies
- AI Energy Analyst chat interface

---

## 🛠️ Tech Stack

### Data & Machine Learning

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost

### Backend

- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL

### AI / RAG

- LLM integration
- Embeddings
- Qdrant
- Retrieval-Augmented Generation

### Frontend

- Streamlit
- Plotly

### DevOps & Testing

- Docker
- Docker Compose
- pytest
- GitHub Actions

---

## 📁 Project Structure

```text
gridmind-gr/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   └── gridmind/
│       ├── data/
│       └── __init__.py
│
├── tests/
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

The project structure will evolve as the API, database, ML, dashboard, and RAG components are implemented.

---

## 🚀 Development Roadmap

- [x] Initialize project structure
- [x] Configure Python development environment
- [ ] Implement ADMIE/IPTO API client
- [ ] Build automated data ingestion pipeline
- [ ] Add data cleaning and validation
- [ ] Add PostgreSQL persistence
- [ ] Build FastAPI backend
- [ ] Perform exploratory data analysis
- [ ] Build electricity-demand forecasting pipeline
- [ ] Benchmark ML forecast against IPTO forecast
- [ ] Add anomaly detection
- [ ] Build Streamlit dashboard
- [ ] Implement RAG pipeline
- [ ] Build AI Energy Analyst
- [ ] Containerize services with Docker
- [ ] Add automated tests
- [ ] Configure GitHub Actions CI
- [ ] Add documentation and demo

---

## 👥 Development

GridMind GR is being developed collaboratively by a two-person team.

Development follows a feature-branch workflow:

```text
main
 ├── feature/admie-ingestion
 ├── feature/database
 ├── feature/fastapi
 ├── feature/forecasting
 ├── feature/streamlit
 ├── feature/rag
 └── feature/docker
```

Changes are developed in feature branches and merged into `main` through pull requests and code review.

---

## 📌 Current Status

The project is currently in its initial development phase.

The first milestone is:

**ADMIE API → Raw Energy Data → Pandas → Clean & Validated Dataset**

From there, the platform will progressively introduce persistence, APIs, machine learning, visualization, and AI capabilities.

---

## 📄 License

A license will be added as the project matures.
