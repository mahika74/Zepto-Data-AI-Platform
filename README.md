# Zepto Data & AI Platform

An end-to-end **Data & AI platform** developed as part of the **IIT Patna AI/ML Certificate Program – Capstone Project**.

The project combines **data engineering, data analytics, machine learning, retrieval-augmented generation (RAG), API development, containerization, and automated testing** into three integrated modules.

---

## 📌 Project Overview

| Module | Focus | Technologies |
|---|---|---|
| **Module 1 – Data Pipeline** | Web scraping, data cleaning, SQLite, SQL and Pandas analytics | Python, BeautifulSoup, Pandas, SQLite, SQL |
| **Module 2 – Analytics & ML** | EDA, preprocessing, classification, regression and model tuning | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib |
| **Module 3 – Support Assistant** | RAG-based Zepto policy assistant | Sentence Transformers, ChromaDB, LangGraph, FastAPI, Docker |

---

## 📂 Project Structure

```text
Zepto-Data-AI-Platform/
│
├── data_pipeline/
│   ├── 01_data_pipeline.ipynb
│   ├── README.md
│   ├── books.db
│   ├── clean_books.csv
│   └── query_results.txt
│
├── analytics/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── titanic.csv
│   ├── best_pipeline.joblib
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   │
│   ├── embeddings.py
│   ├── graph.py
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── .github/
│   └── workflows/
│       └── docker-test.yml
│
├── .dockerignore
└── README.md
```

---

# 🧩 Module 1 – Data Pipeline

## Objective

Build a complete data pipeline that collects book information from **Books to Scrape**, cleans the data, stores it in a normalized SQLite database, and performs analytical queries using SQL and Pandas.

## Workflow

```text
Books to Scrape
       ↓
Web Scraping
       ↓
Data Cleaning
       ↓
Currency Conversion
       ↓
SQLite Database
       ↓
SQL Queries
       ↓
Pandas Analysis
       ↓
Query Results
```

## Key Features

- Scraped **100 books** from the first 5 product listing pages.
- Extracted:
  - Title
  - Price
  - Star rating
  - Availability
  - Category
  - Product link
- Converted GBP prices to INR using:

```text
1 GBP = 105.50 INR
```

- Cleaned price and rating values.
- Converted ratings from text (`One`–`Five`) to integers (`1`–`5`).
- Converted availability into a Boolean `in_stock` field.
- Handled numeric parsing failures using median imputation where applicable.
- Created a normalized SQLite database containing:
  - `categories`
  - `books`
- Implemented primary-key and foreign-key relationships.
- Executed SQL queries covering:
  - `SELECT` / `WHERE`
  - `ORDER BY` / `LIMIT`
  - `DISTINCT`
  - `BETWEEN`
  - `IN`
  - `JOIN`
- Saved SQL query strings and outputs.
- Used `pandas.read_sql()` for database analysis.
- Reproduced SQL JOIN results using `pandas.merge()`.

## Module 1 Artifacts

```text
data_pipeline/
├── 01_data_pipeline.ipynb
├── books.db
├── clean_books.csv
├── query_results.txt
└── README.md
```

---

# 📊 Module 2 – Analytics & Machine Learning

## Objective

Build a complete Titanic data analysis and machine learning workflow covering exploratory data analysis, preprocessing, classification, regression, model evaluation, class imbalance handling, hyperparameter tuning, and model persistence.

## Workflow

```text
Titanic Dataset
       ↓
Data Inspection
       ↓
Missing Value Analysis
       ↓
Exploratory Data Analysis
       ↓
Feature Analysis
       ↓
Visualization
       ↓
Train/Test Split
       ↓
Training-only Preprocessing
       ↓
Classification Models
       ↓
Evaluation
       ↓
Class Imbalance Analysis
       ↓
Hyperparameter Tuning
       ↓
Fare Regression
       ↓
Model Persistence
```

## Exploratory Data Analysis

The analysis includes:

- Dataset shape and structure
- `info()` and `describe()`
- Missing-value percentages
- Missing-value handling based on thresholds
- Age and Fare distributions
- Histograms
- Box plots
- IQR-based outlier analysis
- Mean, median, and mode comparison
- Skewness analysis
- Boolean filtering
- Survival analysis by:
  - Sex
  - Passenger class
  - Sex + passenger class
- Correlation analysis
- Correlation heatmap
- Multiple survival-focused visualizations

## Standardization

Exploratory standardization was performed on:

- Age
- Fare

Before-and-after mean and standard deviation were examined to verify the effect of standardization.

The exploratory standardization was kept separate from the modeling pipeline.

## Machine Learning Models

The classification workflow includes:

### Logistic Regression

A linear classification model used as one of the baseline classifiers.

### Decision Tree

A tree-based model used to capture nonlinear relationships.

### Random Forest

An ensemble model consisting of multiple decision trees.

All classifiers use the same train/test split for comparison.

## Preprocessing

Training-only preprocessing includes:

- Missing-value handling
- Categorical encoding
- Numerical scaling
- `Pipeline`
- `ColumnTransformer`

This keeps preprocessing fitted only on the training data and reduces the risk of data leakage.

## Evaluation Metrics

Classification models are evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- ROC-AUC

The classification results are compared side-by-side.

## Class Imbalance

Class imbalance is investigated using:

1. Baseline classifier
2. Balanced classifier
3. SMOTE applied only to training data

Precision, recall, and F1-score are compared across the approaches.

## Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest model over:

- `n_estimators`
- `max_depth`
- `max_features`

The tuned Random Forest also uses Out-of-Bag evaluation.

## Regression

A separate regression task predicts passenger fare.

The evaluation includes:

- MAE
- RMSE
- R²
- Adjusted R²
- Residual plot
- Heteroscedasticity analysis

## Model Persistence

The complete fitted preprocessing and modeling pipeline is saved using `joblib`.

```text
best_pipeline.joblib
```

The saved pipeline is then reloaded and used to generate predictions from raw input data.

## Module 2 Artifacts

```text
analytics/
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic.csv
├── best_pipeline.joblib
└── README.md
```

---

# 🤖 Module 3 – Zepto Support Assistant

## Objective

Build a retrieval-based AI support assistant that answers Zepto-related questions using a local policy knowledge base.

The system combines:

- Local embeddings
- ChromaDB
- Structured prompting
- LangGraph
- Pydantic
- FastAPI
- Docker

## Architecture

```text
                    User Question
                          │
                          ▼
                    FastAPI /ask
                          │
                          ▼
                  Intent Classification
                          │
                 ┌────────┴────────┐
                 │                 │
          Policy Question     General Question
                 │                 │
                 ▼                 ▼
           ChromaDB Search    Direct Answer
                 │
                 ▼
        Retrieved Policy Context
                 │
                 ▼
          Structured Prompt
                 │
                 ▼
             AI Response
                 │
                 ▼
          Pydantic Validation
                 │
                 ▼
                JSON
```

---

## Knowledge Base

The Support Assistant uses exactly **8 local policy documents**.

```text
support_assistant/
└── docs/
    ├── doc_01.txt
    ├── doc_02.txt
    ├── doc_03.txt
    ├── doc_04.txt
    ├── doc_05.txt
    ├── doc_06.txt
    ├── doc_07.txt
    └── doc_08.txt
```

## Embeddings

The project uses the local Hugging Face model:

```text
all-MiniLM-L6-v2
```

The model converts policy documents and user queries into vector embeddings.

The embeddings are indexed and searched using **ChromaDB**.

No paid embedding API is required.

---

## 🔎 Retrieval-Augmented Generation

For policy-related questions, the workflow is:

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB Similarity Search
      ↓
Relevant Policy Documents
      ↓
Context Construction
      ↓
Structured Prompt
      ↓
Answer
```

The response includes the relevant document sources.

---

## 🧠 Structured Prompting

The Support Assistant uses a structured prompt containing:

- Role
- Context
- Task
- Output format
- Length constraint
- Negative constraint
- Few-shot example

This helps maintain consistent responses and keeps the assistant within the supported Zepto policy domain.

---

## 🔄 LangGraph Workflow

The application contains three main workflow nodes:

### 1. `classify_intent`

Determines whether the user question is related to the supported Zepto policy domain.

### 2. `retrieve_and_answer`

For policy-related questions:

- Performs vector search
- Retrieves relevant documents
- Builds the structured prompt
- Generates the response

### 3. `direct_answer`

Handles general questions outside the supported policy domain.

Conditional routing determines which node executes.

---

## 🧪 Deterministic Mock LLM

The project includes a deterministic mock LLM mode:

```text
MOCK_LLM
```

This allows the application to run and be tested without requiring a paid external language model API.

It supports:

- Local testing
- Docker testing
- CI/CD testing
- Reproducible demonstrations

---

## 📦 Structured Output

Responses are validated using Pydantic.

The response contains:

```text
answer
sources
confidence
```

Example:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01.txt"],
  "confidence": 1.0
}
```

---

# 🚀 FastAPI

The application exposes:

```text
POST /ask
```

### Example Request

```json
{
  "question": "What is the delivery fee?"
}
```

### Example Response

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01.txt"],
  "confidence": 1.0
}
```

General non-policy questions are handled separately.

Example:

```json
{
  "question": "What is the capital of India?"
}
```

The assistant responds that it can only answer questions within the supported Zepto policy domain.

---

# 🐳 Docker

The Support Assistant includes a Dockerfile for containerized execution.

The Docker image:

1. Installs dependencies
2. Downloads the embedding model
3. Copies the application
4. Builds the ChromaDB index
5. Starts the FastAPI server

The application runs on:

```text
Port: 7860
```

---

# ⚙️ GitHub Actions

Automated Docker testing is implemented using GitHub Actions.

The workflow:

```text
Git Push
   ↓
GitHub Actions
   ↓
Build Docker Image
   ↓
Start Container
   ↓
Wait for API
   ↓
Test /ask Endpoint
   ↓
Test Policy Query
   ↓
Test General Query
   ↓
Pass / Fail
```

This provides automated verification of the Dockerized Support Assistant.

---

# 🛠️ Technologies Used

## Data Engineering

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL

## Data Science & Machine Learning

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib

## Generative AI / RAG

- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- Pydantic

## Backend

- FastAPI
- Uvicorn

## Deployment & DevOps

- Docker
- Git
- GitHub
- GitHub Actions

---

# ▶️ How to Run

## Module 1 – Data Pipeline

Open:

```text
data_pipeline/01_data_pipeline.ipynb
```

Run the notebook to:

1. Scrape the Books to Scrape website
2. Clean the collected data
3. Create the SQLite database
4. Execute SQL queries
5. Perform Pandas analysis
6. Generate query outputs

---

## Module 2 – Analytics & ML

Open the notebooks in order:

```text
analytics/01_eda.ipynb
analytics/02_modeling.ipynb
```

The workflow performs:

```text
EDA
→ Cleaning
→ Visualization
→ Preprocessing
→ Classification
→ Evaluation
→ Imbalance Analysis
→ Hyperparameter Tuning
→ Regression
→ Model Persistence
```

The saved model is:

```text
analytics/best_pipeline.joblib
```

---

## Module 3 – Support Assistant

Navigate to the project directory:

```bash
cd support_assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the local vector index:

```bash
python -m support_assistant.embeddings
```

Start the API:

```bash
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

API:

```text
http://localhost:7860
```

Swagger documentation:

```text
http://localhost:7860/docs
```

---

# 🐳 Run with Docker

From the repository root:

```bash
docker build -t zepto-support-assistant ./support_assistant
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

Open:

```text
http://localhost:7860/docs
```

---

# 🏗️ Design Decisions

### Local-first approach

Local models and tools are used wherever possible to avoid dependence on paid APIs.

### Modular architecture

The project is divided into three independent modules covering:

- Data Engineering
- Data Science & Machine Learning
- Generative AI

### Reproducibility

Important datasets and model artifacts are stored with the project where required:

```text
clean_books.csv
books.db
titanic.csv
best_pipeline.joblib
```

### Training-only preprocessing

Machine learning preprocessing is fitted only on training data to reduce the risk of data leakage.

### Structured AI workflow

The Support Assistant uses:

```text
Intent Classification
        ↓
Conditional Routing
        ↓
Retrieval
        ↓
Prompt Construction
        ↓
Structured Response
```

### Deterministic testing

`MOCK_LLM` enables testing without requiring an external paid LLM.

### Containerization

Docker provides a reproducible runtime environment, while GitHub Actions automates Docker testing.

---

# 📈 End-to-End Platform

```text
                  Zepto Data & AI Platform
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   DATA PIPELINE      ANALYTICS & ML    AI SUPPORT
          │                │                │
          ▼                ▼                ▼
     Web Data           Titanic        Policy Docs
          │                │                │
          ▼                ▼                ▼
       Cleaning          EDA + ML       Embeddings
          │                │                │
          ▼                ▼                ▼
       SQLite         Model Tuning     ChromaDB
          │                │                │
          ▼                ▼                ▼
     SQL/Pandas       Joblib Model     LangGraph
                                           │
                                           ▼
                                        FastAPI
                                           │
                                           ▼
                                         Docker
                                           │
                                           ▼
                                    GitHub Actions
```

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Web scraping
- Data cleaning
- Data transformation
- Relational database design
- SQL querying
- Pandas analytics
- Exploratory Data Analysis
- Data visualization
- Feature preprocessing
- Classification
- Regression
- Model evaluation
- Class imbalance handling
- Hyperparameter optimization
- Model persistence
- Vector embeddings
- Semantic search
- Retrieval-Augmented Generation
- LangGraph workflows
- Structured AI outputs
- REST API development
- Docker containerization
- CI/CD testing

---

# 👩‍💻 Project Information

**Project:** Zepto Data & AI Platform

**Program:** IIT Patna AI/ML Certificate Program

**Type:** Capstone Project

---

## 📌 Repository Summary

```text
Zepto-Data-AI-Platform/
│
├── data_pipeline/       → Web Scraping + SQLite + SQL + Pandas
├── analytics/           → EDA + ML + Evaluation + Regression
├── support_assistant/   → RAG + ChromaDB + LangGraph + FastAPI
├── .github/workflows/   → Automated Docker Testing
└── README.md            → Project Documentation
```

---

## Conclusion

The **Zepto Data & AI Platform** brings together data engineering, machine learning, and generative AI into one modular capstone project.

The three modules demonstrate complete workflows from **data acquisition and database analytics**, through **machine learning and model persistence**, to a **retrieval-based AI support assistant with API deployment, Docker, and automated testing**.
