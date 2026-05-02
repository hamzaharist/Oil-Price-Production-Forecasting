# [Project Name]
### [One sharp line — what it does and why it matters]

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Stack](https://img.shields.io/badge/stack-React%20%7C%20Python%20%7C%20SQL-blue)
![Type](https://img.shields.io/badge/type-[Full--Stack%20%7C%20Analytics%20%7C%20ML]-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The Problem

> *One paragraph. Write this like a consultant framing the situation.*
> 
> [Who was suffering from what? What gap existed? What was the cost of not solving it?
> Keep it business-relevant. Non-technical readers should understand why this mattered.]

**Example:** Malaysian cities face intensifying Urban Heat Island effects, but urban planners lacked accessible tools to identify thermal hotspots, track changes over time, and prioritize interventions at scale. Manual satellite analysis was expensive, slow, and required GIS expertise most municipalities don't have.

---

## The Solution

> *What you built. Frame it around the outcome, not the technology.*

[Short paragraph describing what the platform/tool/model does. Lead with the user benefit, follow with the technical approach.]

**What it enables:**
- [Business outcome 1 — keep it action-oriented]
- [Business outcome 2]
- [Business outcome 3]

---

## Results

> *This section is the first thing a recruiter/hiring manager reads. Make the numbers land.*

| Metric | Value |
|---|---|
| [Model Performance] | [e.g., R² = 0.87, RMSE = 1.2°C] |
| [Scale] | [e.g., 80,000 sample dataset across 5 cities] |
| [Impact] | [e.g., Identified 3 critical hotzone clusters for planner review] |
| [Efficiency] | [e.g., Reduced manual analysis time from days to minutes] |

---

## Architecture

```
[Project Root]
├── frontend/          # React 18 — interactive dashboard + map
├── backend/           # FastAPI — REST API, ML inference endpoint
├── ml/                # Model training, evaluation, feature engineering
│   ├── data/          # Raw + processed datasets
│   └── models/        # Saved model artifacts (.pkl)
├── db/                # Supabase schema, migrations
└── docs/              # Architecture diagrams, reports
```

> *Replace with your actual structure. Keep it clean and descriptive.*

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | React 18, Tailwind CSS, Leaflet.js |
| Backend | FastAPI, Python 3.11 |
| ML | Scikit-learn, Pandas, GeoPandas, NumPy |
| Database | Supabase (PostgreSQL + PostGIS) |
| Visualization | Recharts, Plotly |
| Deployment | Vercel (frontend), Railway (backend) |

---

## How It Works

**1. Data Pipeline**
[Describe where data comes from, how it gets cleaned, what gets stored. 2-3 sentences max.]

**2. ML Model**
[What model, trained on what, evaluated how. Include the key metric.]

**3. Dashboard**
[What a user sees and does. What decisions does it support?]

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/[yourusername]/[repo-name].git

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

**Environment variables:**
```env
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
API_BASE_URL=http://localhost:8000
```

---

## Key Decisions & Tradeoffs

> *This is the section that separates a portfolio project from a tutorial copy-paste.
> Show your thinking.*

- **Why Random Forest over XGBoost:** RF gave comparable accuracy (R² 0.87 vs 0.88) with significantly faster inference, better suited for a real-time API endpoint.
- **Why Supabase:** PostGIS extension for spatial queries + built-in auth + free tier for a student project. Would migrate to AWS RDS at production scale.
- **What I would do differently:** [Be honest. Engineers respect self-awareness.]

---

## Screenshots

> *Add 2-3 screenshots max. Caption each one. Quality over quantity.*

| Dashboard Overview | Heatmap Layer | Model Output |
|---|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Heatmap](docs/screenshots/heatmap.png) | ![Model](docs/screenshots/model.png) |

---

## Background

Built as my **Final Year Dissertation** at Universiti Teknologi PETRONAS (UTP), graduating class of 2026 with a BSc in Big Data Analytics.

Prior to this, I spent 8 months as a **Business Intelligence & Process Automation intern at PETRONAS Carigali**, where I built Power BI dashboards, automated reporting workflows, and learned what "production-grade" actually means in an enterprise context.

---

## Connect

**Harist Hamzah Hutapea**
Big Data Analytics | Business Intelligence | Data Products

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/[yourhandle])
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-black)](https://[yourportfolio].vercel.app)
[![Email](https://img.shields.io/badge/Email-Reach%20Out-D14836?logo=gmail)](mailto:[youremail])

---

*Built by Harist Hamzah Hutapea · UTP Big Data Analytics · 2026*
