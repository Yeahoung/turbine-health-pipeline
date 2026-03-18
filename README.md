---
title: Turbine Health Pipeline Demo
emoji: 🌬️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
---

# Wind Turbine Daily Health Pipeline

A lightweight pipeline for extracting wind turbine SCADA anomalies, aggregating daily health metadata, and generating short maintenance-oriented reports using a fine-tuned local language model.

This project is designed as a simple demonstration pipeline rather than a full production system.

---

## Pipeline Overview

The pipeline contains three stages:

1. **Anomaly Extraction**

   Raw SCADA CSV data is scanned for abnormal conditions such as:

   - alarm codes
   - stop flags
   - system errors

   Continuous abnormal rows are grouped into **events**.

2. **Daily Metadata Aggregation**

   Events are aggregated per turbine per day to produce a structured
   **daily health metadata table** containing:

   - event counts
   - total abnormal duration
   - alarm codes
   - severity indicators
   - rule-based health label

3. **LLM Report Generation**

   A fine-tuned **Qwen 0.5B model (`qwen_0_5_fine`)** reads the daily metadata
   and generates a short **maintenance-oriented natural language report**
   containing:

   - health label
   - summary
   - maintenance advice

---

## Running the Pipeline

### Option 1: Command Line (`main.py`)

```
python main.py
```

Requires sample data in `data/raw/` (or symlink `data/demo/` to `data/raw/`).

### Option 2: Streamlit App

```
streamlit run app.py
```

Opens the interactive dashboard at http://localhost:8501 with Diagnostic Pipeline and RL Control Simulation tabs.

### Option 3: Local Development (Presentation + App)

```
./start_all.sh
# or: make start-all
```

- **Presentation:** http://localhost:8000  
- **App:** http://localhost:8501  
- Live Demo slide embeds the app when on port 8000.

### Option 4: Docker (Full-Stack)

```
docker compose up --build
# or: make docker-up
```

Single container serves both Presentation and App at **http://localhost:8080**:

- **Presentation:** http://localhost:8080/
- **App:** http://localhost:8080/app/
- Live Demo slide embeds the app via same-origin `/app/`.

The pipeline will:

1. extract anomalies  
2. build daily turbine metadata  
3. generate natural language maintenance reports

---

## Output

All outputs are written to the `out/` directory.

Generated maintenance reports can be found in:

```
out/generate_reports/
```

Each run creates a timestamped folder containing:

```
generated_reports.csv
generated_reports.txt
run_summary.csv
```

---

## Project Structure

```
project_root
│
├─ main.py
├─ app.py
│
├─ pipeline
│   ├─ label_anomalies.py
│   ├─ build_metadata.py
│   └─ generate_reports.py
│
├─ presentation        <- (Reveal.js slides; embeds app in Live Demo)
│   ├─ slides/
│   ├─ build.js
│   └─ public/         <- (built output)
│
├─ rl_agent            <- (Reinforcement Learning: PPO vs rule-based)
│
├─ data
│   ├─ demo/           <- (default demo data for app)
│   ├─ raw/            <- (for main.py; or symlink to demo)
│   └─ processed/      <- (training tables for model fine-tuning)
│
├─ .streamlit/         <- (config for iframe embedding)
├─ nginx.conf          <- (Docker: static + Streamlit proxy)
├─ Dockerfile
├─ docker-compose.yml
├─ docker-entrypoint.sh
└─ out/
```

---

## Model

The report generation stage uses a fine-tuned local model:

Fine turned model : auto download from [qwen_0_5_fine_report_generator](https://huggingface.co/LAND223/qwen_0_5_fine_report_generator)

The model is loaded using HuggingFace `transformers`.

A CPU fallback is supported for environments without GPU support.

---

## Deployment

| Target | Presentation | App |
|--------|--------------|-----|
| **Docker** | `localhost:8080/` | `localhost:8080/app/` |
| **Vercel + HF Spaces** | Deploy `presentation/` to Vercel | Deploy `app.py` to [Hugging Face Spaces](https://huggingface.co/spaces) |
| **Local (start_all.sh)** | `localhost:8000` | `localhost:8501` |

For Vercel: Presentation Live Demo slide embeds the HF Space URL. Build with `npm run build` in `presentation/`.

For Docker: Model is downloaded from Hugging Face at first pipeline run (requires network).

**Makefile:** `make start-all` | `make run-app` | `make build-presentation` | `make test-rl` | `make docker-up`

---

## Notes

- This repository focuses on demonstrating a **simple anomaly → metadata → report pipeline**.
- The processed training tables used during model fine-tuning are stored under `data/processed`.
- Demo data is in `data/demo/`; `main.py` expects `data/raw/` (symlink or copy as needed).

---

## Disclaimer

This project is intended for experimentation and demonstration purposes.
It is **not a production-grade turbine monitoring system**.
