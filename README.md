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

The entry point is:

```
main.py
```

A sample input file is provided:

```
data/raw/2016_01_01.csv
```

Run:

```
python main.py
```

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
│
├─ pipeline
│   ├─ label_anomalies.py
│   ├─ build_metadata.py
│   └─ generate_reports.py
│
├─ data
│   ├─ raw
│   │   └─ 2016_01_01.csv
│   │
│   └─ processed
│       └─ (training tables used for model fine-tuning)
│
└─ out


```

---

## Model

The report generation stage uses a fine-tuned local model:

Fine turned model : auto download from [qwen_0_5_fine_report_generator](https://huggingface.co/LAND223/qwen_0_5_fine_report_generator)

The model is loaded using HuggingFace `transformers`.

A CPU fallback is supported for environments without GPU support.

---

## Notes

- This repository focuses on demonstrating a **simple anomaly → metadata → report pipeline**.
- The processed training tables used during model fine-tuning are stored under:

```
data/processed
```

- Additional raw SCADA data can be added to `data/raw/`.

---

## Disclaimer

This project is intended for experimentation and demonstration purposes.
It is **not a production-grade turbine monitoring system**.
