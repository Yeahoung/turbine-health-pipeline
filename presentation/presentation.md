# Slide 1 — Title

# AI-Assisted Wind Turbine Health Monitoring

## and Decision Support System

### Integrating SCADA analytics, reinforcement learning, and explainable AI for safe turbine operation support

---

### Core idea

Wind farms produce massive SCADA telemetry, yet operators still face a gap between **fault detection** and **operational decision making**.

This project integrates:

* turbine health monitoring
* explainable maintenance reporting
* reinforcement-learning advisory optimization

into a **single decision-support workflow**.

---

### Key message

This system **does not autonomously control turbines**.

AI provides **interpretable recommendations**, while **human operators remain the final decision-makers**.

---

# Slide 2 — Problem

# Problem & Motivation

Wind turbines operate under:

* variable wind conditions
* turbulence and transient disturbances
* complex mechanical systems

SCADA platforms collect large volumes of data, but turning those signals into **actionable operational guidance** remains difficult.

---

### Current workflow limitations

Most monitoring systems focus on:

* anomaly detection
* alarm generation
* event logging

However, they rarely provide:

* interpretable maintenance context
* operational recommendations
* optimization insights.

---

### Our goal

Build an AI-assisted pipeline that:

1. detects abnormal behaviour
2. explains maintenance implications
3. provides optimization advice **only when safe**

---

# Slide 3 — System Overview

# Overall System Concept

The system is designed as an **integrated decision-support architecture**.

---

### Three coordinated flows

**1. Operational data flow**

SCADA → anomaly detection → health metadata → explanation → operator briefing

---

**2. Simulation advisory flow**

Reinforcement learning explores operational strategies in simulation and provides **optimization advice**.

---

**3. Human-in-the-loop flow**

Operators review AI outputs before any operational adoption.

---

### Key principle

AI **supports turbine operations**,
but **does not directly control turbine hardware**.

---

# Slide 4 — System Pipeline

# Integrated Decision Pipeline

The system connects monitoring, explanation, and optimization into one workflow.

---

### Stage 1 — SCADA anomaly extraction

Raw SCADA rows are scanned for:

* alarm codes
* stop events
* abnormal operational signals

Output:

* anomaly rows
* grouped event records

---

### Stage 2 — Daily health metadata

Events are aggregated into interpretable indicators such as:

* abnormal duration
* event count
* dominant alarms
* severity level
* health label

---

### Stage 3 — LLM maintenance summary

A language model converts structured metadata into **maintenance-style summaries**.

Example output:

> Review pitch lubrication system and verify whether corrective action is required.

---

### Stage 4 — Health gate

Health status determines whether optimization advice is allowed.

---

### Stage 5 — Operator briefing

The final output becomes a **turbine-day operational briefing**.

---

# Slide 5 — Health Metadata Layer

# Turbine Health Metadata

The metadata layer converts raw signals into **interpretable engineering features**.

---

### Example indicators

* event_count
* abnormal_duration
* dominant_alarm
* severity_level
* turbine_health_label

---

### Why this layer matters

It ensures the system remains **interpretable**.

Instead of opaque latent representations, downstream modules use **engineer-readable indicators**.

---

### Engineering advantage

Operators can validate the data directly against SCADA logs.

This improves:

* explainability
* trust
* operational adoption.

---

# Slide 6 — RL Advisory Module

# Reinforcement Learning Advisory Module

The RL component explores turbine operation strategies **in simulation**.

It balances objectives such as:

* power capture
* structural load reduction
* control smoothness

---

### Important constraint

RL **does not control real turbines in this prototype**.

Instead, it provides **advisory optimization guidance**.

---

### Example control variables

Typical optimization targets may include:

* pitch angle adjustment
* torque smoothing
* load balancing

---

### Why RL is useful

Simulation allows exploration of operational strategies that may be difficult to test directly in real turbines.

---

# Slide 7 — Case Study

# Example Turbine-Day Briefings

The demo presents three example turbine cases.

The goal is not only anomaly detection but **operator decision guidance**.

---

### Turbine A — ATTENTION state

Observed condition:

* abnormal event lasting several hours
* alarm code detected

Recommended action:

* review SCADA logs
* confirm operational status

Control policy:

RL allowed only as **limited advisory**.

---

### Turbine B — ALARM state

Observed condition:

* stopping event triggered
* pitch lubrication alarm

Recommended action:

* inspect pitch lubrication system
* prioritize maintenance

Control policy:

RL optimization **disabled**.

---

### Turbine C — repeated alarm pattern

Observed condition:

* repeated pitch system alarms

Recommended action:

* maintenance investigation required

Control policy:

optimization advice **suppressed**.

---

# Slide 8 — Explainable & Controllable AI

# Explainable and Controllable AI Design

A key design objective is **interpretable AI behaviour**.

---

### Explainability mechanisms

The system uses:

* interpretable health metadata
* explicit rule-based safety gating
* transparent reward functions

Each feature corresponds to measurable turbine properties.

---

### Controllability mechanisms

Safety governance ensures:

* health-state gating of optimization
* maintenance status overrides optimization
* human operator remains the final authority.

---

### System philosophy

AI assists operators rather than replacing them.

---

# Slide 9 — Engineering Value

# Engineering Value

The integrated system creates value across several dimensions.

---

### Operational efficiency

Reduced latency from anomaly detection to operator action.

---

### Maintenance planning

Better identification of maintenance priorities and abnormal patterns.

---

### Sustainability

Improved turbine operation may reduce structural stress and extend component lifetime.

---

### Carbon impact

More efficient turbine operation can increase renewable energy output and reduce fossil backup reliance.

---

### Core contribution

The innovation lies in **system-level integration**, not only model performance.

---

# Slide 10 — Deployment Path

# Deployment Path and Future Work

The current prototype demonstrates the full decision-support pipeline.

---

### Prototype capabilities

* SCADA anomaly extraction
* turbine health metadata generation
* LLM maintenance summaries
* RL advisory integration
* unified operator briefing

---

### Future development

Possible next steps include:

* richer SCADA datasets
* higher-fidelity turbine simulations
* multi-turbine optimization
* integration with wind-farm digital twins

---

### Long-term vision

A scalable AI system for **continuous wind-farm monitoring and decision support**.

---

# Final Slide — Conclusion

# Conclusion

This project demonstrates a safe and explainable approach to AI-assisted turbine operations.

---

### Key message

The system integrates:

* SCADA health monitoring
* explainable maintenance reporting
* reinforcement-learning optimization

into a **single operator-facing decision workflow**.

---

### Safety principle

AI does **not autonomously control turbines**.

Instead it provides **interpretable recommendations within safety constraints**.

---

### Final takeaway

AI can improve wind-farm operations when it is designed as **transparent, controllable, and human-centered decision support**.

---

### Thank You
AI-Assisted Wind Turbine Health Monitoring and Decision Support

Integrating monitoring, explanation, and optimization for safer turbine operation.

Thank you for your attention.
