# Recommendations Validation Report

This document provides a 1:1 mapping of every major recommendation in VOICE_AI_BUSINESS_CASE.md to the supporting evidence in the research artifacts.

## 1. Omni vs. Optimized Cascade Architecture Recommendation
The business case recommends native Omni models (GPT-4o Realtime, Gemini 2.0 Flash) for Healthcare due to <500ms requirements and optimized cascaded pipelines (Pipecat + Deepgram + Groq + Cartesia) for Contact Center and Fraud use cases that tolerate 650-950ms within the <800ms target.
Evidence Link: latency_benchmarks.md L68,L72,L85 and VOICE_AI_BUSINESS_CASE.md L18,L66

## 2. VP of CX (Contact Center) Latency Target
Recommends optimized cascade architecture achieving 650-950ms to satisfy the <800ms Gold Standard for natural conversation flow in contact centers.
Evidence Link: latency_benchmarks.md L111 and VOICE_AI_BUSINESS_CASE.md L17,L60

## 3. CMIO (Healthcare) Latency Target
Mandates native Omni models only (280-480ms) because cascaded pipelines are disqualified for clinical safety and real-time documentation.
Evidence Link: latency_benchmarks.md L117 and VOICE_AI_BUSINESS_CASE.md L31,L66

## 4. Head of Fraud Latency Target
Recommends high-reliability providers with either optimized cascade or Omni for critical paths while staying under <800ms.
Evidence Link: latency_benchmarks.md L111 and VOICE_AI_BUSINESS_CASE.md L74

## 5. NPV and ROI Financial Figures
The aggregate deployment model now presents scenario ranges rather than a single headline case: 102-170 effective seats, $7.6M-$15.8M projected 3-year NPV, ~9,200%-17,100% ROI, 0.2-0.4 month payback, $3.141M-$6.355M annual labor savings, and $0.311M-$0.519M annual API/inference cost.
Evidence Link: financials/roi_analysis.md Aggregate Totals table and VOICE_AI_BUSINESS_CASE.md Financial Analysis section

## Consistency Check
Provider rows for GPT-4o Realtime (320-550ms) and Gemini 2.0 Flash (280-480ms) remain aligned with the business-case summary and provider recommendations. Financial claims are now expressed as scenario ranges matching `financials/cba_model.py` output, avoiding the stale single-point NPV/ROI overclaim flagged during final hardening audit.
