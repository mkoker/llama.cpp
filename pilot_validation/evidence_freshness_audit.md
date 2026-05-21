# Evidence Freshness Audit

Date: 2026-05-21
Mission: voice-ai-pilot-decision-readiness-validation-v3
Artifact status: decision-readiness evidence freshness audit; no measured pilot outcomes yet
Evidence register: `pilot_validation/evidence_register.csv`

## Purpose

This audit checks whether every evidence-register row has an accountable freshness record before it is used in a go/no-go packet. It does not convert provider documentation, vendor claims, or planning assumptions into measured pilot performance. Rows with live URLs that are reachable remain planning evidence unless the measurement plan replaces them with benchmark output, billing exports, QA samples, or legal/compliance sign-off.

## Audit method

- Read `pilot_validation/evidence_register.csv` as the source of truth.
- Confirmed required freshness fields are populated: `last_checked`, `next_review`, `owner`, and `caveat`.
- Performed lightweight URL reachability checks where network access allowed; blocked or dynamic vendor pages are marked caveated rather than treated as stale proof.
- Preserved the dry-run/planning boundary: current source status supports pilot planning, not final ROI or production go/no-go claims.

## Freshness register

| # | assumption | source status | last-checked | next-review | owner | caveat |
|---:|---|---|---|---|---|---|
| 1 | OpenAI GPT-4o Realtime can support native omni voice interactions at a planning target of 320-550 ms end-to-end latency for selected healthcare and fraud paths. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Official docs support realtime API capability but not a universal p50/p95 SLA; target-environment benchmark required before production ROI claims. |
| 2 | Google Gemini native/live audio can support the healthcare omni path at a planning target of 280-480 ms end-to-end latency. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Official docs support audio capability; exact latency is region, model, network, and streaming-configuration dependent and must be measured in pilot. |
| 3 | Google Gemini Live / native multimodal API is feasible for low-latency speech-to-speech validation. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Capability source does not prove buyer-network p95 or p99 latency; benchmark harness must capture tail behavior. |
| 4 | Optimized cascaded voice stacks using streaming STT, fast LLM inference, and low-latency TTS can be suitable for contact-center workflows when measured latency stays within threshold. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Framework feasibility does not equal a production latency SLA; additive network, endpointing, orchestration, and provider queueing must be measured. |
| 5 | Deepgram Nova streaming can provide low-latency STT for optimized cascades in contact-center and compliance-audit flows. | caveated: HTTP 404 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Source supports streaming STT capability; endpointing delay, noise, utterance length, and geography can change production latency and accuracy. |
| 6 | Cartesia Sonic can provide low-latency TTS for optimized cascades and help keep contact-center end-to-end latency within target. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Vendor docs support API capability and low-latency positioning; first-audio latency depends on voice, model, streaming settings, region, and network. |
| 7 | ElevenLabs streaming TTS is feasible for high-quality voice paths but may carry higher cost and variable latency by model and voice. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Use selected model and voice in the pilot; high per-minute cost can erode ROI if used broadly instead of premium paths. |
| 8 | Groq-hosted Llama inference can contribute low TTFT in cascaded stacks for routing, summaries, and non-clinical voice-agent reasoning. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | TTFT is prompt-size, model, queueing, and region sensitive; pilot must log model timing separately from STT and TTS. |
| 9 | Contact Center AHT reduction of roughly 30-40% is a planning assumption for routing and automation workflows. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Contact center operations owner | Productivity lift must be measured against baseline calls and control cohorts; tail latency or correction work could erase modeled savings. |
| 10 | Managed voice-agent platforms such as Vapi and Retell are feasible managed cascade options for contact-center pilots. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Technical pilot lead | Vendor explainer supports architecture feasibility but not buyer-specific latency, reliability, integration, or ROI outcomes. |
| 11 | Physician documentation savings of 1.5-2 hours per physician per day are planning assumptions until measured with clinical time-study data. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Clinical pilot owner | Requires physician review workflow, correction-time tracking, EHR integration measurement, and PHI/BAA legal readiness before production claims. |
| 12 | Healthcare voice AI requires strict PHI controls, BAA/vendor review, consent, retention, encryption, and audit logging before production-like data exposure. | caveated: HEAD HTTP 403; GET check failed (HTTPError) | 2026-05-21 | 2026-06-20 | Compliance owner | Technical readiness is not legal approval; legal/compliance owners must review exact workflow, vendor terms, and data handling before PHI use. |
| 13 | Fraud resolution speedup, manual audit reduction, and voice-verification value require pilot evidence before production ROI claims. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Fraud operations owner | Pilot must bound false positives, false negatives, disclosure/consent handling, biometric processing approval, and audit-log completeness. |
| 14 | Voice recording, consent, telemarketing, and biometric-processing controls can become launch blockers for outbound fraud verification and regulated voice workflows. | caveated: freshness probe failed (TimeoutError) | 2026-05-21 | 2026-06-20 | Compliance owner | Jurisdiction-specific counsel must determine whether a workflow is allowed; unresolved legal gates should block launch rather than become a small ROI discount. |
| 15 | Published provider docs and vendor benchmarks are sufficient for planning but not sufficient for final ROI or go/no-go decisions. | reachable HTTP 200 | 2026-05-21 | 2026-06-20 | Pilot product owner | Evidence register should be refreshed after benchmark harness output, billing exports, QA samples, and legal review replace planning assumptions. |

## Decision-readiness interpretation

- Fresh enough for planning: all evidence rows have a last-checked date, next-review date, owner, source URL, confidence label, and caveat.
- Not fresh enough for production claims: provider docs and vendor explainers still need target-environment pilot measurements before measured-performance, ROI, or scale-up claims.
- Rows owned by Compliance owner remain legal-gate evidence only; technical readiness does not imply legal approval for PHI, recording, consent, biometric, retention, or jurisdiction-specific use.
- The next review window is 2026-06-20 for the current register. Any pilot launch decision after that date should refresh URLs, vendor terms, pricing, and legal references before reuse.

## Owner follow-up

| owner | follow-up before limited pilot |
|---|---|
| Technical pilot lead | Replace provider capability claims with benchmark harness p50/p95/p99 latency, provider error, fallback, and component timing output. |
| Contact center operations owner | Replace AHT planning assumption with baseline/control cohort call data and post-pilot productivity deltas. |
| Clinical pilot owner | Replace physician documentation savings assumption with time-study data, correction burden, and clinical QA review. |
| Fraud operations owner | Replace fraud productivity assumptions with case-resolution, audit workload, false-positive, and false-negative measurements. |
| Compliance owner | Record legal approval status separately from technical readiness, including BAA/vendor terms, consent/disclosure, retention, biometric, and jurisdiction review. |
| Pilot product owner | Refresh the full register after pilot telemetry, billing exports, QA samples, and legal review supersede planning assumptions. |

