# Voice AI Pilot Validation Measurement Plan

Date: 2026-05-21
Mission: voice-ai-pilot-validation-pack-v1
Artifact status: pilot measurement design; no measured pilot outcomes yet

## Purpose

This plan converts the voice AI business case into a target-environment validation program. It separates externally sourced planning assumptions from measurements that must be captured during pilots before any production investment decision.

The pilots cover three verticals:

1. Healthcare Omni: ambient documentation and triage workflows for physicians.
2. Contact Center Cascade: inbound routing and post-call summarization for support agents.
3. Fraud Operations: voice verification, fraud alerts, and analyst workflow acceleration.

## Measurement principles

- Treat existing report values as planning assumptions until observed in the buyer environment.
- Measure p50, p95, and p99 where operational risk depends on tail behavior.
- Compare each pilot against a baseline cohort or pre-pilot operating period.
- Log failed calls, escalations, opt-outs, and compliance blocks, not just successful interactions.
- Do not expose PHI, biometric identifiers, or regulated customer data until legal approval, consent language, retention policy, and BAA/vendor terms are complete.
- Tie each metric to a go/no-go threshold, risk adjustment, or ROI sensitivity input.

## Metric catalog

| Category | Metric | Definition | Source during pilot | Why it matters |
|---|---|---|---|---|
| Latency | End-to-end response latency | Time from user speech endpoint to audible AI response start; report p50/p95/p99 | Benchmark harness, call logs, provider telemetry | Validates natural turn-taking and vertical-specific thresholds |
| Latency | Endpointing delay | Time from user stop speaking to system detecting turn completion | Streaming STT/agent logs | Long endpointing delay can make otherwise fast models feel slow |
| Latency | Time to first token/audio | First model token or first synthesized audio packet after request dispatch | Provider telemetry | Identifies whether bottleneck is model, TTS, or orchestration |
| Reliability | Successful interaction rate | Completed target workflow interactions / attempted interactions | Pilot event logs | Feeds operational risk and adoption gates |
| Reliability | Provider error and fallback rate | API errors, timeouts, circuit-breaker events, fallback activations | Observability logs | Quantifies provider stability risk |
| Quality | Task completion | Workflow-specific successful outcome without human rework | QA review, CRM/EHR/work queue state | Connects AI performance to business value |
| Quality | Human correction rate | Percentage of AI outputs requiring correction or rewrite | QA samples, physician/agent review | Determines whether productivity gains are real |
| Adoption | Active user adoption | Eligible users completing minimum weekly pilot usage / eligible users | Seat roster and usage logs | Core driver of realized ROI |
| Adoption | Opt-out and override rate | User or customer refusals, manual takeovers, AI bypasses | Consent logs, agent console events | Detects trust, UX, and compliance friction |
| Cost | Cost per completed interaction | Provider, infrastructure, and monitoring cost / successful workflow completion | Billing export and event logs | Validates API/inference cost assumptions |
| Cost | Voice minutes per workflow | Billable voice minutes consumed by completed interaction | Provider billing and call metadata | Detects budget overrun risk |
| Compliance | Consent capture rate | Interactions with required disclosure and recorded opt-in / regulated interactions | Consent audit logs | Blocks legal exposure in voice recording/biometric workflows |
| Compliance | BAA / data-processing gate | Vendor and workflow legal approval status before PHI or regulated data | Legal checklist, vendor contracts | Required before Healthcare Omni PHI exposure |
| ROI | Baseline productivity delta | AHT reduction, documentation time saved, fraud resolution speedup vs. baseline | Workforce analytics, time studies, ticket/EHR/fraud tools | Converts pilot telemetry into ROI model inputs |
| ROI | Risk-adjusted net benefit | Gross modeled benefit discounted for adoption, latency, compliance, and provider risk | Financial model after pilot data import | Prevents pilot recommendation from overstating unproven value |

## Vertical-specific validation design

### Healthcare Omni

Target workflow: ambient clinical documentation and patient intake/triage assistance using native realtime Omni models only.

Planning assumptions to validate:

- End-to-end latency remains below the healthcare threshold in real clinical network conditions.
- Physicians save meaningful documentation time without increasing correction burden.
- PHI handling is covered by BAA, retention, encryption, audit logging, and legal sign-off before production-like data exposure.

Primary measurements:

- Latency: p50/p95/p99 end-to-end response latency; endpointing delay; failed turn rate.
- Quality: physician correction rate, note acceptance rate, documentation completeness, manual override frequency.
- Adoption: weekly active physician usage, opt-out rate, reported friction from physician interviews.
- Compliance: BAA status, consent workflow pass rate, retention policy enforcement, PHI test-data boundary.
- ROI: documentation minutes saved per physician per day, net of correction/review time.

Initial go/no-go inputs:

- Go only if latency, compliance readiness, and physician correction burden support the existing Healthcare Omni business case.
- No-go if BAA/legal approval is missing for PHI, if p95 latency breaks clinical turn-taking, or if correction time erases documentation savings.

### Contact Center Cascade

Target workflow: inbound routing, intent capture, and post-call summarization using optimized cascade architecture where cost efficiency matters.

Planning assumptions to validate:

- Optimized cascades can meet acceptable conversational latency under real call-center load.
- AHT and after-call-work reduction are measurable against baseline.
- Customer disclosure, recording, and escalation controls operate consistently.

Primary measurements:

- Latency: p50/p95/p99 end-to-end response latency; STT, LLM, and TTS component timings.
- Quality: intent-routing accuracy, summary acceptance rate, first-contact resolution impact.
- Adoption: agent usage rate, manual takeover rate, supervisor exceptions.
- Cost: cost per completed interaction and voice minutes per call type.
- ROI: AHT reduction, after-call-work reduction, QA labor avoided, net of provider cost.

Initial go/no-go inputs:

- Go if latency stays within the contact-center threshold and measured productivity deltas preserve positive ROI after provider cost.
- No-go if tail latency increases AHT, summary corrections shift work back to agents, or consent/disclosure gaps appear.

### Fraud Operations

Target workflow: voice verification support, synthetic voice/deepfake screening, alert triage, and analyst workflow acceleration.

Planning assumptions to validate:

- Voice AI improves fraud resolution speed without unacceptable false positives or false negatives.
- Reliability is high enough for regulated financial workflows with human-in-loop escalation.
- Audit artifacts are complete enough for compliance and dispute review.

Primary measurements:

- Latency: p50/p95/p99 response and alert-generation latency for fraud interaction paths.
- Quality: true/false positive screening outcomes, analyst correction rate, escalation accuracy.
- Adoption: analyst active usage, override rate, human-in-loop escalation rate.
- Compliance: disclosure, consent, biometric-processing approval, audit-log completeness.
- ROI: fraud case resolution time saved, avoided manual review effort, prevented loss estimates if available.

Initial go/no-go inputs:

- Go if measured resolution speed and audit quality support ROI without creating unacceptable compliance exposure.
- No-go if biometric/consent controls are unresolved, false negative risk is not bounded, or audit logs are incomplete.

## Baseline and control requirements

Before pilot launch:

- Capture at least two weeks of baseline metrics for AHT, after-call work, physician documentation time, fraud resolution time, QA correction rates, and escalation rates where available.
- Define pilot cohorts and comparable control cohorts or pre/post windows.
- Freeze metric definitions so post-pilot ROI is not reverse-engineered.
- Confirm provider billing export access and event-log retention.
- Confirm legal approval for all consent, recording, PHI, biometric, and retention controls.

## Data collection plan

| Data source | Minimum fields | Owner | Frequency |
|---|---|---|---|
| Benchmark harness output | timestamp, vertical, workflow, provider_stack, p50/p95/p99 latency, error count, fallback count | Technical pilot lead | Daily during pilot |
| Provider billing export | provider, billable minutes, unit price, total cost, workflow tag | Finance / ops analyst | Weekly |
| CRM/EHR/fraud workflow logs | workflow ID, baseline duration, pilot duration, outcome, escalation, correction flag | Business system owner | Daily extract |
| QA review sample | output ID, defect type, correction minutes, severity, reviewer | QA lead / clinical reviewer / fraud reviewer | Weekly sample |
| Consent and compliance logs | disclosure presented, opt-in captured, BAA/legal gate, retention class, audit ID | Compliance owner | Every regulated interaction |
| User feedback | user role, friction score, trust score, qualitative blockers | Pilot product owner | Weekly |

## Analysis method

1. Validate metric completeness and reject incomplete pilot windows.
2. Calculate latency distributions by vertical, provider, workflow, and time window.
3. Compare productivity metrics against baseline and control cohort where available.
4. Calculate cost per completed interaction and per successful business outcome.
5. Convert measured adoption, latency misses, compliance gaps, and provider errors into explicit ROI risk discounts.
6. Update the ROI model with measured deltas rather than report assumptions.
7. Produce go/no-go recommendation per vertical with evidence links and caveats.

## ROI validation formulas

Use measured pilot data to update the financial model:

- Gross labor benefit = baseline work minutes avoided * fully loaded hourly rate.
- Net operating benefit = gross labor benefit - provider/API cost - incremental QA/review cost - monitoring/support cost.
- Adoption-adjusted benefit = net operating benefit * measured active adoption rate.
- Latency-adjusted benefit = adoption-adjusted benefit * latency pass factor.
- Compliance-adjusted benefit = latency-adjusted benefit * compliance readiness factor.
- Risk-adjusted ROI = (risk-adjusted net benefit - implementation cost) / implementation cost.

Risk discounts must be explicit, bounded, and documented. A compliance blocker for PHI or biometric exposure should be treated as a launch blocker, not a small percentage discount.

## Pilot exit artifacts

At the end of each pilot window, produce:

- Latency summary with p50/p95/p99 and failure modes.
- Cost summary by provider stack and workflow.
- Adoption and opt-out summary by user role.
- Compliance readiness evidence and unresolved legal gates.
- ROI update comparing original assumption, measured value, and risk-adjusted value.
- Recommendation: go, conditional go, pause, or no-go.

## Open assumptions to track in evidence register

- Provider latency and reliability values are externally sourced benchmarks until measured in the target environment.
- Contact Center AHT reduction is a planning assumption until measured against baseline calls.
- Physician documentation savings are planning assumptions until time-study data is captured.
- Fraud resolution speedup and audit reduction require pilot evidence before production ROI claims.
- Legal readiness cannot be inferred from vendor capability claims; it requires documented BAA, consent, retention, and jurisdiction review.
