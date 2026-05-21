# Voice AI Pilot Go/No-Go Checklist

Date: 2026-05-21
Mission: voice-ai-pilot-validation-pack-v1
Artifact status: pilot decision checklist, not pilot results
Related artifacts: `pilot_validation/measurement_plan.md`, `pilot_validation/evidence_register.csv`, `financials/risk_adjusted_roi.md`

## Purpose

This checklist turns the pilot validation package into vertical-specific launch decisions. It should be completed after pilot data collection and before any production rollout decision.

Decision options:

- Go: thresholds are met, launch blockers are cleared, and measured value supports production expansion.
- Conditional go: core value is proven, but named remediation is required before wider rollout.
- Pause: data is incomplete, not comparable to baseline, or too noisy for decision-making.
- No-go: a hard blocker remains, measured workflow impact is negative, or risk-adjusted ROI is not defensible.

Do not treat provider claims, demo performance, or planning assumptions as pilot results. Every checked item must link to pilot evidence, owner sign-off, or measured telemetry.

## Universal gates for all verticals

| Gate | Go threshold | Conditional go | No-go trigger | Evidence owner |
|---|---|---|---|---|
| Baseline completeness | At least two weeks of usable baseline metrics for the workflow or a documented comparable control cohort | Baseline exists but has minor data gaps with approved sensitivity analysis | No baseline/control exists for the productivity claim being used in ROI | Pilot product owner |
| Measurement completeness | >=95% of pilot interactions have latency, outcome, cost, adoption, and error fields populated | 85-94% complete and missingness is not biased toward failures | <85% complete or missing failures/escalations | Technical pilot lead |
| Latency reporting | p50, p95, and p99 end-to-end latency reported by vertical and workflow | p95 reported, p99 pending, and no safety-critical decision depends on p99 | Only average latency is available | Technical pilot lead |
| Cost validation | Cost per completed interaction is measured and within 20% of model input or ROI is recalculated with actual cost | Cost exceeds model by 20-35% but risk-adjusted ROI remains positive with mitigation | Cost exceeds model enough to erase risk-adjusted ROI or billing export is unavailable | Finance / ops analyst |
| Adoption | Sustained active usage meets vertical threshold for two consecutive pilot weeks | Usage improves week-over-week but misses threshold by <=10 percentage points | Eligible users bypass the workflow or opt out at levels that invalidate ROI | Pilot product owner |
| Compliance | Required legal, consent, retention, audit, PHI/biometric, and data-processing gates are signed off before regulated data exposure | Non-blocking remediation has owner/date and pilot remains limited to approved data | Required legal approval, BAA, consent, retention, or biometric approval is missing for regulated data | Compliance owner |
| Human escalation | Escalation path exists, is tested, and high-risk cases route to a human | Escalation works but operational handoff needs tightening before scale | Human-in-loop path is missing or auditability is insufficient | Business owner |
| Risk-adjusted ROI | Updated model remains positive after measured adoption, latency, compliance, and provider risk adjustments | ROI remains positive but depends on one remediated risk factor with named owner/date | Risk-adjusted ROI is negative, unmeasured, or based mainly on planning assumptions | Finance / pilot product owner |

## Healthcare Omni checklist

Target workflow: ambient clinical documentation and patient intake/triage assistance using native realtime Omni models only.

Recommended decision threshold: healthcare requires the strictest latency and compliance posture. A technically successful pilot is still a no-go if PHI/legal gates are unresolved.

| Category | Go threshold | Conditional go | No-go trigger | Evidence required |
|---|---|---|---|---|
| Architecture fit | Native Omni stack used for clinical realtime paths; no STT + LLM + TTS cascade used for live physician conversation | Cascade used only for offline/non-clinical summarization with documented latency irrelevance | Cascade used for realtime clinical interaction where <500ms turn-taking is required | Provider config, architecture diagram, pilot logs |
| End-to-end latency | p95 <=500ms and p99 <=750ms for clinical turn-taking paths | p95 501-650ms with physician acceptance and async fallback for affected workflows | p95 >650ms, p99 unstable, or latency causes interruptions/correction burden | Benchmark harness output, call/session telemetry |
| Endpointing quality | Endpointing delay p95 <=250ms and failed-turn rate <=2% | Endpointing p95 251-400ms with no measurable physician productivity loss | Endpointing causes repeated interruptions, missed turns, or clinician rejection | Streaming logs, QA review |
| Documentation productivity | Net documentation time saved >=45 minutes per physician per day after correction/review time | 20-44 minutes saved with clear specialty/workflow segmentation | Correction/review time erases savings or note quality declines | Time study, EHR timestamps, clinician review |
| Clinical quality | >=90% note acceptance after first review and no critical clinical documentation defects | 80-89% acceptance with defect remediation plan | Critical defect or unsafe triage/documentation output | Clinical QA sample, defect log |
| Physician adoption | >=70% weekly active use among eligible pilot physicians for two consecutive weeks | 50-69% active use with specific friction backlog and clinical sponsor approval | <50% active use, high opt-out, or workflow bypass dominates | Usage logs, interview notes |
| PHI and BAA gate | Signed BAA/vendor terms, retention policy, encryption, access control, audit logging, and legal approval complete before PHI exposure | Synthetic/de-identified pilot may continue while BAA/legal work is pending | PHI is required but BAA/legal/retention approval is missing | Legal sign-off, vendor terms, audit config |
| Patient consent | Required disclosure/consent captured for >=99% of regulated interactions | Limited non-PHI pilot continues while consent workflow is fixed | Consent is not captured, not auditable, or jurisdictional rules are unresolved | Consent logs, compliance review |
| Healthcare Omni decision | Go when latency, clinical quality, physician adoption, PHI/BAA, consent, and positive risk-adjusted ROI all pass | Conditional go only for non-blocking remediation outside PHI/legal approval | No-go for unresolved PHI/BAA, unsafe clinical defects, or poor physician adoption | Signed decision record |

## Contact Center checklist

Target workflow: inbound routing, intent capture, agent assist, and post-call summarization using optimized cascade architecture where cost efficiency matters.

Recommended decision threshold: contact-center pilots can accept cascade architecture if it improves AHT and does not degrade customer experience.

| Category | Go threshold | Conditional go | No-go trigger | Evidence required |
|---|---|---|---|---|
| Architecture fit | Optimized cascade stack logs STT, LLM, TTS, orchestration, and total latency separately | Managed voice-agent platform accepted if component visibility is partial but total SLA is clear | Black-box stack cannot explain latency, errors, or cost drivers | Provider config, observability export |
| End-to-end latency | p95 <=800ms and p99 <=1200ms for live customer interactions | p95 801-950ms if AHT still improves and customer complaints do not rise | p95 >950ms or tail latency increases AHT/customer frustration | Benchmark harness output, call logs |
| Intent routing | >=90% correct routing/intent capture on QA sample | 80-89% with narrow intents remediated before scale | <80% or high-severity misroutes occur | QA sample, CRM/ticket outcomes |
| Summary quality | >=85% post-call summaries accepted without material agent rewrite | 75-84% with bounded defect categories and retraining/remediation | Summary corrections shift work back to agents or create compliance defects | QA review, agent correction logs |
| Productivity | AHT and after-call-work reduction together preserve positive risk-adjusted ROI; target >=20% AHT reduction against baseline | 10-19% AHT reduction if ACW reduction/cost savings still support ROI | AHT rises, ACW savings are not real, or productivity gain is not statistically credible | Workforce analytics, baseline comparison |
| Customer / agent adoption | >=75% agent weekly active use and manual takeover rate <=15% after ramp | 60-74% active use or takeover 16-25% with training/remediation plan | <60% active use, takeover >25%, or agents bypass the workflow | Agent console logs, supervisor feedback |
| Disclosure and recording controls | Required call disclosure, recording notice, retention class, and opt-out handling pass >=99% audit sample | Minor disclosure defects with limited rollout and compliance-approved remediation | Disclosure, recording, TCPA/state consent, or retention controls are missing | Compliance logs, call samples |
| Provider reliability and cost | Provider error/fallback rate <=2%; cost per completed interaction within ROI model tolerance | Error/fallback 2-5% with tested fallback and positive ROI | Error/fallback >5%, outages affect service levels, or cost breaks unit economics | Provider telemetry, billing export |
| Contact Center decision | Go when latency, AHT/ACW, quality, disclosure, cost, adoption, and risk-adjusted ROI all pass | Conditional go for bounded quality/reliability fixes with owner/date | No-go if tail latency worsens AHT, compliance controls fail, or ROI disappears | Signed decision record |

## Fraud checklist

Target workflow: voice verification support, synthetic voice/deepfake screening, alert triage, analyst assist, and audit workflow acceleration.

Recommended decision threshold: fraud workflows require measurable analyst lift plus conservative controls for false negatives, false positives, biometric processing, and auditability.

| Category | Go threshold | Conditional go | No-go trigger | Evidence required |
|---|---|---|---|---|
| Architecture fit | High-reliability Omni or optimized cascade selected by risk tier; human-in-loop required for adverse decisions | Lower-risk triage paths can use cascade while verification decisions remain supervised | Automated decisions occur without human review for high-risk fraud outcomes | Workflow design, escalation logs |
| Response and alert latency | p95 <=800ms and p99 <=1200ms for interactive verification/triage paths; alert generation meets operations SLA | p95 801-1000ms if analyst workflow speed still improves | Latency delays fraud response, increases abandonment, or breaks analyst workflow | Benchmark harness output, fraud queue timestamps |
| False negatives | False negative risk is bounded by QA sample, human escalation, and conservative thresholds; no critical miss in pilot sample | Sample size is limited but every uncertain case escalates to human review | Critical false negative occurs without escalation or measurement cannot bound miss rate | QA labels, confusion matrix, escalation audit |
| False positives | False positives do not materially increase customer friction or analyst workload; target precision threshold defined before pilot | False positives manageable with threshold tuning and manual review | False positives create unacceptable customer harm, account lockouts, or analyst overload | QA labels, customer impact review |
| Analyst productivity | Fraud resolution time improves >=20% or manual audit effort falls >=30% with no quality degradation | Smaller lift still positive after risk-adjusted ROI and operational review | Analyst correction burden or false alarms erase productivity savings | Case management logs, time study |
| Audit completeness | >=99% of pilot decisions have prompt/output, model/provider, timestamp, evidence, escalation, and reviewer fields retained per policy | 95-98% complete with remediation before scale | Audit log incomplete for regulated or disputed decisions | Audit export, compliance review |
| Biometric / consent controls | Biometric-processing approval, disclosure/consent, retention, jurisdiction review, and vendor data-processing terms complete before regulated use | Synthetic/de-identified pilot may continue while approvals are pending | Biometric/consent/legal approval missing for real customer verification | Legal sign-off, consent logs, vendor review |
| Provider resilience | Error/fallback rate <=2%, failover tested, and no single-provider outage can halt critical fraud response | Error/fallback 2-5% with human fallback and incident runbook | Provider outage, queueing, or quality drift creates unbounded operational risk | Provider telemetry, incident drill results |
| Fraud decision | Go when analyst lift, error controls, biometric/consent gates, auditability, provider resilience, and risk-adjusted ROI all pass | Conditional go for supervised low-risk paths only | No-go for unresolved biometric/consent gates, unbounded false negatives, or incomplete audit logs | Signed decision record |

## Decision record template

Use this section at the end of each pilot window.

| Field | Value |
|---|---|
| Vertical | Healthcare Omni / Contact Center / Fraud |
| Pilot dates | TBD |
| Decision | Go / Conditional go / Pause / No-go |
| Primary evidence links | TBD |
| Latency result | p50 TBD / p95 TBD / p99 TBD |
| Adoption result | TBD |
| Quality result | TBD |
| Compliance status | TBD |
| Cost per completed interaction | TBD |
| Risk-adjusted ROI result | TBD |
| Open blockers | TBD |
| Remediation owner/date | TBD |
| Executive approver | TBD |

## Non-negotiable no-go rules

- No Healthcare Omni production expansion without BAA/legal approval, consent/retention controls, and PHI-safe audit logging.
- No Contact Center production expansion if measured tail latency increases AHT or disclosure/recording controls fail.
- No Fraud production expansion if biometric/consent approval is unresolved, false negative risk is unbounded, or audit logs cannot support dispute review.
- No vertical advances from pilot to production on planning assumptions alone; measured target-environment evidence must replace the assumptions tracked in `pilot_validation/evidence_register.csv`.
