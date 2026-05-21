# Pilot Validation Artifact Inventory

Inventory date: 2026-05-21

## Current validation artifacts

| Artifact | Status | Decision-readiness role |
| --- | --- | --- |
| `pilot_validation/validate_results.py` | Present | Stdlib result validator used by dry-run and fixture gates. Enforces required top-level keys, expected vertical coverage, dry-run no-external-call boundary, scenario/sample shape, latency recalculation, summary consistency, and expected dry-run data classification. |
| `pilot_validation/result_schema.json` | Present | Formal JSON Schema contract for benchmark output. It declares `additionalProperties: false` at root, scenario, summary, latency, and sample levels, but the current runtime validator only enforces a subset of this contract. |
| `pilot_validation/dry_run_results.json` | Present and passes validator | Deterministic dry-run result set. Contains three scenarios, 24 samples per scenario, no external API calls, all synthetic/no-PHI/no-biometric/no-customer-data classification. |
| `pilot_validation/benchmark_harness.py` | Present | Produces or supports generation of benchmark output consumed by the validator. |
| `pilot_validation/validate_evidence_register.py` | Present | Separate evidence-register validator; outside Task 1 gate but relevant for later evidence freshness work. |
| `pilot_validation/evidence_register.csv` | Present | Evidence source register for later freshness audit. |
| `pilot_validation/measurement_plan.md` | Present | Measurement plan context for dry-run versus measured-performance boundary. |
| `pilot_validation/go_no_go_checklist.md` | Present | Current checklist for go/no-go readiness. |

## Dry-run fixture coverage baseline

| Scenario | Samples | Success rate | p95 ms | Target p95 ms | Validator status |
| --- | ---: | ---: | ---: | ---: | --- |
| Healthcare Omni | 24 | 1.0 | 851.3 | 1200 | Pass |
| Contact Center Cascade | 24 | 1.0 | 1301.8 | 1800 | Pass |
| Fraud Operations | 24 | 1.0 | 1076.8 | 1500 | Pass |

Dry-run boundary observed: `dry_run=true`, `external_api_calls=0`, expected three verticals present exactly, and every scenario is classified as `synthetic_no_phi_no_biometrics_no_customer_data`.

## Existing fixture coverage

| Fixture | Type | Current expected behavior | Covered failure mode | Gap against mission success criteria |
| --- | --- | --- | --- | --- |
| `pilot_validation/fixtures/malformed_result.json` | Result JSON negative fixture | Exits non-zero under `validate_results.py` | Broad malformed result: external API call in dry-run, zero iterations, empty scenario list | Too broad; does not isolate missing required field, wrong type, invalid value, or unexpected extra field behavior. |
| `pilot_validation/fixtures/malformed_evidence_register.csv` | Evidence CSV negative fixture | Exits non-zero under `validate_evidence_register.py` | Evidence-register malformed row/check coverage | Not a result fixture; does not count toward negative result fixture coverage. |

## Coverage gaps for next tasks

1. Missing required result field fixture: no fixture currently isolates a required key omission.
2. Wrong result field type fixture: no fixture currently isolates an incorrect type while keeping the rest of the result structurally valid.
3. Invalid/out-of-range value fixture: `malformed_result.json` includes out-of-range `iterations_per_scenario=0`, but it is bundled with other failures and does not map cleanly to a single decision risk.
4. Unexpected extra field fixture: `result_schema.json` rejects extra fields formally, but `validate_results.py` does not currently enforce `additionalProperties: false`; this is an implementation gap to resolve before the extra-field fixture can fail for the right reason.

## Validator behavior notes

- The runtime validator validates semantic invariants beyond JSON Schema, including summary recomputation and exact expected vertical set.
- The runtime validator does not currently perform full JSON Schema validation and does not reject unexpected additional properties on its own.
- Future negative fixtures should include a short decision-risk comment in adjacent documentation or fixture naming, because JSON does not support comments.
