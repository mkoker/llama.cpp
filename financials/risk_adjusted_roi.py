#!/usr/bin/env python3
"""
Risk-adjusted ROI model for the Voice AI pilot validation pack.

Self-contained planning model that applies explicit, bounded discounts for
adoption, latency, compliance, and provider risk to the existing Conservative,
Base, and Aggressive scenario structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


DEFAULTS = {
    "contact_center_agents": 100,
    "physicians": 50,
    "fraud_analysts": 20,
    "agent_hourly_rate": 28.0,
    "physician_hourly_rate": 120.0,
    "fraud_analyst_hourly_rate": 55.0,
    "discount_rate": 0.10,
    "time_horizon_years": 3,
    "annual_maint_pct": 0.15,
    "health_docs_hours_saved_per_day": 1.75,
    "fraud_resolution_speedup": 0.40,
    "fraud_audit_reduction": 0.70,
    "contact_center_min_per_agent_day": 120,
    "health_min_per_physician_day": 180,
    "fraud_min_per_analyst_day": 90,
    "working_days_per_year": 250,
}

SCENARIOS = {
    "Conservative": {
        "adoption_rate": 0.60,
        "contact_center_aht_reduction": 0.25,
        "one_time_impl_per_100_seats": 90_000.0,
    },
    "Base": {
        "adoption_rate": 0.80,
        "contact_center_aht_reduction": 0.35,
        "one_time_impl_per_100_seats": 75_000.0,
    },
    "Aggressive": {
        "adoption_rate": 1.00,
        "contact_center_aht_reduction": 0.45,
        "one_time_impl_per_100_seats": 60_000.0,
    },
}

# Per-minute planning costs: contact center/fraud cascade, healthcare omni.
COST_PER_MINUTE = {
    "contact_center": 0.1243,
    "health": 0.04,
    "financial": 0.1243,
}


@dataclass(frozen=True)
class RiskFactor:
    name: str
    pass_factor: float
    rationale: str

    @property
    def risk_discount(self) -> float:
        """Discount applied by this risk factor, bounded to 0-100%."""
        bounded_pass_factor = min(max(self.pass_factor, 0.0), 1.0)
        return 1.0 - bounded_pass_factor


@dataclass(frozen=True)
class ScenarioFinancials:
    scenario: str
    one_time_impl: float
    annual_net_benefit: float
    npv: float
    roi_pct: float
    payback_months: float


@dataclass(frozen=True)
class RiskAdjustedScenario:
    scenario: str
    gross_npv: float
    gross_roi_pct: float
    risk_adjusted_npv: float
    risk_adjusted_roi_pct: float
    gross_payback_months: float
    risk_adjusted_payback_months: float
    annual_net_benefit: float
    risk_adjusted_annual_net_benefit: float
    composite_pass_factor: float
    risk_factors: List[RiskFactor]


RISK_FACTORS: Dict[str, List[RiskFactor]] = {
    "Conservative": [
        RiskFactor("adoption", 0.90, "60% rollout is already modeled; extra discount captures opt-out and weekly-active uncertainty."),
        RiskFactor("latency", 0.88, "Tail latency remains unmeasured in target network and could reduce realized productivity."),
        RiskFactor("compliance", 0.80, "PHI, consent, recording, and biometric approvals are launch blockers until cleared."),
        RiskFactor("provider", 0.90, "Provider errors, fallback rates, and billing variance are unvalidated."),
    ],
    "Base": [
        RiskFactor("adoption", 0.85, "80% adoption is plausible but still requires sustained weekly active use."),
        RiskFactor("latency", 0.90, "Latency assumptions need p95/p99 benchmark confirmation by vertical."),
        RiskFactor("compliance", 0.85, "Legal and data-handling controls must pass before production-like data exposure."),
        RiskFactor("provider", 0.92, "Availability, failover, and unit costs need pilot billing/log validation."),
    ],
    "Aggressive": [
        RiskFactor("adoption", 0.80, "100% adoption is unlikely without workflow friction and override issues."),
        RiskFactor("latency", 0.86, "Aggressive productivity gains are more exposed to poor conversational turn-taking."),
        RiskFactor("compliance", 0.82, "Broader rollout increases unresolved PHI/biometric/recording exposure."),
        RiskFactor("provider", 0.88, "Higher volume increases vendor concentration and queueing/cost variance."),
    ],
}


def compute_npv(cash_flows: Iterable[float], discount_rate: float) -> float:
    return sum(cash_flow / ((1.0 + discount_rate) ** year) for year, cash_flow in enumerate(cash_flows))


def calculate_roi(one_time_impl: float, total_net_benefit_over_horizon: float) -> float:
    if one_time_impl <= 0:
        return 0.0
    return (total_net_benefit_over_horizon / one_time_impl) * 100.0


def calculate_payback_months(one_time_impl: float, annual_net_benefit: float) -> float:
    if annual_net_benefit <= 0:
        return float("inf")
    return (one_time_impl / annual_net_benefit) * 12.0


def composite_pass_factor(risk_factors: Iterable[RiskFactor]) -> float:
    pass_factor = 1.0
    for factor in risk_factors:
        pass_factor *= min(max(factor.pass_factor, 0.0), 1.0)
    return pass_factor


def scenario_financials(scenario_name: str) -> ScenarioFinancials:
    assumptions = DEFAULTS.copy()
    assumptions.update(SCENARIOS[scenario_name])
    adoption = assumptions["adoption_rate"]

    contact_center_agents = int(DEFAULTS["contact_center_agents"] * adoption)
    physicians = int(DEFAULTS["physicians"] * adoption)
    fraud_analysts = int(DEFAULTS["fraud_analysts"] * adoption)
    working_days = assumptions["working_days_per_year"]

    contact_center_labor_savings = (
        contact_center_agents
        * assumptions["agent_hourly_rate"]
        * 8
        * working_days
        * assumptions["contact_center_aht_reduction"]
    )
    physician_labor_savings = (
        physicians
        * assumptions["health_docs_hours_saved_per_day"]
        * assumptions["physician_hourly_rate"]
        * working_days
    )
    fraud_labor_savings = (
        fraud_analysts
        * assumptions["fraud_analyst_hourly_rate"]
        * 8
        * working_days
        * ((assumptions["fraud_resolution_speedup"] + assumptions["fraud_audit_reduction"]) / 2.0)
    )
    annual_labor_savings = contact_center_labor_savings + physician_labor_savings + fraud_labor_savings

    annual_api_cost = (
        contact_center_agents * assumptions["contact_center_min_per_agent_day"] * working_days * COST_PER_MINUTE["contact_center"]
        + physicians * assumptions["health_min_per_physician_day"] * working_days * COST_PER_MINUTE["health"]
        + fraud_analysts * assumptions["fraud_min_per_analyst_day"] * working_days * COST_PER_MINUTE["financial"]
    )

    total_seats = contact_center_agents + physicians + fraud_analysts
    one_time_impl = (total_seats / 100.0) * assumptions["one_time_impl_per_100_seats"]
    annual_maint = one_time_impl * assumptions["annual_maint_pct"]
    annual_net_benefit = annual_labor_savings - annual_api_cost - annual_maint

    cash_flows = [annual_net_benefit - one_time_impl]
    for _ in range(1, assumptions["time_horizon_years"]):
        cash_flows.append(annual_net_benefit)

    horizon_net = annual_net_benefit * assumptions["time_horizon_years"]
    return ScenarioFinancials(
        scenario=scenario_name,
        one_time_impl=one_time_impl,
        annual_net_benefit=annual_net_benefit,
        npv=compute_npv(cash_flows, assumptions["discount_rate"]),
        roi_pct=calculate_roi(one_time_impl, horizon_net),
        payback_months=calculate_payback_months(one_time_impl, annual_net_benefit),
    )


def risk_adjust_scenario(scenario_name: str) -> RiskAdjustedScenario:
    financials = scenario_financials(scenario_name)
    risk_factors = RISK_FACTORS[scenario_name]
    pass_factor = composite_pass_factor(risk_factors)

    risk_adjusted_annual_net = financials.annual_net_benefit * pass_factor
    years = DEFAULTS["time_horizon_years"]
    risk_adjusted_cash_flows = [risk_adjusted_annual_net - financials.one_time_impl]
    for _ in range(1, years):
        risk_adjusted_cash_flows.append(risk_adjusted_annual_net)

    risk_adjusted_horizon_net = risk_adjusted_annual_net * years
    risk_adjusted_npv = compute_npv(risk_adjusted_cash_flows, DEFAULTS["discount_rate"])
    risk_adjusted_roi = calculate_roi(financials.one_time_impl, risk_adjusted_horizon_net)
    risk_adjusted_payback = calculate_payback_months(financials.one_time_impl, risk_adjusted_annual_net)

    return RiskAdjustedScenario(
        scenario=scenario_name,
        gross_npv=financials.npv,
        gross_roi_pct=financials.roi_pct,
        risk_adjusted_npv=risk_adjusted_npv,
        risk_adjusted_roi_pct=risk_adjusted_roi,
        gross_payback_months=financials.payback_months,
        risk_adjusted_payback_months=risk_adjusted_payback,
        annual_net_benefit=financials.annual_net_benefit,
        risk_adjusted_annual_net_benefit=risk_adjusted_annual_net,
        composite_pass_factor=pass_factor,
        risk_factors=risk_factors,
    )


def fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def main() -> None:
    results = [risk_adjust_scenario(name) for name in SCENARIOS]

    print("Risk-adjusted Voice AI ROI model")
    print("Discounts applied: adoption, latency, compliance, provider risk")
    print()
    print(f"{'Scenario':<14} {'Gross NPV':>14} {'Risk-adjusted NPV':>20} {'Gross ROI':>12} {'Risk-adjusted ROI':>19} {'Risk pass':>10}")
    print("-" * 94)
    for result in results:
        print(
            f"{result.scenario:<14} "
            f"{fmt_money(result.gross_npv):>14} "
            f"{fmt_money(result.risk_adjusted_npv):>20} "
            f"{result.gross_roi_pct:>11.1f}% "
            f"{result.risk_adjusted_roi_pct:>18.1f}% "
            f"{result.composite_pass_factor:>9.1%}"
        )

    print()
    print("Risk discount detail")
    for result in results:
        parts = [f"{factor.name}={factor.risk_discount:.0%}" for factor in result.risk_factors]
        print(f"- {result.scenario}: " + ", ".join(parts))

    print()
    print("Interpretation: risk-adjusted ROI remains positive in all planning scenarios, but the model now explicitly discounts unvalidated adoption, latency, compliance, and provider assumptions. Compliance blockers should still stop launch rather than be treated as only a percentage haircut.")


if __name__ == "__main__":
    main()
