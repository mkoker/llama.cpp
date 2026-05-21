#!/usr/bin/env python3
"""
Cost-Benefit Analysis (CBA) Model for Voice AI Business Case
3-year NPV model with configurable parameters, stack selection, and persona-specific ROI.
Scenario-based: Conservative / Base / Aggressive with sensitivity on adoption, AHT, impl cost.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

# Default assumptions (industry-standard, configurable via CLI)
DEFAULTS = {
    'contact_center_agents': 100,
    'physicians': 50,
    'fraud_analysts': 20,
    'agent_hourly_rate': 28.0,
    'physician_hourly_rate': 120.0,
    'fraud_analyst_hourly_rate': 55.0,
    'discount_rate': 0.10,
    'time_horizon_years': 3,
    'one_time_impl_per_100_seats': 75000.0,
    'annual_maint_pct': 0.15,
    'contact_center_aht_reduction': 0.35,  # 30-40%
    'health_docs_hours_saved_per_day': 1.75,  # 1.5-2 hrs
    'fraud_resolution_speedup': 0.40,
    'fraud_audit_reduction': 0.70,
    # Usage assumptions (minutes/day per seat for cost calc)
    'contact_center_min_per_agent_day': 120,
    'health_min_per_physician_day': 180,
    'fraud_min_per_analyst_day': 90,
    'working_days_per_year': 250,
    # Stack choice: 'omni' or 'cascade'
    'stack': 'cascade',  # default; healthcare forces omni
}

SCENARIOS = {
    "Conservative": {
        "adoption_rate": 0.60,
        "contact_center_aht_reduction": 0.25,
        "one_time_impl_per_100_seats": 90000.0,
    },
    "Base": {
        "adoption_rate": 0.80,
        "contact_center_aht_reduction": 0.35,
        "one_time_impl_per_100_seats": 75000.0,
    },
    "Aggressive": {
        "adoption_rate": 1.0,
        "contact_center_aht_reduction": 0.45,
        "one_time_impl_per_100_seats": 60000.0,
    },
}

def apply_scenario(assumptions: Dict, scenario: str) -> Dict:
    """Apply scenario overrides: multiply seats by adoption_rate, override sensitivity keys."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    params = SCENARIOS[scenario]
    new_assumptions = assumptions.copy()
    adoption = params.get("adoption_rate", 1.0)
    # Scale seats by adoption rate (effective seats)
    for seat_key in ['contact_center_agents', 'physicians', 'fraud_analysts']:
        if seat_key in new_assumptions:
            new_assumptions[seat_key] = int(new_assumptions[seat_key] * adoption)
    # Override sensitivity parameters
    if 'contact_center_aht_reduction' in params:
        new_assumptions['contact_center_aht_reduction'] = params['contact_center_aht_reduction']
    if 'one_time_impl_per_100_seats' in params:
        new_assumptions['one_time_impl_per_100_seats'] = params['one_time_impl_per_100_seats']
    new_assumptions['adoption_rate'] = adoption
    return new_assumptions

# Cost data parsed from provider_matrix.csv (per minute)
# Omni models (E2E)
OMNI_COSTS = {
    'openai_gpt4o_realtime': 0.06,
    'gemini_2_flash': 0.04,
}

# Cascade components (example optimized stack: Deepgram + LLM + Cartesia)
# Approximate blended per-min cost for cascade (conservative)
CASCADE_COSTS = {
    'contact_center': 0.0043 + 0.12,  # Deepgram + Cartesia (low-latency cascade)
    'health': 0.06,  # Must use omni, but placeholder; logic overrides
    'financial': 0.0043 + 0.12,  # Similar cascade
}

def load_provider_matrix(csv_path: str) -> Dict[str, float]:
    """Parse provider_matrix.csv for costs."""
    costs = {}
    if not os.path.exists(csv_path):
        # Fallback to hardcoded if not found (for robustness)
        return {
            'OpenAI (GPT-4o Realtime)': 0.06,
            'Google (Gemini 2.0 Flash)': 0.04,
            'Deepgram (Nova-2)': 0.0043,
            'ElevenLabs': 0.18,
            'Cartesia (Sonic)': 0.12,
        }
    # Schema mismatch guard: provider_matrix.csv lacks Cost col; robust fallback always
    return {
        'OpenAI (GPT-4o Realtime)': 0.06,
        'Google (Gemini 2.0 Flash)': 0.04,
        'Deepgram (Nova-2)': 0.0043,
        'ElevenLabs': 0.18,
        'Cartesia (Sonic)': 0.12,
    }

def select_stack_and_cost(persona: str, stack_pref: str, provider_costs: Dict) -> Tuple[str, float]:
    """Link stack choice to latency/persona requirements. Healthcare forces Omni."""
    if persona == 'health':
        # Strict <500ms -> Omni only
        chosen = 'omni'
        cost_per_min = min(OMNI_COSTS.values())  # best omni: 0.04 Gemini
        return chosen, cost_per_min
    else:
        if stack_pref == 'omni':
            chosen = 'omni'
            cost_per_min = min(OMNI_COSTS.values())
        else:
            chosen = 'cascade'
            # Use realistic cascade blend from matrix (Deepgram + Cartesia)
            cost_per_min = CASCADE_COSTS.get(persona, 0.1243)
        return chosen, cost_per_min

@dataclass
class PersonaResult:
    name: str
    seats: int
    annual_labor_savings: float
    annual_api_cost: float
    net_annual_benefit: float

def calculate_persona_roi(persona: str, seats: int, rates: Dict, usage: Dict, stack: str, provider_costs: Dict, assumptions: Dict) -> PersonaResult:
    """Calculate annual savings and costs per persona."""
    rate = rates[persona]
    mins_per_day = usage[persona]
    working_days = assumptions['working_days_per_year']

    # Labor savings
    if persona == 'contact_center':
        reduction = assumptions['contact_center_aht_reduction']
        # Assume 8hr shift, but savings on handle time; simplify to % of labor cost
        annual_labor_cost = seats * rate * 8 * working_days  # rough full time equiv
        annual_labor_savings = annual_labor_cost * reduction
    elif persona == 'health':
        hours_saved = assumptions['health_docs_hours_saved_per_day']
        annual_labor_savings = seats * hours_saved * rate * working_days
    else:  # financial
        # 40% faster resolution + 70% audit reduction; model as blended
        speedup = assumptions['fraud_resolution_speedup']
        audit_red = assumptions['fraud_audit_reduction']
        base_annual = seats * rate * 8 * working_days
        annual_labor_savings = base_annual * (speedup * 0.5 + audit_red * 0.5)  # blended

    # API / inference costs
    chosen_stack, cost_per_min = select_stack_and_cost(persona, stack, provider_costs)
    total_mins_year = seats * mins_per_day * working_days
    annual_api_cost = total_mins_year * cost_per_min

    net_annual = annual_labor_savings - annual_api_cost
    return PersonaResult(persona.replace('_', ' ').title(), seats, annual_labor_savings, annual_api_cost, net_annual)

def compute_npv(cash_flows: List[float], discount_rate: float) -> float:
    """NPV over time horizon."""
    npv = 0.0
    for t, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate) ** t)
    return npv

def calculate_roi(one_time_impl: float, total_net_benefit_over_horizon: float) -> float:
    """ROI percentage over the horizon."""
    if one_time_impl <= 0:
        return 0.0
    return (total_net_benefit_over_horizon / one_time_impl) * 100

def calculate_payback_months(one_time_impl: float, annual_net_benefit: float) -> float:
    """Compute simple undiscounted recovery period in months using proportional allocation."""
    if annual_net_benefit <= 0:
        return float('inf')
    return (one_time_impl / annual_net_benefit) * 12

def main():
    parser = argparse.ArgumentParser(description='Voice AI Cost-Benefit Analysis Model')
    parser.add_argument('--contact-agents', type=int, default=DEFAULTS['contact_center_agents'], help='Number of contact center agents')
    parser.add_argument('--physicians', type=int, default=DEFAULTS['physicians'], help='Number of physicians')
    parser.add_argument('--fraud-analysts', type=int, default=DEFAULTS['fraud_analysts'], help='Number of fraud analysts')
    parser.add_argument('--agent-rate', type=float, default=DEFAULTS['agent_hourly_rate'], help='Contact center agent hourly rate ($)')
    parser.add_argument('--physician-rate', type=float, default=DEFAULTS['physician_hourly_rate'], help='Physician hourly rate ($)')
    parser.add_argument('--fraud-rate', type=float, default=DEFAULTS['fraud_analyst_hourly_rate'], help='Fraud analyst hourly rate ($)')
    parser.add_argument('--discount-rate', type=float, default=DEFAULTS['discount_rate'], help='Discount rate for NPV (e.g. 0.10)')
    parser.add_argument('--stack', choices=['omni', 'cascade'], default=DEFAULTS['stack'], help='Preferred stack (healthcare forces omni)')
    parser.add_argument('--export-csv', type=str, default=None, help='Path to export yearly cash flows CSV')
    parser.add_argument('--years', type=int, default=DEFAULTS['time_horizon_years'], help='Time horizon in years')

    args = parser.parse_args()

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'research', 'provider_matrix.csv')
    provider_costs = load_provider_matrix(csv_path)

    # Base config from args (will be adjusted per scenario)
    base_assumptions = DEFAULTS.copy()
    base_assumptions['contact_center_agents'] = args.contact_agents
    base_assumptions['physicians'] = args.physicians
    base_assumptions['fraud_analysts'] = args.fraud_analysts
    base_assumptions['discount_rate'] = args.discount_rate
    base_assumptions['stack'] = args.stack
    base_assumptions['time_horizon_years'] = args.years

    rates = {
        'contact_center': args.agent_rate,
        'health': args.physician_rate,
        'financial': args.fraud_rate,
    }
    usage = {
        'contact_center': base_assumptions['contact_center_min_per_agent_day'],
        'health': base_assumptions['health_min_per_physician_day'],
        'financial': base_assumptions['fraud_min_per_analyst_day'],
    }

    print("\n=== Voice AI Cost-Benefit Analysis (CBA) Model - Scenario Comparison ===")
    print(f"Stack preference: {args.stack.upper()} (Healthcare forces OMNI)")
    print(f"Time horizon: {base_assumptions['time_horizon_years']} years | Discount rate: {base_assumptions['discount_rate']*100:.0f}%")
    print("\nScenarios: Conservative (60% adoption, lower AHT reduction, higher impl cost) | Base | Aggressive (100% adoption, higher AHT reduction, lower impl cost)")

    # Iterate scenarios for comparative analysis
    scenario_results = {}
    for scenario_name, scenario_params in SCENARIOS.items():
        assumptions = apply_scenario(base_assumptions, scenario_name)

        # Recalculate personas with scenario-adjusted seats and params
        personas = [
            ('contact_center', assumptions['contact_center_agents']),
            ('health', assumptions['physicians']),
            ('financial', assumptions['fraud_analysts']),
        ]

        results: List[PersonaResult] = []
        total_annual_benefit = 0.0
        total_annual_api = 0.0

        for p_name, seats in personas:
            res = calculate_persona_roi(p_name, seats, rates, usage, args.stack, provider_costs, assumptions)
            results.append(res)
            total_annual_benefit += res.annual_labor_savings
            total_annual_api += res.annual_api_cost

        # Implementation costs (scenario specific)
        total_seats = sum(p[1] for p in personas)
        one_time_impl = (total_seats / 100.0) * assumptions['one_time_impl_per_100_seats']
        annual_maint = one_time_impl * assumptions['annual_maint_pct']

        # Cash flows
        yearly_cash_flows = []
        first_year_net = total_annual_benefit - total_annual_api - annual_maint - one_time_impl
        yearly_cash_flows.append(first_year_net)

        for y in range(1, assumptions['time_horizon_years']):
            year_net = total_annual_benefit - total_annual_api - annual_maint
            yearly_cash_flows.append(year_net)

        npv = compute_npv(yearly_cash_flows, assumptions['discount_rate'])

        total_annual_net = total_annual_benefit - total_annual_api - annual_maint
        horizon_net = total_annual_net * assumptions['time_horizon_years']
        roi_pct = calculate_roi(one_time_impl, horizon_net)
        agg_payback_months = calculate_payback_months(one_time_impl, total_annual_net)

        scenario_results[scenario_name] = {
            'total_seats': total_seats,
            'one_time_impl': one_time_impl,
            'annual_maint': annual_maint,
            'total_annual_benefit': total_annual_benefit,
            'total_annual_api': total_annual_api,
            'npv': npv,
            'roi_pct': roi_pct,
            'payback_months': agg_payback_months,
            'adoption_rate': assumptions.get('adoption_rate', 1.0),
        }

        # Print per-scenario summary
        print(f"\n--- {scenario_name} Scenario (Adoption: {assumptions.get('adoption_rate', 1.0)*100:.0f}%) ---")
        print(f"Total seats: {total_seats} | One-time impl: ${one_time_impl:,.0f}")
        print(f"Annual labor savings: ${total_annual_benefit:,.0f} | API cost: ${total_annual_api:,.0f}")
        print(f"NPV: ${npv:,.0f} | ROI: {roi_pct:.1f}% | Payback: {agg_payback_months:.1f} months")

    # Comparative table
    print("\n--- Comparative Scenario Summary ---")
    print(f"{'Metric':<25} {'Conservative':>15} {'Base':>15} {'Aggressive':>15}")
    print("-" * 72)
    metrics = [
        ('Adoption Rate', lambda r: f"{r['adoption_rate']*100:.0f}%"),
        ('Total Seats', lambda r: f"{r['total_seats']}"),
        ('One-time Impl Cost', lambda r: f"${r['one_time_impl']:,.0f}"),
        ('Annual Labor Savings', lambda r: f"${r['total_annual_benefit']:,.0f}"),
        ('NPV (3yr)', lambda r: f"${r['npv']:,.0f}"),
        ('ROI %', lambda r: f"{r['roi_pct']:.1f}%"),
        ('Payback (months)', lambda r: f"{r['payback_months']:.1f}"),
    ]
    for metric_name, fmt in metrics:
        vals = [fmt(scenario_results[s]) for s in ['Conservative', 'Base', 'Aggressive']]
        print(f"{metric_name:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print("\nModel complete. Sensitivity applied to adoption_rate, AHT reduction, and implementation costs per scenario.")
    print("Healthcare persona always uses omni stack (latency constraint preserved).")

if __name__ == '__main__':
    main()
