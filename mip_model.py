import pulp
import numpy as np

def solve_mix_optimization(
        demand_units, available_hours, production_rate_base,
        cement_price_mwk_kg,
        virgin_coarse_6_20_price_mwk_kg,
        virgin_coarse_20_40_price_mwk_kg,
        virgin_fine_sand_price_mwk_kg,
        recycled_coarse_price_mwk_kg,
        water_price_mwk_l,
        electricity_price_mwk_kwh,
        cement_available_kg,
        virgin_coarse_6_20_available_kg,
        virgin_coarse_20_40_available_kg,
        virgin_fine_sand_available_kg,
        recycled_coarse_available_kg,
        budget_mwk,
        target_strength_min,
        max_wc_ratio=None,
        min_cement_kg_per_block=1.2,
        target_absorption_max=15.0,
        absorption_intercept=12.0,
        absorption_coef_recycled=0.1,
        absorption_curing_coef=2.0,
        curing_score=0.6,
        max_recycled_coarse_prop=0.2,
        max_coarse_20_40_prop=0.1,
        min_fine_proportion=0.30,
        ef_cement_kgco2_kg=0.82,
        ef_electricity_kgco2_kwh=0.12,
        w_cost=0.4416, w_resource=0.4391, w_env=0.1193,
        machine_power_kw=15.0,
        block_volume_m3=0.016,
        concrete_density_kg_m3=2200,
        norm_bounds=None,
        blocks_per_batch=1200,
        ignore_sustainability=False
):
    if ignore_sustainability:
        w_env = 0.0
        total_cr = w_cost + w_resource
        if total_cr > 0:
            w_cost /= total_cr
            w_resource /= total_cr
        else:
            w_cost = w_resource = 0.5
        force_no_recycled = True
    else:
        force_no_recycled = False

    if max_wc_ratio is None:
        max_wc_ratio = 15.0 / target_strength_min
        max_wc_ratio = np.clip(max_wc_ratio, 0.4, 0.7)

    if norm_bounds is None:
        norm_bounds = {
            'cost': (0, budget_mwk * 1.2),
            'resource': (0.4, 1.0),
            'env': (0, 30_000)
        }

    model = pulp.LpProblem("Block_Mix_Optimisation", pulp.LpMaximize)

    cement = pulp.LpVariable("cement", lowBound=min_cement_kg_per_block, upBound=5.0)
    water = pulp.LpVariable("water", lowBound=0.8, upBound=3.0)
    coarse_6_20 = pulp.LpVariable("coarse_6_20", lowBound=0, upBound=20.0)
    coarse_20_40 = pulp.LpVariable("coarse_20_40", lowBound=0, upBound=5.0)
    fine_sand = pulp.LpVariable("fine_sand", lowBound=0, upBound=15.0)
    recycled_coarse = pulp.LpVariable("recycled_coarse", lowBound=0, upBound=10.0)

    if force_no_recycled:
        model += recycled_coarse == 0

    total_coarse = coarse_6_20 + coarse_20_40 + recycled_coarse
    total_agg = total_coarse + fine_sand
    mass_per_block = concrete_density_kg_m3 * block_volume_m3

    model += cement + water + total_agg == mass_per_block
    model += water <= max_wc_ratio * cement

    if not force_no_recycled and max_recycled_coarse_prop > 0:
        model += recycled_coarse <= max_recycled_coarse_prop * total_coarse

    model += coarse_20_40 <= max_coarse_20_40_prop * total_coarse
    model += fine_sand >= min_fine_proportion * total_agg

    lhs_abs = (absorption_intercept - absorption_curing_coef * curing_score - target_absorption_max) * total_coarse + absorption_coef_recycled * recycled_coarse
    model += lhs_abs <= 0

    total_hours = demand_units / production_rate_base
    model += total_hours <= available_hours

    model += cement * demand_units <= cement_available_kg
    model += coarse_6_20 * demand_units <= virgin_coarse_6_20_available_kg
    model += coarse_20_40 * demand_units <= virgin_coarse_20_40_available_kg
    model += fine_sand * demand_units <= virgin_fine_sand_available_kg
    if not force_no_recycled:
        model += recycled_coarse * demand_units <= recycled_coarse_available_kg

    mat_cost_per_block = (cement * cement_price_mwk_kg +
                          coarse_6_20 * virgin_coarse_6_20_price_mwk_kg +
                          coarse_20_40 * virgin_coarse_20_40_price_mwk_kg +
                          fine_sand * virgin_fine_sand_price_mwk_kg +
                          recycled_coarse * recycled_coarse_price_mwk_kg +
                          water * water_price_mwk_l)
    total_mat_cost = mat_cost_per_block * demand_units
    electricity_cost = total_hours * machine_power_kw * electricity_price_mwk_kwh
    total_cost = total_mat_cost + electricity_cost
    model += total_cost <= budget_mwk

    total_carbon = (cement * demand_units * ef_cement_kgco2_kg +
                    total_hours * machine_power_kw * ef_electricity_kgco2_kwh)

    availability = total_hours / available_hours if available_hours > 0 else 1.0
    performance = 1.0
    oee = availability * performance * 1.0
    resource_efficiency = (oee + 1.0) / 2

    cost_norm = (total_cost - norm_bounds['cost'][0]) / max(1e-6, norm_bounds['cost'][1] - norm_bounds['cost'][0])
    res_norm = (resource_efficiency - norm_bounds['resource'][0]) / max(1e-6, norm_bounds['resource'][1] - norm_bounds['resource'][0])
    env_norm = (total_carbon - norm_bounds['env'][0]) / max(1e-6, norm_bounds['env'][1] - norm_bounds['env'][0])

    model += (w_resource * res_norm - w_cost * cost_norm - w_env * env_norm)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    model.solve(solver)

    if pulp.LpStatus[model.status] != 'Optimal':
        return {'status': 'Infeasible'}

    cement_opt = cement.varValue
    water_opt = water.varValue
    coarse_6_20_opt = coarse_6_20.varValue
    coarse_20_40_opt = coarse_20_40.varValue
    fine_sand_opt = fine_sand.varValue
    recycled_opt = recycled_coarse.varValue

    total_coarse_val = coarse_6_20_opt + coarse_20_40_opt + recycled_opt
    recycled_prop = recycled_opt / total_coarse_val if total_coarse_val > 0 else 0
    wc_ratio = water_opt / cement_opt if cement_opt > 0 else 999

    strength_est = 2.5 * cement_opt + 5.0 / max(0.4, wc_ratio)
    strength_est = min(15.0, strength_est)
    absorption_actual = absorption_intercept + absorption_coef_recycled * recycled_prop - absorption_curing_coef * curing_score

    total_hours_opt = demand_units / production_rate_base
    full_batches = (demand_units + blocks_per_batch - 1) // blocks_per_batch
    total_blocks_produced = full_batches * blocks_per_batch

    # Recompute cost using optimal values
    mat_cost_per_block_opt = (cement_opt * cement_price_mwk_kg +
                              coarse_6_20_opt * virgin_coarse_6_20_price_mwk_kg +
                              coarse_20_40_opt * virgin_coarse_20_40_price_mwk_kg +
                              fine_sand_opt * virgin_fine_sand_price_mwk_kg +
                              recycled_opt * recycled_coarse_price_mwk_kg +
                              water_opt * water_price_mwk_l)
    total_mat_cost_opt = mat_cost_per_block_opt * demand_units
    electricity_cost_opt = total_hours_opt * machine_power_kw * electricity_price_mwk_kwh
    total_cost_opt = total_mat_cost_opt + electricity_cost_opt
    total_carbon_opt = (cement_opt * demand_units * ef_cement_kgco2_kg +
                        total_hours_opt * machine_power_kw * ef_electricity_kgco2_kwh)

    actual_rate = total_blocks_produced / total_hours_opt if total_hours_opt > 0 else 0
    performance_actual = actual_rate / production_rate_base
    availability_actual = total_hours_opt / available_hours if available_hours > 0 else 1.0
    oee_opt = availability_actual * performance_actual * 1.0
    resource_efficiency_opt = (oee_opt + 1.0) / 2

    # Compute constraint slack (how close we are to limits)
    slack = {
        'budget': budget_mwk - total_cost_opt,
        'cement_avail': cement_available_kg - cement_opt * demand_units,
        'hours': available_hours - total_hours_opt,
        'recycled_avail': recycled_coarse_available_kg - recycled_opt * demand_units if not force_no_recycled else None,
    }

    return {
        'status': 'Optimal',
        'cement_kg': cement_opt,
        'water_kg': water_opt,
        'coarse_6_20_kg': coarse_6_20_opt,
        'coarse_20_40_kg': coarse_20_40_opt,
        'fine_sand_kg': fine_sand_opt,
        'recycled_coarse_kg': recycled_opt,
        'recycled_coarse_prop': recycled_prop,
        'demand_units': demand_units,
        'total_blocks': total_blocks_produced,
        'batches': full_batches,
        'blocks_per_batch': blocks_per_batch,
        'total_cost_mwk': total_cost_opt,
        'total_carbon_kgco2': total_carbon_opt,
        'oee': oee_opt,
        'resource_efficiency': resource_efficiency_opt,
        'strength_estimate_mpa': strength_est,
        'absorption_percent': absorption_actual,
        'production_time_hours': total_hours_opt,
        'wc_ratio': wc_ratio,
        'slack': slack,
        'ignore_sustainability': ignore_sustainability
    }