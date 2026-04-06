import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mip_model import solve_mix_optimization

st.set_page_config(page_title="BlockMix Optimiser", layout="wide", page_icon="🧱")

# Helper function – renamed parameters to avoid shadowing
def run_optimisation(input_params, force_flag):
    if force_flag:
        with st.spinner("🔍 Solving MIP model (force recompute)... (30-60 seconds)"):
            return solve_mix_optimization(**input_params)
    else:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached(p):
            return solve_mix_optimization(**p)
        with st.spinner("🔍 Solving MIP model... (30-60 seconds)"):
            return _cached(tuple(sorted(input_params.items())))

def export_results_csv(result_data):
    df = pd.DataFrame([result_data])
    for col in df.columns:
        if isinstance(df[col].iloc[0], (list, dict)):
            df[col] = df[col].astype(str)
    return df.to_csv(index=False).encode('utf-8')

# ------------------ Sidebar inputs ------------------
with st.sidebar:
    st.header("📦 Order & Production")
    demand = st.number_input("Total blocks required", min_value=1, value=10000, step=500)
    available_hours = st.number_input("Available machine hours (monthly)", value=160.0, step=10.0)
    production_rate = st.number_input("Production rate (blocks/hour)", value=1000, step=100)
    blocks_per_batch = st.number_input("Blocks per batch", value=1200, step=100)

    st.header("💰 Material Costs (MWK per unit)")
    cement_price = st.number_input("Cement (MWK/kg)", value=576.0, step=10.0)
    coarse_6_20_price = st.number_input("Virgin coarse 6‑20 mm (MWK/kg)", value=31.5, step=1.0)
    coarse_20_40_price = st.number_input("Virgin coarse 20‑40 mm (MWK/kg)", value=31.5, step=1.0)
    fine_sand_price = st.number_input("Virgin fine sand (MWK/kg)", value=31.5, step=1.0)
    recycled_price = st.number_input("Recycled coarse aggregate (MWK/kg)", value=20.0, step=1.0)
    water_price = st.number_input("Water (MWK/L)", value=9.0, step=0.5)
    elec_price = st.number_input("Electricity (MWK/kWh)", value=207.0, step=10.0)

    st.header("📦 Raw Material Availability (kg)")
    cement_avail = st.number_input("Cement available", value=100000, step=5000, min_value=0)
    coarse_6_20_avail = st.number_input("Coarse 6‑20 mm available", value=200000, step=10000, min_value=0)
    coarse_20_40_avail = st.number_input("Coarse 20‑40 mm available", value=50000, step=5000, min_value=0)
    fine_sand_avail = st.number_input("Fine sand available", value=150000, step=10000, min_value=0)
    recycled_avail = st.number_input("Recycled coarse available", value=50000, step=5000, min_value=0)

    st.header("🔧 Technical Parameters")
    target_strength = st.number_input("Min compressive strength (MPa)", value=7.0, step=0.5)
    target_absorption = st.number_input("Max water absorption (%)", value=15.0, step=1.0)
    curing_score = st.slider("Curing efficiency (0‑1)", 0.0, 1.0, 0.6, 0.05)
    min_cement = st.number_input("Min cement per block (kg)", value=1.2, step=0.1)
    max_recycled_prop = st.slider("Max recycled proportion of coarse", 0.0, 0.5, 0.2, 0.05)
    max_coarse_20_40_prop = st.slider("Max proportion of 20‑40 mm in coarse", 0.0, 0.3, 0.1, 0.05)
    min_fine_prop = st.slider("Min fine proportion of total aggregate", 0.2, 0.5, 0.30, 0.05)

    st.header("⚖️ DEMATEL Weights")
    w_cost = st.number_input("Cost weight", value=0.4416, step=0.01, min_value=0.0, max_value=1.0)
    w_resource = st.number_input("Resource efficiency weight", value=0.4391, step=0.01, min_value=0.0, max_value=1.0)
    w_env = st.number_input("Environmental weight", value=0.1193, step=0.01, min_value=0.0, max_value=1.0)

    st.header("🌍 Sustainability Override")
    ignore_sustainability = st.checkbox("Ignore sustainability (no recycled, renormalise weights)")

    st.header("💰 Budget & Machine")
    budget = st.number_input("Total budget (MWK)", value=100_000_000, step=5_000_000, min_value=0)
    machine_power = st.number_input("Machine power (kW)", value=15.0, step=1.0)
    block_volume = st.number_input("Block volume (m³)", value=0.016, step=0.001)
    concrete_density = st.number_input("Concrete density (kg/m³)", value=2200, step=50)

    st.header("📊 Normalisation Bounds (optional)")
    use_custom_bounds = st.checkbox("Use custom bounds", value=False)
    norm_bounds = None
    if use_custom_bounds:
        col1, col2 = st.columns(2)
        with col1:
            cost_min = st.number_input("Cost min (MWK)", value=0)
            res_min = st.number_input("Resource min", value=0.4, step=0.05)
            env_min = st.number_input("Env min (kg CO₂e)", value=0)
        with col2:
            cost_max = st.number_input("Cost max (MWK)", value=budget*1.2)
            res_max = st.number_input("Resource max", value=1.0, step=0.05)
            env_max = st.number_input("Env max (kg CO₂e)", value=30_000)
        norm_bounds = {'cost': (cost_min, cost_max), 'resource': (res_min, res_max), 'env': (env_min, env_max)}

    st.header("⚙️ Optimisation Settings")
    force_recompute = st.checkbox("Force recompute (ignore cache)", value=False, help="Use this if results don't seem to update after changing inputs.")

# ------------------ Main area ------------------
st.title("🧱 Concrete Block Mix Design Optimiser (MIP + DEMATEL)")
st.markdown("Optimise cost, resource efficiency, and CO₂e under real factory constraints.")

if st.button("🚀 Run Optimisation", type="primary", use_container_width=True):
    # Build parameter dictionary
    params_dict = {
        'demand_units': demand,
        'available_hours': available_hours,
        'production_rate_base': production_rate,
        'cement_price_mwk_kg': cement_price,
        'virgin_coarse_6_20_price_mwk_kg': coarse_6_20_price,
        'virgin_coarse_20_40_price_mwk_kg': coarse_20_40_price,
        'virgin_fine_sand_price_mwk_kg': fine_sand_price,
        'recycled_coarse_price_mwk_kg': recycled_price,
        'water_price_mwk_l': water_price,
        'electricity_price_mwk_kwh': elec_price,
        'cement_available_kg': cement_avail,
        'virgin_coarse_6_20_available_kg': coarse_6_20_avail,
        'virgin_coarse_20_40_available_kg': coarse_20_40_avail,
        'virgin_fine_sand_available_kg': fine_sand_avail,
        'recycled_coarse_available_kg': recycled_avail,
        'budget_mwk': budget,
        'target_strength_min': target_strength,
        'max_wc_ratio': None,
        'min_cement_kg_per_block': min_cement,
        'target_absorption_max': target_absorption,
        'absorption_intercept': 12.0,
        'absorption_coef_recycled': 0.1,
        'absorption_curing_coef': 2.0,
        'curing_score': curing_score,
        'max_recycled_coarse_prop': max_recycled_prop,
        'max_coarse_20_40_prop': max_coarse_20_40_prop,
        'min_fine_proportion': min_fine_prop,
        'ef_cement_kgco2_kg': 0.82,
        'ef_electricity_kgco2_kwh': 0.12,
        'w_cost': w_cost,
        'w_resource': w_resource,
        'w_env': w_env,
        'machine_power_kw': machine_power,
        'block_volume_m3': block_volume,
        'concrete_density_kg_m3': concrete_density,
        'norm_bounds': norm_bounds,
        'blocks_per_batch': blocks_per_batch,
        'ignore_sustainability': ignore_sustainability
    }

    try:
        # Call the helper function with renamed parameters
        optimisation_result = run_optimisation(params_dict, force_recompute)
    except Exception as e:
        st.error(f"❌ Optimisation engine crashed: {str(e)}")
        st.stop()

    if optimisation_result['status'] == 'Optimal':
        st.success("✅ Optimal solution found! Ready for production.")
        st.balloons()

        st.session_state['last_result'] = optimisation_result

        # Show key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cement per block", f"{optimisation_result['cement_kg']:.2f} kg")
            st.metric("Water per block", f"{optimisation_result['water_kg']:.2f} L")
            st.metric("W/C ratio", f"{optimisation_result['wc_ratio']:.2f}")
        with col2:
            st.metric("Virgin coarse 6‑20", f"{optimisation_result['coarse_6_20_kg']:.2f} kg")
            st.metric("Virgin coarse 20‑40", f"{optimisation_result['coarse_20_40_kg']:.2f} kg")
            st.metric("Virgin fine sand", f"{optimisation_result['fine_sand_kg']:.2f} kg")
        with col3:
            st.metric("Recycled coarse", f"{optimisation_result['recycled_coarse_kg']:.2f} kg")
            st.metric("Recycled proportion", f"{optimisation_result['recycled_coarse_prop']:.1%}")
            st.metric("Total cost", f"MWK {optimisation_result['total_cost_mwk']:,.0f}")
        with col4:
            st.metric("Total CO₂e", f"{optimisation_result['total_carbon_kgco2']:,.0f} kg")
            st.metric("OEE", f"{optimisation_result['oee']:.1%}")
            st.metric("Resource efficiency", f"{optimisation_result['resource_efficiency']:.1%}")

        st.info(
            f"**Production time:** {optimisation_result['production_time_hours']:.1f} hours | "
            f"**Blocks produced (with safety stock):** {int(optimisation_result['total_blocks'])} | "
            f"**Batches:** {optimisation_result['batches']} × {optimisation_result['blocks_per_batch']} blocks/batch"
        )
        st.info(
            f"**Predicted strength:** {optimisation_result['strength_estimate_mpa']:.1f} MPa (≥{target_strength}) | "
            f"**Water absorption:** {optimisation_result['absorption_percent']:.1f}% (≤{target_absorption})"
        )

        # Show constraint slack
        with st.expander("🔍 Constraint slack (how close to limits)"):
            slack = optimisation_result['slack']
            st.write(f"**Budget remaining:** MWK {slack['budget']:,.0f}")
            st.write(f"**Cement remaining:** {slack['cement_avail']:,.0f} kg")
            st.write(f"**Production hours remaining:** {slack['hours']:.1f} hrs")
            if slack.get('recycled_avail') is not None:
                st.write(f"**Recycled aggregate remaining:** {slack['recycled_avail']:,.0f} kg")
            st.caption("If any slack is large (>10% of limit), that constraint is not binding. Changing that availability won't affect the solution until it becomes binding.")

        # Mix composition chart
        fig, ax = plt.subplots()
        materials = ['Cement', 'Water', '6‑20 mm', '20‑40 mm', 'Fine sand', 'Recycled']
        values = [optimisation_result['cement_kg'], optimisation_result['water_kg'],
                  optimisation_result['coarse_6_20_kg'], optimisation_result['coarse_20_40_kg'],
                  optimisation_result['fine_sand_kg'], optimisation_result['recycled_coarse_kg']]
        ax.barh(materials, values, color=['#2c3e50', '#3498db', '#95a5a6', '#7f8c8d', '#f1c40f', '#27ae60'])
        ax.set_xlabel('kg per block')
        ax.set_title('Mix composition per block')
        st.pyplot(fig)

        csv_data = export_results_csv(optimisation_result)
        st.download_button("📥 Download full results (CSV)", data=csv_data, file_name="optimal_mix.csv", mime="text/csv")

    else:
        st.error("❌ No feasible mix found under current constraints.")
        st.markdown("**Suggestions:**")
        st.markdown("- Increase budget or raw material availabilities")
        st.markdown("- Lower target strength or increase maximum water absorption")
        st.markdown("- Reduce minimum cement per block or increase max recycled proportion")
        if ignore_sustainability:
            st.markdown("- Try unchecking 'Ignore sustainability' – recycled aggregate might help feasibility")