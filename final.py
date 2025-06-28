import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from climada.util.api_client import Client
from climada.entity import Entity, ImpactFuncSet, ImpfTropCyclone, MeasureSet, Measure, DiscRates
from climada.engine.cost_benefit import CostBenefit, risk_aai_agg,risk_rp_100,risk_rp_250
from climada.engine import ImpactCalc

# ---- PAGE CONFIG ----
st.set_page_config(page_title="UBUNTU CLIMATE FINANCE  TOOL", page_icon="🌪️", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f9fafc; }
        .stButton>button { background-color:#4b8bbe; color:white; border-radius:10px; }
        .stTabs [data-baseweb="tab-list"] { background-color: #eef3f8; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://climada.ethz.ch/_static/climada_logo.png", width=120)
st.sidebar.title("🛠️ Configuration Panel")

# Sidebar Inputs
country = st.sidebar.selectbox("🌍 Country", ["Haiti", "Guinea", "Kenya", "United States", "Japan", "Bangladesh"])
future_year = st.sidebar.selectbox("📆 Future Year", [2040, 2060, 2080])
growth_rate = st.sidebar.slider("📈 Growth Rate (%)", 0, 10, 2)
haz_scenarios = {"Historical": "historical", "RCP2.6": "rcp26", "RCP4.5": "rcp45", "RCP6.0": "rcp60"}
climate_label = st.sidebar.selectbox("🌡️ Climate Scenario", list(haz_scenarios.keys()))
climate_scenario = haz_scenarios[climate_label]
nb_tracks = 10
st.sidebar.warning("⚠️ If you change the climate scenario, make sure to reload the hazard data before  clicking '🚀 Run Full CBA Analysis' ")


# Session state init
for key in ["exp_present", "exp_future", "haz_present", "haz_future", "impf_set", "meas_set", "entity_present", "entity_future", "cb"]:
    if key not in st.session_state:
        st.session_state[key] = None

# PART 1
st.title("🌀 UBUNTU CLIMATE FINANCE  TOOL")
with st.expander("📁 Part 1: Data Gathering & Setup", expanded=True):

    st.markdown("#### 📊 Exposure Data")
    if st.button("⬇️ Download Exposure"):
        client = Client()
        try:
            exp = client.get_litpop(country)
            future_exp = exp.copy()
            future_exp.ref_year = future_year
            growth = (1 + growth_rate / 100) ** (future_year - exp.ref_year)
            future_exp.gdf["value"] *= growth
            for e in [exp, future_exp]:
                e.gdf["impf_TC"] = 1
            st.session_state.exp_present = exp
            st.session_state.exp_future = future_exp
            st.success("✅ Exposure loaded.")
            st.dataframe(exp.gdf.head())
        except Exception as e:
            st.error(f"❌ Failed to load exposure: {e}")
st.markdown("#### 🌪️ Download Hazard Scenario")

if st.button("⬇️ Download Hazard from API"):
    with st.spinner("Requesting hazard data..."):
        try:
            client = Client()
            properties = {
                "country_name": country,
                "climate_scenario": climate_scenario,
                "nb_synth_tracks": str(nb_tracks),
            }
            if climate_scenario != "historical":
                properties["ref_year"] = str(future_year)

            hazard = client.get_hazard("tropical_cyclone", properties=properties)

            # Set both present and future hazard, even if the same
            st.session_state.haz_present = hazard
            st.session_state.haz_future = hazard  # ensures downstream code always works

            st.success(f"✅ Hazard data loaded for {country} ({climate_label})")
        except Exception as e:
            st.error(f"❌ Error loading hazard data: {e}")

st.markdown("#### 📍 Assign Centroids to Exposure")

if st.button("📌 Assign Centroids"):
    exp = st.session_state.get("exp_present")
    haz = st.session_state.get("haz_present")

    if exp is None or haz is None:
        st.warning("⚠️ Please make sure both exposure and hazard data are loaded first.")
    else:
        try:
            exp.assign_centroids(haz, distance="approx")
            st.session_state.exp_present = exp  # update with centroids
            st.success("✅ Centroids assigned successfully.")
        except Exception as e:
            st.error(f"❌ Failed to assign centroids: {e}")
st.markdown("#### 📈 Define Impact Function")

if st.button("🧩 Load Impact Function"):
    try:
        impf = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet([impf])
        impf_set.check()
        st.session_state.impf_set = impf_set
        st.success("✅ Impact function loaded and stored.")
        st.info("ℹ️ The curve will be visualized in the '📈 Visualizations' tab.")
    except Exception as e:
        st.error(f"❌ Failed to load impact function: {e}")
st.markdown("#### 🛡️ Define Adaptation Measures")

# Sidebar selection
st.sidebar.subheader("🛠️ Adaptation Measure Selection")

adaptation_choice = st.sidebar.radio(
    "Choose Adaptation Type:",
    ["Standard1", "Standard2", "Custom"]
)
st.session_state["adaptation_mode"] = adaptation_choice


# Define Standard Measures
Standard1 = Measure(
    haz_type="TC",
    name="Wind Speed Reduction (5 m/s)",
    color_rgb=np.array([0.8, 0.1, 0.1]),
    cost=5_000_000_000,
    hazard_inten_imp=(1, -5),
    risk_transf_cover=0,
)

Standard2 = Measure(
    haz_type="TC",
    name="10% Assets Protected",
    color_rgb=np.array([0.1, 0.1, 0.8]),
    cost=220_000_000,
    paa_impact=(1, -0.10),
)

selected_measures = []

# Logic for each option
if adaptation_choice == "Standard1":
    selected_measures = [Standard1]
elif adaptation_choice == "Standard2":
    selected_measures = [Standard2]
elif adaptation_choice == "Custom":
    wind_reduction = st.sidebar.number_input("Wind Speed Reduction (m/s)", min_value=0.0, value=5.0, step=0.5)
    paa_protected = st.sidebar.number_input("Percent of Assets Protected (%)", min_value=0.0, value=10.0, step=1.0)
    custom_cost = st.sidebar.number_input("Measure Cost (USD)", min_value=0, value=1_000_000_000, step=1_000_000)

    custom_measure = Measure(
        haz_type="TC",
        name=f"Custom: {wind_reduction} m/s & {paa_protected}%",
        color_rgb=np.array([0.1, 0.7, 0.1]),
        cost=custom_cost,
        hazard_inten_imp=(1, -float(wind_reduction)),
        paa_impact=(1, -float(paa_protected) / 100.0),
    )
    selected_measures = [custom_measure]

# Build MeasureSet
if selected_measures:
    try:
        meas_set = MeasureSet(measure_list=selected_measures)
        meas_set.check()
        st.session_state.meas_set = meas_set
        st.success(f"✅ Adaptation Measure Loaded: {selected_measures[0].name}")
    except Exception as e:
        st.error(f"❌ Failed to create MeasureSet:\n\n{e}")
st.session_state["meas_set"] = meas_set


st.markdown("#### 💰 Set Discount Rate")

# Sidebar selection for discount rate option
st.sidebar.subheader("📉 Discount Rate Settings")
discount_choice = st.sidebar.radio(
    "Select Discount Rate Type:",
    ["No Discount (0%)", "Stern Review (1.4%)", "Custom (%)"]
)

# Assign actual discount value
if discount_choice == "No Discount (0%)":
    discount_rate = 0.0
elif discount_choice == "Stern Review (1.4%)":
    discount_rate = 0.014
else:
    discount_rate = st.sidebar.number_input(
        "Enter custom discount rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1
    ) / 100.0

# Store in session state
st.session_state.discount_rate = discount_rate
st.success(f"✅ Discount rate set to {discount_rate*100:.2f}%")
st.markdown("#### 🧱 Create Entity Objects")

if st.button("🏗️ Build Entity Objects"):
    exp_present = st.session_state.get("exp_present")
    exp_future = st.session_state.get("exp_future")
    impf_set = st.session_state.get("impf_set")
    meas_set = st.session_state.get("meas_set")
    discount = st.session_state.get("discount_rate")

    # Safety check
    if not all([exp_present, exp_future, impf_set, meas_set, discount is not None]):
        st.error("❌ Missing exposure, impact function, adaptation measures, or discount rate.")
    else:
        try:
            years = np.arange(exp_present.ref_year, exp_future.ref_year + 1)
            rates = np.ones(len(years)) * discount

            entity_present = Entity(
                exposures=exp_present,
                impact_func_set=impf_set,
                measure_set=meas_set,
                disc_rates=DiscRates(years, rates)
            )

            entity_future = Entity(
                exposures=exp_future,
                impact_func_set=impf_set,
                measure_set=meas_set,
                disc_rates=DiscRates(years, rates)
            )

            st.session_state.entity_present = entity_present
            st.session_state.entity_future = entity_future

            st.success("✅ Entity objects created for present and future.")
        except Exception as e:
            st.error(f"❌ Failed to create entity objects:\n\n{e}")



# -----------------------------------------------
# 📊 Part 2: Impact & Cost-Benefit Analysis
# -----------------------------------------------
with st.expander("⚙️ Part 2: Impact & CBA Analysis", expanded=True):
    exp_present = st.session_state.get("exp_present", None)
    exp_future = st.session_state.get("exp_future", None)
    haz_present = st.session_state.get("haz_present", None)
    haz_future = st.session_state.get("haz_future", None)
    impf_set = st.session_state.get("impf_set", None)
    discount_rate = st.session_state.get("discount_rate", None)
    adaptation_mode = st.session_state.get("adaptation_mode", None)

 
# --- Run Analysis Block ---
with st.expander("⚙️ Part 2: Impact & CBA Analysis", expanded=True):
    exp_present = st.session_state.get("exp_present")
    exp_future = st.session_state.get("exp_future")
    haz_present = st.session_state.get("haz_present")
    haz_future = st.session_state.get("haz_future")
    impf_set = st.session_state.get("impf_set")
    discount_rate = st.session_state.get("discount_rate")


    if st.button("🚀 Run CBA Analysis"):
        if not all([exp_present, exp_future, haz_present, impf_set]) or discount_rate is None or adaptation_mode is None:
            st.error("⚠️ One or more components are missing. Complete Part 1 before running analysis.")
        else:
            try:
                # Recreate MeasureSet dynamically
                selected_measures = []
                if adaptation_mode == "Standard1":
                    selected_measures = [Standard1]
                elif adaptation_mode == "Standard2":
                    selected_measures = [Standard2]
                elif adaptation_mode == "Custom":
                    custom_measure = Measure(
                        haz_type="TC",
                        name=f"Custom: {wind_reduction} m/s & {paa_protected}%",
                        color_rgb=np.array([0.1, 0.7, 0.1]),
                        cost=custom_cost,
                        hazard_inten_imp=(1, -float(wind_reduction)),
                        paa_impact=(1, -float(paa_protected) / 100.0),
                    )
                    selected_measures = [custom_measure]

                meas_set = MeasureSet(measure_list=selected_measures)
                meas_set.check()
                st.session_state.meas_set = meas_set

                disc_rates = DiscRates(
                    np.arange(exp_present.ref_year, exp_future.ref_year + 1),
                    np.ones(exp_future.ref_year - exp_present.ref_year + 1) * discount_rate
                )

                entity_present = Entity(
                    exposures=exp_present,
                    impact_func_set=impf_set,
                    measure_set=meas_set,
                    disc_rates=disc_rates
                )
                entity_future = Entity(
                    exposures=exp_future,
                    impact_func_set=impf_set,
                    measure_set=meas_set,
                    disc_rates=disc_rates
                )

                st.session_state.entity_present = entity_present
                st.session_state.entity_future = entity_future

                # Run CBA for different risk functions
                cb_agg = CostBenefit()
                cb_agg.calc(
                    haz_present,
                    entity_present,
                    haz_future=haz_future,
                    ent_future=entity_future,
                    future_year=exp_future.ref_year,
                    risk_func=risk_aai_agg,
                    imp_time_depen=1,
                    save_imp=True
                )

                cb_rp100 = CostBenefit()
                cb_rp100.calc(
                    haz_present,
                    entity_present,
                    haz_future=haz_future,
                    ent_future=entity_future,
                    future_year=exp_future.ref_year,
                    risk_func=risk_rp_100,
                    imp_time_depen=1,
                    save_imp=True
                )

                cb_rp250 = CostBenefit()
                cb_rp250.calc(
                    haz_present,
                    entity_present,
                    haz_future=haz_future,
                    ent_future=entity_future,
                    future_year=exp_future.ref_year,
                    risk_func=risk_rp_250,
                    imp_time_depen=1,
                    save_imp=True
                )
                #st.write("Keys in cb_agg.imp_meas_present:", list(cb_agg.imp_meas_present.keys()))
                #st.write("Keys in cb_rp100.imp_meas_present:", list(cb_rp100.imp_meas_present.keys()))
                #st.write("Keys in cb_rp250.imp_meas_present:", list(cb_rp250.imp_meas_present.keys()))

                st.session_state.cb = cb_agg
                st.session_state.cb = cb_rp100
                st.session_state.cb = cb_rp250
                df = pd.DataFrame({
                    "Measure": list(cb_agg.cost_ben_ratio.keys()),
                    "Investment or Cost (USD bn)": [cb_agg.imp_meas_future[m].get("cost", [None])[0]/1e9 for m in cb_agg.cost_ben_ratio],
                    "Expected Benefit (USD bn)": [cb_agg.benefit[m]/1e9 for m in cb_agg.cost_ben_ratio],
                    "Benefit/Cost": [cb_agg.cost_ben_ratio[m] for m in cb_agg.cost_ben_ratio],
                    "AAL Avoided (USD bn)": [cb_agg.benefit[m]/(exp_future.ref_year - exp_present.ref_year + 1)/1e9 for m in cb_agg.cost_ben_ratio],
                    "RP100 Avoided (bn)": [cb_rp100.benefit.get(m, float("nan"))/1e9 for m in cb_agg.cost_ben_ratio],
                    "RP250 Avoided (bn)": [cb_rp250.benefit.get(m, float("nan"))/1e9 for m in cb_agg.cost_ben_ratio],
                })
                st.session_state.cba_df = df
                st.success("✅ CBA analysis complete.")
      
            except Exception as e:
                st.error(f"❌ CBA error: {e}")
               
# 📌 Part 3: Results Summary & Visualization
# -----------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Risk Metrics", "🧠 Insights", "📈 Visualizations"])

with tab1:
    cb = st.session_state.get("cb")
    df = st.session_state.get("cba_df")
    if cb and df is not None:
        st.subheader("📉 Summary Risk Metrics")
        tot_risk = getattr(cb, "tot_climate_risk", np.nan)
        aal = getattr(cb, "annual_risk", np.nan)
        residual = min([v["risk"] for k, v in getattr(cb, "imp_meas_future", {}).items() if k != "no measure"], default=np.nan)
        metrics_table = pd.DataFrame({
            "Metric": ["Total Climate Risk", "Average Annual Loss", "Residual Risk"],
            "Value (USD bn)": [tot_risk/1e9, aal/1e9, residual/1e9]
        })
        st.table(metrics_table.style.format({"Value (USD bn)": "{:.2f}"}))

        st.markdown("### 💡 Benefit-Cost Table")
        st.dataframe(df.style.format({
            "Investment or Cost (USD bn)": "{:.2f}",
            "Expected Benefit (USD bn)": "{:.2f}",
            "Benefit/Cost Ratio": "{:.2f}",
            "RP100 Avoided Loss (USD bn)": "{:.2f}",
            "RP250 Avoided Loss (USD bn)": "{:.2f}"
        }))
    else:
        st.info("ℹ️ Run the analysis to see results.")

with tab2:
    df = st.session_state.get("cba_df")
    if df is not None and not df.empty:
        bcr_col = next((col for col in df.columns if "Benefit/Cost" in col), None)
        if bcr_col:
            max_row = df.loc[df[bcr_col].idxmax()]
            st.markdown("### 🔍 Key Insights")
            st.markdown(f"The most cost-effective adaptation is **{max_row['Measure']}**")
            st.markdown(f"It yields a benefit-cost ratio of **{max_row[bcr_col]:.2f}**")
            st.markdown(f"Total investment required: **{max_row['Investment or Cost (USD bn)']:.2f} billion**")
            st.markdown("✅ Recommendation: Prioritize high B/C ratio measures and refine custom strategies.")
        else:
            st.warning("⚠️ Could not locate Benefit/Cost Ratio column.")
    else:
        st.info("ℹ️ Run CBA to see insights.")

import matplotlib.pyplot as plt

with tab3:
    cb = st.session_state.get("cb")
    entity_present = st.session_state.get("entity_present")
    haz_present = st.session_state.get("haz_present")
    haz_future = st.session_state.get("haz_future")
    entity_future = st.session_state.get("entity_future")

    if cb and entity_present and haz_present:
        try:
            plt.figure(figsize=(3, 2))  # 👈 Adjust width and height here
            ax = cb.plot_waterfall(haz_present, entity_present, haz_future, entity_future, risk_func=risk_aai_agg)
            fig = ax.get_figure()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Failed to generate waterfall plot: {e}")
    else:
        st.info("ℹ️ Run full analysis to enable chart.")

st.subheader("Installed Python Packages")
packages = os.popen("pip list").read()
st.code(packages)
