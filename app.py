"""Bangkok 3-Station Flood Research Dashboard.

Run from this folder: python -m streamlit run app.py
Historical pilot analysis only. Not a live flood warning service.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard_utils import (
    STATIONS, FLOOD_SENSORS, load_bundle, classification_metrics,
    rain_before, scenario_features,
)

ROOT = Path(__file__).resolve().parent
COLORS = {"RF.BKA.01": "#60A5FA", "RF.SMI.01": "#34D399", "RF.LSI.01": "#FBBF24"}
TEST_THRESHOLD = 0.25  # Selected by the original pilot's VALIDATION split, never by test.

st.set_page_config(page_title="Bangkok Flood Lab | 3 Stations", page_icon="🌧️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; max-width: 1500px;}
[data-testid="stMetric"] {background: #17243A; border:1px solid #273952; border-radius:12px; padding: 14px 16px;}
[data-testid="stMetricLabel"] {color:#9DB4D1;}
.kicker {font-size: 0.75rem; letter-spacing: 0.16em; color: #82ADDC; font-weight: 700;}
.subtitle {color: #9EB0C8; margin-top:-10px; margin-bottom:14px;}
.subtle {color:#9EB0C8; font-size:.86rem;}
[data-testid="stSidebar"] {background: #101A2C;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading 3-station pilot files...")
def data():
    return load_bundle(ROOT)

@st.cache_resource(show_spinner="Loading trusted local Random Forest artifact...")
def trusted_model():
    import joblib
    # WARNING: joblib/pickle deserialization can execute code. Never load external models without trust.
    return joblib.load(ROOT / "models" / "random_forest.joblib")

try:
    d = data()
except (ValueError, OSError, KeyError) as exc:
    st.error(f"Cannot open the bundled pilot data: {exc}")
    st.info("Keep the data/ directory alongside app.py. Re-download the complete ZIP if files are missing.")
    st.stop()

rain, flood, levels = d["rain"], d["flood"], d["levels"]
fixed_metrics = classification_metrics(d["test"], TEST_THRESHOLD)
report = d["report"]

with st.sidebar:
    st.markdown("### BANGKOK FLOOD LAB")
    st.caption("3 rainfall stations · historical 6-hour road-water pilot")
    st.divider()
    station = st.selectbox("Rain station", list(STATIONS), format_func=lambda k: STATIONS[k]["name"])
    view = st.radio("Prediction period", ["test", "validation", "replay"],
                    format_func={"test": "Test · 23–26 May 2025", "validation": "Validation · 15–17 May 2025", "replay": "Full pilot replay · May 2025"}.get)
    selected_preds = d[view]
    selected_time = st.select_slider(
        "Historical evaluation time (Bangkok)", options=list(selected_preds.index),
        value=selected_preds.index[len(selected_preds)//2],
        format_func=lambda v: v.strftime("%d %b %Y · %H:%M")
    )
    st.divider()
    st.markdown("**Research target**")
    st.caption("Water ≥5 cm at any of three road sensors in the following 6 hours. This includes continuing flooding; it is NOT strict onset prediction.")
    st.caption("Label threshold 5 cm: experimental, not an official BMA warning level.")
    st.divider()
    st.caption("Source: BMA rainfall and road-water systems; May 2025 selected pilot windows. Offline replay, not live data.")

st.markdown('<div class="kicker">FLOOD RESEARCH / BANGKOK, THAILAND</div>', unsafe_allow_html=True)
st.title("Bangkok Flood Lab")
st.markdown('<div class="subtitle">Interactive dashboard · Bang Khen / Sai Mai / Lak Si · 6-hour road-flood retrospective pilot</div>', unsafe_allow_html=True)
st.warning("**Research prototype, NOT an operational warning system.** The trained model has weak independent-test results. Scores have not been calibrated, and the 288 rainfall hours come from selected May 2025 windows—not ten years.", icon="⚠️")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Test precision", f"{fixed_metrics['precision']:.1%}")
c2.metric("Test recall", f"{fixed_metrics['recall']:.1%}")
c3.metric("Test F1", f"{fixed_metrics['f1']:.1%}")
c4.metric("Test average precision", f"{fixed_metrics['average_precision']:.3f}")
c5.metric("False alarm hours", str(fixed_metrics["fp"]))
st.caption("Fixed test scores: threshold 0.25 chosen on validation; 67 test hours, 7 positive targets, 1 TP, 13 FP, 6 FN. Scores are per overlapping hour, not per independent flood event.")

overview, stations_tab, replay_tab, validation_tab, scenario_tab, methods_tab = st.tabs([
    "Overview & map", "Rainfall stations", "Flood replay", "Model validation", "What-if lab", "Data & methods"
])

with overview:
    left,right = st.columns([1.18, 1], gap="large")
    with left:
        st.subheader("Monitoring locations")
        rows=[]
        amounts=rain_before(rain,selected_time,24)
        for code,meta in STATIONS.items():
            amount=amounts[code]
            rows.append({"lat":meta["lat"],"lon":meta["lon"],"station":meta["name"],
                         "rain24_mm":amount,"size":float(max(50,amount*8)) if pd.notna(amount) else 50.0})
        points=pd.DataFrame(rows)
        st.map(points,latitude="lat",longitude="lon",size="size",zoom=10)
        st.caption("Map pins are rain-gauge locations, sized by observed past-24h rainfall where complete. They are NOT predicted flood polygons or street inundation boundaries.")
        tbl=points[["station","rain24_mm"]].rename(columns={"station":"Rain station","rain24_mm":"Observed rain, past 24h (mm)"})
        st.dataframe(tbl,hide_index=True,use_container_width=True)
    with right:
        st.subheader("Selected historical hour")
        st.caption(selected_time.strftime("%d %B %Y · %H:%M Asia/Bangkok"))
        pick=selected_preds.loc[selected_time]
        a,b=st.columns(2)
        a.metric("RF model score",f"{pick['probability']:.1%}")
        b.metric("Experimental warning", "ON" if pick["probability"]>=TEST_THRESHOLD else "OFF")
        st.caption("Random Forest classification score (uncalibrated), NOT a verified flood probability.")
        if pd.notna(pick.get("target", np.nan)):
            st.metric("Observed target: flood during next 6h", "YES" if int(pick["target"]) else "NO")
        else:
            st.info("Ground truth is not provided for this replay timestamp; no accuracy judgment is shown.")
        st.divider()
        st.subheader("Data coverage — not 10 years")
        st.write(f"**Rain:** {d['audit']['rain_timestamps']} observed hours per station, across three selected May 2025 periods.")
        st.write(f"**Road sensors:** {len(levels):,} five-minute timestamps in May 2025.")
        st.write(f"**Positive road-water hours:** {d['audit']['recorded_flood_hours']} / {len(flood)} hours under the experimental 5 cm rule.")
        st.write(f"**Test F1:** {fixed_metrics['f1']:.3f}, compared with persistence baseline **{report['splits']['test']['persistence']['f1']:.3f}**.")
        st.error("The current Random Forest F1 is below the persistence baseline on this test period.")

with stations_tab:
    st.subheader("Hourly rainfall — actual BMA stations")
    view_start=selected_time-pd.Timedelta(hours=48)
    view_end=selected_time+pd.Timedelta(hours=12)
    chart_rain=rain.loc[view_start:view_end,list(STATIONS)]
    fig=go.Figure()
    for code,meta in STATIONS.items():
        fig.add_trace(go.Bar(x=chart_rain.index,y=chart_rain[code],name=meta['name'],marker_color=COLORS[code]))
    fig.add_vline(x=selected_time.timestamp()*1000,line_dash="dash",line_color="#E5E7EB",
                  annotation_text="Evaluation hour",annotation_position="top")
    fig.update_layout(barmode="group",height=390,yaxis_title="Observed rainfall (mm/h)",
                      xaxis_title="Local time (Asia/Bangkok)",legend_title="Station",margin=dict(l=10,r=10,t=18,b=5))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Missing hourly observations remain gaps (NaN), never zero. Rainfall measurements are historical; none of the plots above are future weather forecasts.")
    col1,col2=st.columns([1,1])
    with col1:
        st.subheader("Selected station — intensity profile")
        selected=rain.loc[view_start:view_end,station]
        b=go.Figure(go.Bar(x=selected.index,y=selected,marker_color=COLORS[station]))
        b.update_layout(height=300,yaxis_title="mm/h",xaxis_title="Local time",margin=dict(l=5,r=5,t=18,b=5))
        st.plotly_chart(b,use_container_width=True)
    with col2:
        st.subheader("Coverage by hour")
        c=rain[list(STATIONS)].notna().sum(axis=1).rename("stations_available")
        area=c.loc[view_start:view_end]
        coverage=go.Figure(go.Scatter(x=area.index,y=area,mode="lines",line=dict(color="#34D399"),fill="tozeroy"))
        coverage.update_layout(height=300,yaxis=dict(title="Reporting stations",range=[0,3.1],dtick=1),
                               xaxis_title="Local time",margin=dict(l=5,r=5,t=18,b=5))
        st.plotly_chart(coverage,use_container_width=True)
    st.download_button("Download observed rainfall CSV",d["rain"].to_csv(index_label="datetime"),
                       file_name="bangkok_3stations_observed_rainfall_pilot.csv",mime="text/csv")

with replay_tab:
    st.subheader("Historical flood-warning replay")
    st.caption("Observe how the SAVED model score behaves relative to the official road-water sensor-derived label. Ground truth is future six hours, not current rain and not observed flood onset.")
    if view=="replay":
        plotted=selected_preds.reindex(pd.date_range(selected_preds.index.min(),selected_preds.index.max(),freq="h"))
    else:
        plotted=selected_preds
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.10,row_heights=[0.66,0.34],
                      subplot_titles=("RF score and validation-selected threshold","Target: any flood in following 6h (when known)"))
    fig.add_trace(go.Scatter(x=plotted.index,y=plotted['probability'],mode="lines",name="Model score",
                             line=dict(color="#60A5FA",width=2)),row=1,col=1)
    fig.add_hline(y=TEST_THRESHOLD, row=1,col=1, line_dash="dash",line_color="#FBBF24",
                  annotation_text="0.25 threshold")
    if "target" in plotted:
        fig.add_trace(go.Scatter(x=plotted.index,y=plotted['target'],mode="lines",line_shape="hv",
                                 line=dict(color="#FB7185",width=2),name="Observed 6h label"),row=2,col=1)
    fig.add_vline(x=selected_time.timestamp()*1000,line_dash="dot",line_color="#DAE7F6",row=1,col=1)
    fig.update_yaxes(title_text="Model score",range=[0,1],row=1,col=1)
    fig.update_yaxes(title_text="Observed 0 / 1",range=[-.05,1.12],dtick=1,row=2,col=1)
    fig.update_layout(height=510,margin=dict(l=10,r=10,t=40,b=5),legend=dict(orientation="h",y=-0.1))
    st.plotly_chart(fig,use_container_width=True)
    cols=st.columns(3)
    cols[0].metric("Selected model score",f"{selected_preds.loc[selected_time,'probability']:.1%}")
    observed_now=flood['flooded'].get(selected_time,np.nan)
    cols[1].metric("Road flooding at selected hour",("Observed" if observed_now==1 else "Not observed" if observed_now==0 else "Unknown"))
    cols[2].metric("Prediction window","Next 6 hours")
    st.divider()
    st.subheader("Road-water sensors: observed depth near replay hour")
    around=levels.loc[selected_time-pd.Timedelta(hours=12):selected_time+pd.Timedelta(hours=12)]
    f=go.Figure()
    for i,(code,name) in enumerate(FLOOD_SENSORS.items()):
        f.add_trace(go.Scatter(x=around.index,y=around[code],name=f"{name} [{code}]",mode="lines"))
    f.add_hline(y=5,line_dash="dash",line_color="#FB7185",annotation_text="Experimental 5 cm rule")
    f.update_layout(height=360,yaxis_title="Observed road water depth (cm)",xaxis_title="Bangkok local time",
                    margin=dict(l=10,r=10,t=18,b=5))
    st.plotly_chart(f,use_container_width=True)
    st.caption("The chart is measured road-water depth, NOT river height or flood-inundated area. The experimental target requires 12 five-minute readings in each road sensor per labelled hour.")
    st.download_button("Download selected replay scores",selected_preds.to_csv(index_label="datetime"),
                       file_name=f"{view}_predictions.csv",mime="text/csv")

with validation_tab:
    st.subheader("Independent pilot evaluation")
    st.caption("The original model was selected on validation. Test data must not be reused for model or threshold tuning; the slider below is diagnostic only.")
    experimental=st.slider("Explore decision threshold (analysis only; NOT a new validated model)",0.05,0.95,TEST_THRESHOLD,0.05)
    if abs(experimental-TEST_THRESHOLD)>1e-8:
        st.warning("You have changed the displayed threshold AFTER viewing the test set. These new metrics are exploratory and must NOT be reported as unbiased test performance.")
    m=classification_metrics(d["test"],experimental)
    a,b,c,e=st.columns(4)
    a.metric("Test precision",f"{m['precision']:.1%}")
    b.metric("Test recall",f"{m['recall']:.1%}")
    c.metric("Test F1",f"{m['f1']:.3f}")
    e.metric("False alarms",m['fp'])
    left,right=st.columns([0.8,1.2])
    with left:
        st.markdown("**Confusion matrix · Test**")
        mat=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        fig=go.Figure(go.Heatmap(z=mat,x=["Predicted dry","Predicted flood"],
                                  y=["Actual dry","Actual flood"],colorscale="Blues",showscale=False,
                                  text=mat,texttemplate="%{text}",textfont=dict(size=22)))
        fig.update_layout(height=340,margin=dict(l=10,r=10,t=10,b=5),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("**Precision–recall curve · Test (67 hourly targets)**")
        from sklearn.metrics import precision_recall_curve
        y=d['test']['target'].astype(int)
        p=d['test']['probability']
        prec,rec,_=precision_recall_curve(y,p)
        fig=go.Figure(go.Scatter(x=rec,y=prec,name="Random Forest",mode="lines",line=dict(color="#60A5FA",width=3)))
        fig.add_hline(y=float(y.mean()),line_dash="dash",line_color="#FBBF24",annotation_text="Positive class prevalence")
        fig.update_layout(height=340,xaxis=dict(title="Recall",range=[0,1]),yaxis=dict(title="Precision",range=[0,1]),
                          margin=dict(l=10,r=10,t=10,b=5))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f"Average precision = {m['average_precision']:.3f} · Brier score = {m['brier']:.3f}; very small, correlated test sample; NOT calibrated probabilities.")
    st.markdown("**Fixed original comparison at threshold 0.25**")
    compare=pd.DataFrame([
        {"Method":"Random Forest · Validation", "Precision":report['splits']['validation']['precision'],
         "Recall":report['splits']['validation']['recall'],"F1":report['splits']['validation']['f1'],
         "Average precision":report['splits']['validation']['pr_auc_average_precision']},
        {"Method":"Persistence · Validation", "Precision":report['splits']['validation']['persistence']['precision'],
         "Recall":report['splits']['validation']['persistence']['recall'],"F1":report['splits']['validation']['persistence']['f1'],
         "Average precision":report['splits']['validation']['persistence']['pr_auc_average_precision']},
        {"Method":"Random Forest · Test", "Precision":report['splits']['test']['precision'],
         "Recall":report['splits']['test']['recall'],"F1":report['splits']['test']['f1'],
         "Average precision":report['splits']['test']['pr_auc_average_precision']},
        {"Method":"Persistence · Test", "Precision":report['splits']['test']['persistence']['precision'],
         "Recall":report['splits']['test']['persistence']['recall'],"F1":report['splits']['test']['persistence']['f1'],
         "Average precision":report['splits']['test']['persistence']['pr_auc_average_precision']},
    ])
    st.dataframe(compare.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}","Average precision":"{:.3f}"}),
                 hide_index=True,use_container_width=True)
    st.error("Current evidence is insufficient for an operational warning: 1 correct positive, 6 missed positives and 13 false alarms on the independent pilot test period.")

with scenario_tab:
    st.subheader("What-if rainfall explorer")
    st.caption("Like the original flood simulator, this tab explores a user-controlled scenario, but it modifies PAST rainfall at an already observed hour and recomputes RF features. It is not a weather forecast or flood-depth simulation.")
    trusted=st.checkbox("I trust this locally supplied .joblib model; enable model loading",value=False,
                        help="joblib uses pickle deserialization, which can execute code. Only enable for files you trust.")
    if not trusted:
        st.info("Model loading is OFF by default. Enable it to inspect genuine feature importance and explore rainfall perturbations on the existing Random Forest.")
    else:
        try:
            artifact=trusted_model()
            model=artifact['model']
            st.markdown("**Feature importance from the saved pilot Random Forest**")
            fi=pd.DataFrame({"feature":artifact['features'],"importance":model.feature_importances_}).sort_values('importance').tail(15)
            graph=go.Figure(go.Bar(x=fi['importance'],y=fi['feature'],orientation='h',marker_color='#60A5FA'))
            graph.update_layout(height=470,xaxis_title="Impurity-based feature importance (not causal)",margin=dict(l=10,r=10,t=20,b=5))
            st.plotly_chart(graph,use_container_width=True)
            station2=st.selectbox("Add hypothetical rain at",list(STATIONS),format_func=lambda s:STATIONS[s]['name'],key="what_if_station")
            span=st.selectbox("Modify preceding observed hours",[1,3,6],index=1)
            added=st.slider("Additional rainfall per hour (mm/h)",0.0,30.0,5.0,1.0)
            original_rain=rain[list(STATIONS)]
            features_original=scenario_features(original_rain,selected_time,station2,0.0,span)
            features_counter=scenario_features(original_rain,selected_time,station2,added,span)
            fcols=artifact['features']
            score_original=float(model.predict_proba(features_original[fcols])[0,1])
            score_counter=float(model.predict_proba(features_counter[fcols])[0,1])
            x,y=st.columns(2)
            x.metric("Original model score",f"{score_original:.1%}")
            y.metric("What-if score",f"{score_counter:.1%}",delta=f"{score_counter-score_original:+.1%} score change",delta_color="off")
            st.warning("Do not interpret these numbers as calibrated chances of flooding or as a live forecast. Rainfall is manually altered; observed road-water labels do NOT change.")
        except (KeyError,ValueError,FileNotFoundError,ImportError,AttributeError) as exc:
            st.error(f"Could not load or run the local model: {exc}")
            st.info("The other dashboard tabs still work from their precomputed CSV results.")

with methods_tab:
    st.subheader("Methodology and provenance")
    st.markdown("""
**Research question:** Can observed rainfall at three Bangkok stations predict whether at least one of three nearby road-water sensors will record >= 5 cm at any hour during the following 6 hours? (Includes existing flooding.)

**Rain stations:** RF.BKA.01 Bang Khen; RF.SMI.01 Sai Mai; RF.LSI.01 Lak Si. Fifteen rolling-sum features: previous 1/3/6/12/24 observed hours per station. **No future rainfall enters the feature set.**

**Road sensors:** FL.BKA.02 Phahon Yothin 55/1; FL.SMI.01 Phahon Yothin 60/1; FL.LSI.06 Ngamwongwan Soi Chin Khet. A valid negative requires 12 five-minute readings from all three sensors within the corresponding hour.

**Model:** RandomForestClassifier, 300 trees, min_samples_leaf=5, class_weight='balanced', random_state=42. Validation-selected warning threshold = 0.25. Test set not used to choose this threshold.

**Data split (selected May 2025 pilot windows only):** 73 train rows (before May 15), 72 validation rows (May 15–17), 67 test rows (May 23–26). The periods were preselected around rain events, introducing selection bias. Adjacent six-hour targets overlap; rows are not independent events.

**What this dashboard does NOT do:** Predict spatially gridded flood depth, number of buildings affected, flood extent or population impact. The uploaded simulator includes DEM, population, GIS, radar and hydrodynamic-style static water-level scenarios, none of which are measured outputs of this three-station model.
""")
    st.markdown("**Data limitations**")
    for limit in report['limitations']:
        st.markdown(f"- {limit}")
    st.markdown("""
Additional limitations: no verified station-health flags; 5 cm is a pilot research convention, not an official alert level; no continuous 2016–2025 station/flood dataset; no independent-event uncertainty intervals, calibration, or operational latency study. Flood occurrence for the next 6h differs from the START of a new flood event.

**Source systems:**
- [Bangkok rainfall historical station system](https://weather.bangkok.go.th/rain/RainHistory)
- [Bangkok road-water measurement system](https://weather.bangkok.go.th/floodbangkok/Graph/Station)

Data in this ZIP are saved historical extracts and model artifacts provided with the research, not a live feed. Event list coverage is broader than the pilot time windows and must not be taken as 19 independently tested events.
""")
    st.markdown("**Original study assets**")
    st.json({"stations":report['stations'],"train_rows":report['train_rows'],
             "validation_rows":report['splits']['validation']['rows'],"test_rows":report['splits']['test']['rows'],
             "validation_selected_threshold":report['threshold'],
             "test_F1":report['splits']['test']['f1'],"test_precision":report['splits']['test']['precision'],
             "test_recall":report['splits']['test']['recall']})
    st.download_button("Download pilot report (JSON)",(ROOT/'data'/'pilot_report.json').read_bytes(),
                       file_name='pilot_report.json',mime='application/json')

st.divider()
st.caption("Bangkok Flood Lab · Research pilot 12 Oct 2026 · Historical evidence only · 3 stations · DO NOT use for public safety decisions.")
