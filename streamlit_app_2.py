import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Realisasi Anggaran Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
        border: 1px solid #2e3452;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #ffffff; }
    .metric-card .label { font-size: 0.85rem; color: #8b92a5; margin-top: 4px; }
    .metric-card .sub { font-size: 0.8rem; margin-top: 6px; }
    .predict-card {
        background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
        border: 1px solid #2e3452;
        border-radius: 12px;
        padding: 24px;
    }
    .result-ya {
        background: linear-gradient(135deg, #0d3b1e, #155724);
        border: 1px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-tidak {
        background: linear-gradient(135deg, #3b0d0d, #571515);
        border: 1px solid #dc3545;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e6f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #2e3452;
    }
    .stTabs [data-baseweb="tab"] { color: #8b92a5; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #4f8ef7 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data & model loaders ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/02_realisasi_anggaran_klasifikasi.csv")
    return df


@st.cache_resource
def load_model():
    with open("models/Best_model.pkcls", "rb") as f:
        return pickle.load(f)


# ── Orange prediction helper ──────────────────────────────────────────────────
def predict_orange(model, jumlah_spm, revisi_dipa, deviasi_rpd_persen,
                   skor_ikpa, tipe_satker):
    from Orange.data import Instance
    tipe_map = {
        "Dekonsentrasi":    [1, 0, 0, 0],
        "Kantor Daerah":    [0, 1, 0, 0],
        "Kantor Pusat":     [0, 0, 1, 0],
        "Tugas Pembantuan": [0, 0, 0, 1],
    }
    oh = tipe_map.get(tipe_satker, [0, 0, 0, 0])
    vals = [jumlah_spm, revisi_dipa, deviasi_rpd_persen, skor_ikpa] + oh + [None]
    inst = Instance(model.domain, vals)
    pred_idx = model(inst, model.Value)
    proba = model(inst, model.Probs)
    label = model.domain.class_var.values[int(pred_idx)]
    return label, float(proba[0]), float(proba[1])   # label, p_tidak, p_ya


# ── Load assets ───────────────────────────────────────────────────────────────
df = load_data()
model = load_model()

COLOR_YA    = "#28a745"
COLOR_TIDAK = "#dc3545"
COLOR_BLUE  = "#4f8ef7"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8cfe0", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filter Data")
    st.markdown("---")

    kementerian_list = ["Semua"] + sorted(df["nama_kementerian"].unique())
    sel_kementerian  = st.multiselect("Kementerian", kementerian_list[1:],
                                      default=kementerian_list[1:])

    provinsi_list = ["Semua"] + sorted(df["provinsi"].unique())
    sel_provinsi  = st.multiselect("Provinsi", provinsi_list[1:],
                                   default=provinsi_list[1:])

    tipe_list = ["Semua"] + sorted(df["tipe_satker"].unique())
    sel_tipe  = st.multiselect("Tipe Satker", tipe_list[1:],
                               default=tipe_list[1:])

    jenis_list = ["Semua"] + sorted(df["jenis_belanja_utama"].unique())
    sel_jenis  = st.multiselect("Jenis Belanja", jenis_list[1:],
                                default=jenis_list[1:])

    st.markdown("---")
    pagu_range = st.slider(
        "Pagu (Miliar Rp)",
        float(df["pagu_miliar"].min()),
        float(df["pagu_miliar"].max()),
        (float(df["pagu_miliar"].min()), float(df["pagu_miliar"].max())),
        step=1.0,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_kementerian:
    fdf = fdf[fdf["nama_kementerian"].isin(sel_kementerian)]
if sel_provinsi:
    fdf = fdf[fdf["provinsi"].isin(sel_provinsi)]
if sel_tipe:
    fdf = fdf[fdf["tipe_satker"].isin(sel_tipe)]
if sel_jenis:
    fdf = fdf[fdf["jenis_belanja_utama"].isin(sel_jenis)]
fdf = fdf[(fdf["pagu_miliar"] >= pagu_range[0]) & (fdf["pagu_miliar"] <= pagu_range[1])]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2rem; font-weight:700; color:#e0e6f0; margin-bottom:4px'>
💰 Dashboard Realisasi Anggaran
</h1>
<p style='color:#8b92a5; margin-bottom:24px'>
Analisis & Prediksi Ketercapaian Realisasi Anggaran ≥ 95%
</p>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔎 Analisis Detail", "🤖 Prediksi"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    total     = len(fdf)
    ya_count  = (fdf["realisasi_tercapai_95persen"] == "Ya").sum()
    tdk_count = total - ya_count
    pct_ya    = ya_count / total * 100 if total else 0
    avg_tw3   = fdf["realisasi_tw3_persen"].mean()
    avg_ikpa  = fdf["skor_ikpa"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, str(total),           "Total Satker",               None),
        (c2, str(ya_count),        "Tercapai ≥95%",              f"<span style='color:{COLOR_YA}'>✔ {pct_ya:.1f}%</span>"),
        (c3, str(tdk_count),       "Belum Tercapai",             f"<span style='color:{COLOR_TIDAK}'>✖ {100-pct_ya:.1f}%</span>"),
        (c4, f"{avg_tw3:.1f}%",    "Avg Realisasi TW3",          None),
        (c5, f"{avg_ikpa:.1f}",    "Avg Skor IKPA",              None),
    ]
    for col, val, lbl, sub in metrics:
        with col:
            sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>{val}</div>
                <div class='label'>{lbl}</div>
                {sub_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1])

    # Donut
    with col_l:
        st.markdown("<div class='section-header'>Distribusi Ketercapaian</div>", unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Tidak Tercapai", "Tercapai ≥95%"],
            values=[tdk_count, ya_count],
            hole=0.65,
            marker_colors=[COLOR_TIDAK, COLOR_YA],
            textinfo="percent+label",
            textfont=dict(size=13),
        ))
        fig_donut.add_annotation(text=f"<b>{pct_ya:.1f}%</b><br>Tercapai",
                                  showarrow=False, font=dict(size=16, color="#fff"),
                                  x=0.5, y=0.5)
        fig_donut.update_layout(**PLOTLY_THEME, height=320, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Realisasi per TW
    with col_r:
        st.markdown("<div class='section-header'>Tren Realisasi per Triwulan</div>", unsafe_allow_html=True)
        tw_means = {
            "TW1": fdf["realisasi_tw1_persen"].mean(),
            "TW2": fdf["realisasi_tw2_persen"].mean(),
            "TW3": fdf["realisasi_tw3_persen"].mean(),
        }
        fig_tw = go.Figure()
        fig_tw.add_trace(go.Scatter(
            x=list(tw_means.keys()), y=list(tw_means.values()),
            mode="lines+markers+text",
            line=dict(color=COLOR_BLUE, width=3),
            marker=dict(size=10, color=COLOR_BLUE),
            text=[f"{v:.1f}%" for v in tw_means.values()],
            textposition="top center",
            fill="tozeroy",
            fillcolor="rgba(79,142,247,0.15)",
        ))
        fig_tw.add_hline(y=95, line_dash="dot", line_color="#f0ad4e",
                         annotation_text="Target 95%", annotation_font_color="#f0ad4e")
        fig_tw.update_layout(**PLOTLY_THEME, height=320, yaxis_range=[0, 105],
                              yaxis_title="Rata-rata (%)")
        st.plotly_chart(fig_tw, use_container_width=True)

    # By Kementerian
    st.markdown("<div class='section-header'>Ketercapaian per Kementerian</div>", unsafe_allow_html=True)
    grp = fdf.groupby(["nama_kementerian", "realisasi_tercapai_95persen"]).size().unstack(fill_value=0)
    grp.columns = [c for c in grp.columns]
    grp_pct = grp.div(grp.sum(axis=1), axis=0) * 100
    grp_pct = grp_pct.reset_index().sort_values("Ya", ascending=True) if "Ya" in grp_pct.columns else grp_pct.reset_index()

    fig_bar = go.Figure()
    if "Tidak" in grp_pct.columns:
        fig_bar.add_trace(go.Bar(name="Tidak Tercapai", x=grp_pct["Tidak"],
                                  y=grp_pct["nama_kementerian"], orientation="h",
                                  marker_color=COLOR_TIDAK, text=grp_pct["Tidak"].round(1).astype(str)+"%",
                                  textposition="inside"))
    if "Ya" in grp_pct.columns:
        fig_bar.add_trace(go.Bar(name="Tercapai ≥95%", x=grp_pct["Ya"],
                                  y=grp_pct["nama_kementerian"], orientation="h",
                                  marker_color=COLOR_YA, text=grp_pct["Ya"].round(1).astype(str)+"%",
                                  textposition="inside"))
    fig_bar.update_layout(**PLOTLY_THEME, barmode="stack", height=380,
                           xaxis_title="Persentase (%)", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        # Box: skor IKPA by hasil
        st.markdown("<div class='section-header'>Skor IKPA vs Ketercapaian</div>", unsafe_allow_html=True)
        fig_box = px.box(fdf, x="realisasi_tercapai_95persen", y="skor_ikpa",
                         color="realisasi_tercapai_95persen",
                         color_discrete_map={"Ya": COLOR_YA, "Tidak": COLOR_TIDAK},
                         points="outliers", labels={"skor_ikpa": "Skor IKPA",
                         "realisasi_tercapai_95persen": ""})
        fig_box.update_layout(**PLOTLY_THEME, showlegend=False, height=340)
        st.plotly_chart(fig_box, use_container_width=True)

        # Deviasi RPD
        st.markdown("<div class='section-header'>Deviasi RPD vs Ketercapaian</div>", unsafe_allow_html=True)
        fig_dev = px.box(fdf, x="realisasi_tercapai_95persen", y="deviasi_rpd_persen",
                         color="realisasi_tercapai_95persen",
                         color_discrete_map={"Ya": COLOR_YA, "Tidak": COLOR_TIDAK},
                         points="outliers", labels={"deviasi_rpd_persen": "Deviasi RPD (%)",
                         "realisasi_tercapai_95persen": ""})
        fig_dev.update_layout(**PLOTLY_THEME, showlegend=False, height=340)
        st.plotly_chart(fig_dev, use_container_width=True)

    with col_r:
        # Scatter: TW3 vs IKPA
        st.markdown("<div class='section-header'>Realisasi TW3 vs Skor IKPA</div>", unsafe_allow_html=True)
        fig_sc = px.scatter(fdf, x="skor_ikpa", y="realisasi_tw3_persen",
                            color="realisasi_tercapai_95persen",
                            color_discrete_map={"Ya": COLOR_YA, "Tidak": COLOR_TIDAK},
                            hover_data=["nama_kementerian", "provinsi", "tipe_satker"],
                            labels={"skor_ikpa": "Skor IKPA",
                                    "realisasi_tw3_persen": "Realisasi TW3 (%)",
                                    "realisasi_tercapai_95persen": ""},
                            opacity=0.75)
        fig_sc.add_hline(y=95, line_dash="dot", line_color="#f0ad4e",
                         annotation_text="Target 95%")
        fig_sc.update_layout(**PLOTLY_THEME, height=340,
                              legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_sc, use_container_width=True)

        # By Tipe Satker
        st.markdown("<div class='section-header'>Ketercapaian per Tipe Satker</div>", unsafe_allow_html=True)
        grp_t = fdf.groupby(["tipe_satker", "realisasi_tercapai_95persen"]).size().unstack(fill_value=0)
        grp_t_pct = grp_t.div(grp_t.sum(axis=1), axis=0) * 100
        grp_t_pct = grp_t_pct.reset_index()
        fig_tipe = go.Figure()
        if "Tidak" in grp_t_pct.columns:
            fig_tipe.add_trace(go.Bar(name="Tidak", y=grp_t_pct["tipe_satker"],
                                       x=grp_t_pct["Tidak"], orientation="h",
                                       marker_color=COLOR_TIDAK))
        if "Ya" in grp_t_pct.columns:
            fig_tipe.add_trace(go.Bar(name="Ya", y=grp_t_pct["tipe_satker"],
                                       x=grp_t_pct["Ya"], orientation="h",
                                       marker_color=COLOR_YA))
        fig_tipe.update_layout(**PLOTLY_THEME, barmode="stack", height=340,
                                xaxis_title="Persentase (%)", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_tipe, use_container_width=True)

    # Heatmap: Provinsi x Kementerian (avg TW3)
    st.markdown("<div class='section-header'>Rata-rata Realisasi TW3 per Provinsi & Jenis Belanja</div>", unsafe_allow_html=True)
    pivot = fdf.pivot_table(values="realisasi_tw3_persen",
                             index="provinsi", columns="jenis_belanja_utama",
                             aggfunc="mean")
    fig_heat = px.imshow(pivot, color_continuous_scale="RdYlGn",
                          zmin=0, zmax=100, aspect="auto",
                          labels=dict(color="Avg TW3 (%)"))
    fig_heat.update_layout(**PLOTLY_THEME, height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Raw table
    with st.expander("📋 Lihat Data Mentah", expanded=False):
        display_cols = ["kode_satker", "nama_kementerian", "provinsi", "tipe_satker",
                        "jenis_belanja_utama", "pagu_miliar", "skor_ikpa",
                        "realisasi_tw3_persen", "realisasi_tercapai_95persen"]
        st.dataframe(fdf[display_cols].reset_index(drop=True),
                     use_container_width=True, height=350)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <p style='color:#8b92a5; margin-bottom:24px'>
    Model: <b style='color:#4f8ef7'>Logistic Regression</b> (Orange3) — 
    fitur: jumlah_spm, revisi_dipa, deviasi_rpd_persen, skor_ikpa, tipe_satker
    </p>""", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-header'>Input Parameter Satker</div>", unsafe_allow_html=True)

        tipe_satker_input = st.selectbox(
            "Tipe Satker", ["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"]
        )
        jumlah_spm_input = st.number_input(
            "Jumlah SPM", min_value=0, max_value=500, value=80,
            help="Jumlah Surat Perintah Membayar yang diterbitkan"
        )
        revisi_dipa_input = st.number_input(
            "Revisi DIPA", min_value=0, max_value=20, value=2,
            help="Jumlah revisi DIPA dalam tahun berjalan"
        )
        deviasi_rpd_input = st.number_input(
            "Deviasi RPD (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1,
            help="Deviasi antara rencana dan realisasi penyerapan (%)"
        )
        skor_ikpa_input = st.number_input(
            "Skor IKPA", min_value=70.0, max_value=100.0, value=85.0, step=0.1,
            help="Indikator Kinerja Pelaksanaan Anggaran (0-100)"
        )

        predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-header'>Hasil Prediksi</div>", unsafe_allow_html=True)

        if predict_btn:
            label, p_tidak, p_ya = predict_orange(
                model, jumlah_spm_input, revisi_dipa_input,
                deviasi_rpd_input, skor_ikpa_input, tipe_satker_input
            )
            is_ya = (label == "Ya")
            card_cls = "result-ya" if is_ya else "result-tidak"
            icon      = "✅" if is_ya else "❌"
            txt_col   = COLOR_YA if is_ya else COLOR_TIDAK
            verdict   = "TERCAPAI ≥ 95%" if is_ya else "BELUM TERCAPAI"

            st.markdown(f"""
            <div class='{card_cls}'>
                <div style='font-size:3rem'>{icon}</div>
                <div style='font-size:1.5rem; font-weight:700; color:{txt_col}; margin-top:8px'>{verdict}</div>
                <div style='color:#c8cfe0; margin-top:6px; font-size:0.9rem'>
                    Prediksi realisasi anggaran akhir tahun
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_ya * 100,
                title={"text": "Probabilitas Tercapai", "font": {"color": "#c8cfe0", "size": 14}},
                number={"suffix": "%", "font": {"color": "#ffffff", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b92a5"},
                    "bar": {"color": COLOR_YA if is_ya else COLOR_TIDAK},
                    "bgcolor": "#1e2235",
                    "bordercolor": "#2e3452",
                    "steps": [
                        {"range": [0, 50],  "color": "rgba(220,53,69,0.2)"},
                        {"range": [50, 75], "color": "rgba(240,173,78,0.2)"},
                        {"range": [75, 100],"color": "rgba(40,167,69,0.2)"},
                    ],
                    "threshold": {"line": {"color": "#f0ad4e", "width": 3},
                                  "thickness": 0.8, "value": 50},
                },
            ))
            fig_gauge.update_layout(**PLOTLY_THEME, height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Prob bar
            fig_prob = go.Figure(go.Bar(
                x=["Tidak Tercapai", "Tercapai ≥95%"],
                y=[p_tidak * 100, p_ya * 100],
                marker_color=[COLOR_TIDAK, COLOR_YA],
                text=[f"{p_tidak*100:.1f}%", f"{p_ya*100:.1f}%"],
                textposition="outside",
                textfont=dict(size=14),
            ))
            fig_prob.update_layout(**PLOTLY_THEME, height=220, yaxis_range=[0, 115],
                                    yaxis_title="Probabilitas (%)", showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#8b92a5'>
                <div style='font-size:3rem; margin-bottom:16px'>🤖</div>
                <div style='font-size:1rem'>Isi parameter di kiri dan klik<br>
                <b style='color:#4f8ef7'>Prediksi Sekarang</b></div>
            </div>""", unsafe_allow_html=True)

    # ── Batch prediction ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>🗂️ Prediksi Batch dari Dataset</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b92a5; font-size:0.9rem'>Jalankan prediksi model pada seluruh data yang difilter dan bandingkan dengan label aktual.</p>", unsafe_allow_html=True)

    if st.button("▶ Jalankan Prediksi Batch", use_container_width=False):
        from Orange.data import Instance

        results = []
        for _, row in fdf.iterrows():
            lbl, p_t, p_y = predict_orange(
                model, row["jumlah_spm"], row["revisi_dipa"],
                row["deviasi_rpd_persen"], row["skor_ikpa"], row["tipe_satker"]
            )
            results.append({"Prediksi": lbl, "Prob_Ya": round(p_y * 100, 1),
                             "Prob_Tidak": round(p_t * 100, 1)})

        res_df = pd.concat([
            fdf[["kode_satker", "nama_kementerian", "tipe_satker",
                  "skor_ikpa", "deviasi_rpd_persen",
                  "realisasi_tercapai_95persen"]].reset_index(drop=True),
            pd.DataFrame(results)
        ], axis=1)

        correct = (res_df["realisasi_tercapai_95persen"] == res_df["Prediksi"]).sum()
        acc = correct / len(res_df) * 100

        mc1, mc2, mc3 = st.columns(3)
        for col, val, lbl in [
            (mc1, str(len(res_df)), "Total Diprediksi"),
            (mc2, str(correct),     "Prediksi Benar"),
            (mc3, f"{acc:.1f}%",    "Akurasi"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='value'>{val}</div>
                    <div class='label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # Confusion visual
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(res_df["realisasi_tercapai_95persen"], res_df["Prediksi"],
                              labels=["Tidak", "Ya"])
        fig_cm = px.imshow(cm, text_auto=True, x=["Pred: Tidak", "Pred: Ya"],
                           y=["Aktual: Tidak", "Aktual: Ya"],
                           color_continuous_scale="Blues", aspect="auto")
        fig_cm.update_layout(**PLOTLY_THEME, title="Confusion Matrix", height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.dataframe(res_df.reset_index(drop=True), use_container_width=True, height=320)
