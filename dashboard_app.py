import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Jalur dataset dan model di repository.
DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi.csv")
MODEL_PATH = Path("model/Best_model.pkcls")

st.set_page_config(
    page_title="Dashboard Realisasi Anggaran",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    """Memuat dataset dan menambahkan label biner untuk target."""
    df = pd.read_csv(path)
    df["target"] = df["realisasi_tercapai_95persen"].map({"Ya": 1, "Tidak": 0})
    df["provinsi"] = df["provinsi"].astype("category")
    df["jenis_belanja_utama"] = df["jenis_belanja_utama"].astype("category")
    df["tipe_satker"] = df["tipe_satker"].astype("category")
    return df

@st.cache_resource
def load_model(path: Path):
    """Memuat model Orange yang disimpan dalam file pickle."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_model_feature_names(model) -> list[str]:
    """Mengembalikan urutan fitur yang digunakan oleh model."""
    return [attr.name for attr in model.domain.attributes]


def build_feature_matrix(df: pd.DataFrame, model) -> np.ndarray:
    """Membangun matriks fitur sesuai urutan model."""
    feature_names = get_model_feature_names(model)
    X = np.zeros((len(df), len(feature_names)), dtype=float)

    for i, name in enumerate(feature_names):
        if name.startswith("tipe_satker="):
            tipe_nama = name.split("=", 1)[1]
            X[:, i] = (df["tipe_satker"] == tipe_nama).astype(float)
        else:
            X[:, i] = df[name].astype(float)

    return X


def predict_from_features(X: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """Menghasilkan prediksi kelas dan probabilitas target positif."""
    prediction = model.skl_model.predict(X).astype(int)
    probability = model.skl_model.predict_proba(X)[:, 1]
    return prediction, probability


def build_manual_input(model) -> tuple[np.ndarray, dict]:
    """Membuat input manual untuk prediksi interaktif di sidebar."""
    st.sidebar.subheader("Input Manual untuk Prediksi")
    jumlah_spm = st.sidebar.number_input(
        "Jumlah SPM", min_value=0, max_value=1000, value=30, step=1
    )
    revisi_dipa = st.sidebar.number_input(
        "Revisi DIPA", min_value=0, max_value=20, value=1, step=1
    )
    deviasi_rpd_persen = st.sidebar.slider(
        "Deviasi RPD (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1
    )
    skor_ikpa = st.sidebar.slider(
        "Skor IKPA", min_value=0.0, max_value=100.0, value=80.0, step=0.1
    )
    tipe_satker = st.sidebar.selectbox(
        "Tipe Satker",
        ["Kantor Pusat", "Kantor Daerah", "Dekonsentrasi", "Tugas Pembantuan"],
    )

    input_data = {
        "jumlah_spm": jumlah_spm,
        "revisi_dipa": revisi_dipa,
        "deviasi_rpd_persen": deviasi_rpd_persen,
        "skor_ikpa": skor_ikpa,
        "tipe_satker": tipe_satker,
    }

    feature_names = get_model_feature_names(model)
    row = np.zeros((1, len(feature_names)), dtype=float)

    for i, name in enumerate(feature_names):
        if name.startswith("tipe_satker="):
            tipe_nama = name.split("=", 1)[1]
            row[0, i] = 1.0 if tipe_satker == tipe_nama else 0.0
        else:
            row[0, i] = float(input_data[name])

    return row, input_data


def main() -> None:
    st.title("Dashboard Realisasi Anggaran dan Prediksi 95%")
    st.markdown(
        "Dashboard ini menampilkan data anggaran, evaluasi model, dan prediksi interaktif untuk target `realisasi_tercapai_95persen`."
    )

    df = load_dataset(DATA_PATH)
    model = load_model(MODEL_PATH)

    X_all = build_feature_matrix(df, model)
    prediction_all, probability_all = predict_from_features(X_all, model)

    df = df.copy()
    df["prediksi"] = np.where(prediction_all == 1, "Ya", "Tidak")
    df["prediksi_proba"] = probability_all

    st.sidebar.header("Filter Data")
    provinsi_filter = st.sidebar.multiselect(
        "Provinsi",
        sorted(df["provinsi"].cat.categories),
        default=list(df["provinsi"].cat.categories),
    )
    tipe_satker_filter = st.sidebar.multiselect(
        "Tipe Satker",
        sorted(df["tipe_satker"].cat.categories),
        default=list(df["tipe_satker"].cat.categories),
    )
    jenis_belanja_filter = st.sidebar.multiselect(
        "Jenis Belanja Utama",
        sorted(df["jenis_belanja_utama"].cat.categories),
        default=list(df["jenis_belanja_utama"].cat.categories),
    )

    filtered_df = df[
        df["provinsi"].isin(provinsi_filter)
        & df["tipe_satker"].isin(tipe_satker_filter)
        & df["jenis_belanja_utama"].isin(jenis_belanja_filter)
    ]

    st.subheader("Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Satker", len(filtered_df))
    col2.metric(
        "Target Tercapai 95%",
        f"{filtered_df['target'].mean() * 100:.1f}%",
        delta=f"{(filtered_df['target'].mean() - df['target'].mean()) * 100:.1f}% vs total",
    )
    col3.metric("Rata-rata Skor IKPA", f"{filtered_df['skor_ikpa'].mean():.2f}")
    col4.metric("Rata-rata Deviasi RPD", f"{filtered_df['deviasi_rpd_persen'].mean():.2f}%")

    with st.expander("📊 Statistik Deskriptif Fitur Numerik"):
        st.dataframe(
            filtered_df[
                [
                    "pagu_miliar",
                    "jumlah_pegawai",
                    "jumlah_spm",
                    "revisi_dipa",
                    "realisasi_tw1_persen",
                    "realisasi_tw2_persen",
                    "realisasi_tw3_persen",
                    "deviasi_rpd_persen",
                    "skor_ikpa",
                ]
            ].describe().T
        )

    st.subheader("Distribusi Target dan Probabilitas Prediksi")
    target_share = filtered_df.groupby("provinsi")["target"].mean()
    pred_share = filtered_df.groupby("provinsi")["prediksi_proba"].mean()

    chart_df = pd.DataFrame(
        {
            "Target 95% Tercapai": target_share,
            "Probabilitas Prediksi": pred_share,
        }
    )
    st.bar_chart(chart_df)

    if st.checkbox("Tampilkan data mentah yang difilter", value=True):
        st.dataframe(filtered_df.drop(columns=["target"]))

    st.subheader("Evaluasi Model pada Seluruh Dataset")
    accuracy = (prediction_all == df["target"]).mean()
    st.write(f"Akurasi model terhadap seluruh dataset: **{accuracy * 100:.2f}%**")

    coef_values = model.skl_model.coef_[0]
    coef_names = get_model_feature_names(model)
    coef_df = pd.DataFrame(
        {
            "Fitur": coef_names,
            "Koefisien": coef_values,
            "Kekuatan": np.abs(coef_values),
        }
    ).sort_values(by="Kekuatan", ascending=False)

    st.subheader("Interpretasi Model")
    st.write(
        "Koefisien regresi logistik menunjukkan arah pengaruh fitur terhadap peluang target `Ya` (tercapai 95%)."
    )
    st.dataframe(coef_df.reset_index(drop=True))
    st.bar_chart(coef_df.set_index("Fitur")["Koefisien"])

    st.subheader("Prediksi Interaktif")
    input_mode = st.radio("Pilih sumber input:", ["Pilih baris data", "Input manual"])

    if input_mode == "Pilih baris data":
        index = st.selectbox("Pilih index baris:", filtered_df.index, index=0)
        sample_row = filtered_df.loc[index]
        st.write("### Data Terpilih")
        st.write(sample_row.to_frame().T)
        X_sample = build_feature_matrix(sample_row.to_frame(), model)
        pred, prob = predict_from_features(X_sample, model)
    else:
        X_sample, input_data = build_manual_input(model)
        st.write("### Input Manual")
        st.json(input_data)
        pred, prob = predict_from_features(X_sample, model)

    st.markdown(
        f"**Hasil Prediksi:** {'Ya' if pred == 1 else 'Tidak'} dengan probabilitas {prob * 100:.2f}%"
    )

    st.markdown(
        "---\n"
        "### Penjelasan Kode\n"
        "1. `load_dataset`: memuat CSV dan menyiapkan label target biner.\n"
        "2. `load_model`: memuat model Orange dari `model/Best_model.pkcls`.\n"
        "3. `build_feature_matrix`: membuat fitur numerik dan biner tipe_satker agar cocok dengan model.\n"
        "4. `predict_from_features`: memakai `model.skl_model` internal untuk menghasilkan prediksi.\n"
        "5. Bagian utama: filter, ringkasan statistik, visualisasi, evaluasi model, dan prediksi interaktif.\n"
    )


if __name__ == "__main__":
    main()
