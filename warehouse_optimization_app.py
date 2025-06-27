
"""
📦 Aplikasi Interaktif: Optimasi Layout Gudang dan Alur Picking

📎 File simulasi dapat diunduh di sini:
https://drive.google.com/file/d/1QxqGBvZsA18gMIu2EfVSZeTbrCRqTa4H/view?usp=sharing
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

st.set_page_config(page_title="Optimasi Gudang", layout="wide")

st.title("📦 Optimasi Layout Gudang dan Alur Picking")
st.markdown("Unggah file layout gudang (CSV/XLSX) yang berisi kolom: `slot_id`, `item_type`, `x`, `y`.")

uploaded_file = st.file_uploader("📁 Upload File Layout Gudang", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Layout Gudang")
    st.dataframe(df)

    with st.sidebar:
        st.header("⚙️ Parameter Simulasi")
        num_orders = st.number_input("Jumlah Order", 10, 1000, 50)
        items_per_order = st.slider("Jumlah Slot per Order", 1, 5, 3)
        picker_speed = st.number_input("Kecepatan Picker (m/s)", 0.1, 5.0, 1.5)
        orders_per_day = st.number_input("Order per Hari", 1, 1000, 300)
        working_days_per_month = st.number_input("Hari Kerja per Bulan", 1, 31, 26)

    item_zone_map = {"consumable": 0, "spare_part": 1, "machine_part": 2}
    df['zone_by_type'] = df['item_type'].map(item_zone_map)

    optimized_layout = df.copy()
    optimized_layout['x'] = optimized_layout['zone_by_type'] * 6 + np.random.randint(0, 3, size=len(df))
    optimized_layout['y'] = optimized_layout.groupby('zone_by_type').cumcount() % 20

    st.subheader("🗺️ Visualisasi Layout Gudang")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for t in df['item_type'].unique():
        axes[0].scatter(df[df['item_type'] == t]['x'], df[df['item_type'] == t]['y'], label=t, alpha=0.6)
    axes[0].set_title("Sebelum Optimasi")
    axes[0].legend()

    for t in optimized_layout['item_type'].unique():
        axes[1].scatter(optimized_layout[optimized_layout['item_type'] == t]['x'],
                        optimized_layout[optimized_layout['item_type'] == t]['y'], label=t, alpha=0.6)
    axes[1].set_title("Setelah Optimasi")
    axes[1].legend()
    st.pyplot(fig)

    st.subheader("🚶 Simulasi Picking")
    packing_point = np.array([0, 0])
    def compute_distance(order_slots, layout):
        coords = layout[layout['slot_id'].isin(order_slots)][['x', 'y']].values
        return sum(np.linalg.norm(coord - packing_point) for coord in coords)

    slot_ids = df['slot_id'].tolist()
    orders = [np.random.choice(slot_ids, items_per_order, replace=False).tolist() for _ in range(num_orders)]

    before_d = np.array([compute_distance(o, df) for o in orders])
    after_d = np.array([compute_distance(o, optimized_layout) for o in orders])
    before_t = before_d / picker_speed
    after_t = after_d / picker_speed
    saved_t = before_t - after_t

    result_df = pd.DataFrame({
        "order_id": range(1, num_orders + 1),
        "time_before_sec": before_t.round(2),
        "time_after_sec": after_t.round(2),
        "time_saved_sec": saved_t.round(2)
    })

    st.dataframe(result_df)

    st.subheader("📊 Visualisasi Waktu Picking")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=pd.melt(result_df, id_vars=["order_id"], 
                             value_vars=["time_before_sec", "time_after_sec"]),
                x="variable", y="value", ax=ax2)
    ax2.set_title("⏱️ Perbandingan Waktu Picking")
    ax2.set_xlabel("")
    ax2.set_ylabel("Waktu (detik)")
    st.pyplot(fig2)

    st.subheader("📈 Distribusi Waktu Picking")
    fig3, ax3 = plt.subplots()
    ax3.hist(before_t, bins=10, alpha=0.6, label="Sebelum")
    ax3.hist(after_t, bins=10, alpha=0.6, label="Sesudah")
    ax3.legend()
    ax3.set_xlabel("Waktu (detik)")
    ax3.set_ylabel("Jumlah Order")
    ax3.set_title("📊 Distribusi Waktu")
    st.pyplot(fig3)

    st.subheader("📌 Ringkasan Efisiensi")
    mean_saved = saved_t.mean()
    percent_saved = 100 * mean_saved / before_t.mean()
    st.metric("⏳ Rata-rata Waktu Hemat per Order (detik)", f"{mean_saved:.2f}")
    st.metric("📉 Persentase Efisiensi", f"{percent_saved:.2f}%")

    st.subheader("📅 Estimasi Waktu Hemat Harian & Bulanan")
    total_daily = mean_saved * orders_per_day
    total_monthly = total_daily * working_days_per_month
    st.write(f"🕒 Total Hemat per Hari: `{total_daily:.2f}` detik ({total_daily/3600:.2f} jam)")
    st.write(f"🕒 Total Hemat per Bulan: `{total_monthly:.2f}` detik ({total_monthly/3600:.2f} jam)")
