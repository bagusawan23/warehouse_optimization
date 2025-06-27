"""
📦 Aplikasi Optimasi Layout Gudang dan Alur Picking

📎 File simulasi dapat diunduh di sini:
https://drive.google.com/file/d/1QxqGBvZsA18gMIu2EfVSZeTbrCRqTa4H/view?usp=sharing
"""

#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate dummy warehouse slot data

warehouse_slots = pd.read_csv('warehouse_slots.csv', sep=';')

# Step 2: Assign zones based on item_type for optimization
item_zone_map = {"consumable": 0, "spare_part": 1, "machine_part": 2}
warehouse_slots['zone_by_type'] = warehouse_slots['item_type'].map(item_zone_map)

# Step 3: Create optimized layout
optimized_layout = warehouse_slots.copy()
optimized_layout['x'] = optimized_layout['zone_by_type'] * 6 + np.random.randint(0, 3, size=num_slots)
optimized_layout['y'] = optimized_layout.groupby('zone_by_type').cumcount() % 20

# Step 4: Visualize before and after optimization
fig, ax = plt.subplots(figsize=(12, 5))

# Before optimization
plt.subplot(1, 2, 1)
for item_type in warehouse_slots['item_type'].unique():
    subset = warehouse_slots[warehouse_slots['item_type'] == item_type]
    plt.scatter(subset['x'], subset['y'], label=item_type, alpha=0.6)
plt.title("Before Optimization 📦")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# After optimization by item type
plt.subplot(1, 2, 2)
for item_type in optimized_layout['item_type'].unique():
    subset = optimized_layout[optimized_layout['item_type'] == item_type]
    plt.scatter(subset['x'], subset['y'], label=item_type, alpha=0.6)
plt.title("After Optimization by Item Type 🧩")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.tight_layout()
plt.show()


# In[83]:


warehouse_slots


# In[94]:


# Asumsikan titik packing point di (0, 0)
packing_point = np.array([0, 0])

# Fungsi untuk menghitung total jarak untuk satu order
def compute_order_distance(order_slots, layout_df):
    coords = layout_df[layout_df['slot_id'].isin(order_slots)][['x', 'y']].values
    distance = 0
    for coord in coords:
        distance += np.linalg.norm(coord - packing_point)
    return distance

# Buat dummy 50 order, masing-masing ambil 3 item unik
unique_slots = warehouse_slots['slot_id'].tolist()
orders = [np.random.choice(unique_slots, 3, replace=False).tolist() for _ in range(50)]

# Hitung total jarak dan waktu untuk sebelum dan sesudah optimasi
before_distances = np.array([compute_order_distance(order, warehouse_slots) for order in orders])
after_distances = np.array([compute_order_distance(order, optimized_layout) for order in orders])

# Kecepatan picker: 1.5 meter per detik
picker_speed = 1.5
before_times = before_distances / picker_speed
after_times = after_distances / picker_speed
time_saved = before_times - after_times

# Simpan hasil analisis dalam DataFrame
comparison_df = pd.DataFrame({
    "order_id": range(1, 51),
    "distance_before": before_distances.round(2),
    "distance_after": after_distances.round(2),
    "time_before_sec": before_times.round(2),
    "time_after_sec": after_times.round(2),
    "time_saved_sec": time_saved.round(2)
})

import seaborn as sns

# Visualisasi boxplot perbandingan waktu
plt.figure(figsize=(10, 6))
comparison_plot = pd.melt(comparison_df, id_vars=["order_id"], value_vars=["time_before_sec", "time_after_sec"],
                          var_name="Condition", value_name="Time (seconds)")
sns.boxplot(data=comparison_plot, x="Condition", y="Time (seconds)")
plt.title("⏱️ Perbandingan Waktu Picking Sebelum vs Sesudah Optimasi")
plt.ylabel("Total Waktu Picking per Order (detik)")
plt.xlabel("")
plt.grid(True)
plt.show()


comparison_df


# In[88]:


# Hitung ringkasan statistik efisiensi waktu
summary_stats = {
    "Rata-rata Waktu Sebelum (detik)": round(before_times.mean(), 2),
    "Rata-rata Waktu Sesudah (detik)": round(after_times.mean(), 2),
    "Rata-rata Penghematan Waktu (detik)": round(time_saved.mean(), 2),
    "Persentase Efisiensi Rata-rata (%)": round(100 * time_saved.mean() / before_times.mean(), 2)
}

# Buat histogram distribusi waktu picking
plt.figure(figsize=(12, 6))
plt.hist(before_times, bins=10, alpha=0.6, label="Sebelum Optimasi")
plt.hist(after_times, bins=10, alpha=0.6, label="Sesudah Optimasi")
plt.title("📊 Distribusi Waktu Picking Sebelum dan Sesudah Optimasi")
plt.xlabel("Waktu Picking per Order (detik)")
plt.ylabel("Jumlah Order")
plt.legend()
plt.grid(True)
plt.show()



summary_stats


# In[91]:


# Asumsi:
# - Rata-rata order per hari: 300 order
# - Hari kerja per bulan: 26 hari

orders_per_day = 300
working_days_per_month = 26

# Estimasi penghematan waktu total per hari dan per bulan
daily_time_saved_total_sec = time_saved.mean() * orders_per_day
monthly_time_saved_total_sec = daily_time_saved_total_sec * working_days_per_month

# Konversi ke jam
daily_time_saved_hours = daily_time_saved_total_sec / 3600
monthly_time_saved_hours = monthly_time_saved_total_sec / 3600

# Buat ringkasan estimasi
saving_summary = pd.DataFrame([{
    "Rata-rata Order per Hari": orders_per_day,
    "Hari Kerja per Bulan": working_days_per_month,
    "Total Waktu Hemat per Hari (detik)": round(daily_time_saved_total_sec, 2),
    "Total Waktu Hemat per Bulan (detik)": round(monthly_time_saved_total_sec, 2),
    "Total Waktu Hemat per Hari (jam)": round(daily_time_saved_hours, 2),
    "Total Waktu Hemat per Bulan (jam)": round(monthly_time_saved_hours, 2)
}])

saving_summary


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




