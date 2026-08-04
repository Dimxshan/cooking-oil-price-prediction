# 🛢️ FuzzyOil — Cooking Oil Price Prediction with Fuzzy Logic

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Final%20Project-success)
![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey)

Sistem prediksi harga minyak goreng domestik Indonesia berbasis **Fuzzy Inference System (FIS)**, dengan implementasi metode **Mamdani** dan **Sugeno** yang dibangun sepenuhnya *from scratch* (tanpa `scikit-fuzzy`) menggunakan Python dan divisualisasikan melalui dashboard interaktif Streamlit.

📄 **Laporan Lengkap:** [Google Docs Report](https://docs.google.com/document/d/1OLKYKMNi1AVNo385z-SqgQBwoSUJp7TTR2zpTvhv-3A/edit?usp=sharing)

---

## 📑 Daftar Isi

- [Konteks Proyek](#-konteks-proyek)
- [Anggota Tim](#-anggota-tim)
- [Deskripsi & Metodologi Sistem](#-deskripsi--metodologi-sistem)
- [Basis Aturan Fuzzy (Fuzzy Rules)](#-basis-aturan-fuzzy-fuzzy-rules)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Struktur Repositori](#-struktur-repositori)
- [Cara Menjalankan Proyek](#-cara-menjalankan-proyek)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)

---

## 📌 Konteks Proyek

Proyek ini merupakan **Final Project (Tugas Besar)** untuk mata kuliah **Introduction to AI (Desain dan Kecerdasan Algoritma)**. Sistem ini bertujuan untuk memodelkan dinamika harga komoditas global guna memprediksi harga minyak goreng domestik di Indonesia, dengan membandingkan performa dua metode defuzzifikasi fuzzy logic yang paling umum digunakan dalam literatur: **Mamdani (Center of Gravity)** dan **Sugeno (Weighted Average)**.

## 👥 Anggota Tim

| Nama | NIM |
|---|---|
| Dimas Raditya Putra Handoko | 103012440016 |
| Louis Ashton Yang | 103012440021 |
| Muhammad Gavin Satrio Prabaswara | 103012440005 |

## 🧠 Deskripsi & Metodologi Sistem

### Dataset

Sistem ini dilatih dan dievaluasi menggunakan dataset historis gabungan sebanyak **32.779 baris data**, yang mencakup **34 provinsi di Indonesia** untuk periode **2022–2024**. Dataset dibentuk dengan menggabungkan (merge) harga minyak goreng harian per provinsi (format *wide*, diubah ke format *long*) dengan tiga dataset komoditas global berdasarkan tanggal transaksi.

### Variabel Sistem

| Jenis | Variabel | Satuan | Rentang Nilai | Label Linguistik |
|---|---|---|---|---|
| Input | Harga Palm Oil Futures | MYR/MT | 3.000 – 8.500 | Rendah · Sedang · Tinggi |
| Input | Perubahan Harian Palm Oil | % | -11 – +12 | Turun · Stabil · Naik |
| Input | Harga Crude Oil WTI | USD/bbl | 60 – 125 | Rendah · Sedang · Tinggi |
| Input | Perubahan Harian Crude Oil | % | -13 – +10 | Turun · Stabil · Naik |
| Input | Harga US Sugar #11 | USD ¢/lb | 16 – 30 | Rendah · Sedang · Tinggi |
| **Output** | **Harga Minyak Goreng** | **IDR/liter** | **13.000 – 37.000** | Sangat Rendah · Rendah · Sedang · Tinggi · Sangat Tinggi |

Fungsi keanggotaan (*membership function*) dibangun menggunakan kombinasi kurva **trapesium (trapezoidal)** dan **segitiga (triangular)**:

**Fungsi keanggotaan segitiga:**

$$\mu(x) = \max\left(\min\left(\frac{x-a}{b-a},\ \frac{c-x}{c-b}\right),\ 0\right)$$

**Fungsi keanggotaan trapesium:**

$$\mu(x) = \max\left(\min\left(\frac{x-a}{b-a},\ 1,\ \frac{d-x}{d-c}\right),\ 0\right)$$

### Fuzzy Inference: Mamdani vs. Sugeno

Kedua metode menggunakan tahap **fuzzifikasi** dan **inferensi (operator AND / min)** yang identik, namun berbeda pada tahap **agregasi** dan **defuzzifikasi**.

#### 1. Metode Mamdani — *Center of Gravity* (CoG)

Pada metode Mamdani, setiap aturan yang aktif menghasilkan area fuzzy output yang dipotong (*clipped*) sesuai *firing strength*-nya, kemudian seluruh area digabungkan (agregasi max). Nilai akhir (*crisp output*) dihitung dari titik pusat massa (centroid) area gabungan tersebut:

$$z^{*} = \frac{\displaystyle\int z \cdot \mu_A(z)\, dz}{\displaystyle\int \mu_A(z)\, dz} \;\approx\; \frac{\displaystyle\sum_{i=1}^{n} z_i \cdot \mu_A(z_i)}{\displaystyle\sum_{i=1}^{n} \mu_A(z_i)}$$

> ✅ Intuitif dan mudah diinterpretasikan secara linguistik.
> ❌ Beban komputasi lebih berat karena membutuhkan integrasi/aproksimasi numerik atas seluruh rentang output.

#### 2. Metode Sugeno — *Weighted Average*

Pada metode Sugeno, setiap aturan menghasilkan output berupa nilai **singleton** (konstanta), bukan bentuk kurva fuzzy. Nilai akhir dihitung sebagai rata-rata berbobot dari seluruh singleton, dengan bobot berupa *firing strength* masing-masing aturan:

$$z^{*} = \frac{\displaystyle\sum_{i=1}^{n} w_i \cdot z_i}{\displaystyle\sum_{i=1}^{n} w_i}$$

dengan $w_i$ adalah *firing strength* (derajat keanggotaan minimum dari seluruh anteseden) aturan ke-$i$, dan $z_i$ adalah nilai singleton output dari aturan ke-$i$.

> ✅ Komputasi jauh lebih efisien serta akurasi numerik cenderung lebih baik.
> ❌ Kurang intuitif secara linguistik karena output berupa nilai konstan, bukan kurva fuzzy.

## 📋 Basis Aturan Fuzzy (Fuzzy Rules)

Sistem menggunakan basis **20 aturan fuzzy (IF–THEN rules)** yang merepresentasikan hubungan harga dasar antar komoditas, momentum pasar, dan sinyal inflasi terhadap harga minyak goreng.

| # | Palm Price | Palm Δ% | Crude Price | Crude Δ% | Sugar Price | → Output |
|---|---|---|---|---|---|---|
| 1 | Rendah | Stabil | Rendah | Stabil | Rendah | Sangat Rendah |
| 2 | Rendah | Stabil | Sedang | Stabil | Rendah | Rendah |
| 3 | Sedang | Stabil | Sedang | Stabil | Sedang | Sedang |
| 4 | Tinggi | Stabil | Sedang | Stabil | Sedang | Tinggi |
| 5 | Tinggi | Naik | Tinggi | Naik | Tinggi | Sangat Tinggi |
| 6 | Sedang | Naik | Sedang | Naik | Sedang | Tinggi |
| 7 | Sedang | Turun | Rendah | Turun | Rendah | Rendah |
| 8 | Rendah | Turun | Rendah | Turun | Rendah | Sangat Rendah |
| 9 | Sedang | Naik | Sedang | Stabil | Sedang | Sedang |
| 10 | Tinggi | Naik | Tinggi | Naik | Tinggi | Sangat Tinggi |
| 11 | Tinggi | Stabil | Sedang | Stabil | Tinggi | Sangat Tinggi |
| 12 | Rendah | Stabil | Rendah | Stabil | Rendah | Sangat Rendah |
| 13 | Sedang | Stabil | Tinggi | Stabil | Sedang | Tinggi |
| 14 | Sedang | Stabil | Rendah | Stabil | Tinggi | Sedang |
| 15 | Rendah | Turun | Rendah | Turun | Rendah | Sangat Rendah |
| 16 | Sedang | Stabil | Tinggi | Naik | Rendah | Tinggi |
| 17 | Rendah | Stabil | Tinggi | Stabil | Sedang | Sedang |
| 18 | Tinggi | Stabil | Rendah | Stabil | Sedang | Sedang |
| 19 | Sedang | Naik | Sedang | Naik | Tinggi | Sangat Tinggi |
| 20 | Rendah | Turun | Sedang | Stabil | Rendah | Rendah |

## 📊 Hasil Evaluasi

Evaluasi dilakukan pada `pilot_1_.ipynb` menggunakan **3.000 sampel acak** dari dataset, dengan membandingkan harga hasil prediksi terhadap harga minyak goreng aktual (`MG_Price`) menggunakan metrik **MAE**, **RMSE**, dan **MAPE**.

| Metrik | Mamdani (CoG) | Sugeno (Weighted Average) | Metode Terbaik |
|---|---|---|---|
| MAE | Rp 4.643 | Rp 2.660 | ✅ Sugeno |
| RMSE | Rp 5.568 | Rp 3.308 | ✅ Sugeno |
| MAPE | 26,04% | 14,27% | ✅ Sugeno |

> Secara umum, metode **Sugeno** menghasilkan galat prediksi yang lebih kecil dibanding **Mamdani** pada dataset ini, sejalan dengan sifatnya yang lebih presisi secara numerik. Namun demikian, **Mamdani** tetap unggul dari sisi interpretasi linguistik atas hasil prediksi.

## 📂 Struktur Repositori

```
📦 cooking-oil-price-prediction-fuzzy-logic
├── 📄 app.py                                     # Dashboard interaktif Streamlit (fuzzy engine + UI)
├── 📓 pilot_1_.ipynb                             # Notebook eksperimen: data, membership function,
│                                                  #   visualisasi, dan evaluasi MAE/RMSE/MAPE
├── 📊 Minyak_Goreng_Kemasan_Sederhana.csv         # Dataset harga minyak goreng (34 provinsi, output)
├── 📊 Palm_Oil_Futures_Historical_Data.csv        # Dataset harga Palm Oil Futures (input)
├── 📊 Crude_Oil_WTI_Futures_Historical_Data.csv   # Dataset harga Crude Oil WTI (input)
├── 📊 US_Sugar_11_Futures_Historical_Data.csv     # Dataset harga US Sugar #11 (input)
├── ⚙️ .streamlit/
│   └── config.toml                               # Konfigurasi tema (warna, font) dashboard Streamlit
├── 🚫 .gitignore                                  # Daftar file/folder yang diabaikan Git
└── 📘 README.md                                   # Dokumentasi proyek
```

> **Catatan:** Agar tema kustom pada `config.toml` terbaca otomatis oleh Streamlit, pastikan file tersebut diletakkan di dalam folder `.streamlit/` pada root repositori.

## 🚀 Cara Menjalankan Proyek

### 1. Clone Repositori

```bash
git clone <url-repositori-anda>
cd cooking-oil-price-prediction-fuzzy-logic
```

### 2. Buat & Aktifkan Virtual Environment (opsional, direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Instalasi Dependencies

```bash
pip install pandas numpy matplotlib streamlit jupyter
```

### 4. Menjalankan Eksperimen (Jupyter Notebook)

Gunakan notebook berikut untuk eksplorasi data, membership function, visualisasi, hingga evaluasi model:

```bash
jupyter notebook pilot_1_.ipynb
```

### 5. Menjalankan Dashboard Interaktif (Streamlit)

Setelah eksperimen selesai, jalankan dashboard prediksi harga minyak goreng secara interaktif:

```bash
streamlit run app.py
```

Dashboard akan otomatis terbuka di browser pada `http://localhost:8501`, menampilkan 4 tab utama: **Dashboard** (prediksi & metrik), **Visualizations** (kurva membership function & agregasi output), **Rule Base** (20 aturan fuzzy beserta status aktivasinya), dan **Methodology** (penjelasan metode Mamdani vs Sugeno).

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| **Python 3** | Bahasa pemrograman utama |
| **Pandas** | Manipulasi dan penggabungan dataset |
| **NumPy** | Komputasi numerik (fuzzifikasi, defuzzifikasi) |
| **Matplotlib** | Visualisasi membership function & hasil evaluasi |
| **Streamlit** | Dashboard web interaktif |
| **Jupyter Notebook** | Lingkungan eksperimen dan analisis data |

---

<p align="center">
  <sub>Disusun untuk memenuhi Final Project mata kuliah <b>Introduction to AI (Desain dan Kecerdasan Algoritma)</b></sub>
</p>
