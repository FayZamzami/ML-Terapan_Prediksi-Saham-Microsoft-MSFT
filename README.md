# Prediksi Harga Saham Microsoft (MSFT) - Analisis Data dan Machine Learning

## Domain Proyek
### Latar Belakang

Pasar saham merupakan salah satu instrumen investasi yang menarik banyak minat dari investor, baik pemula maupun profesional. Microsoft Corporation (MSFT) sebagai salah satu perusahaan teknologi terbesar di dunia memiliki pergerakan saham yang menjadi perhatian banyak investor. Prediksi pergerakan harga saham yang akurat dapat memberikan keuntungan kompetitif bagi investor dalam mengambil keputusan investasi yang lebih baik.

Melalui penggunaan machine learning dan analisis data, investor dapat memanfaatkan pola historis untuk memprediksi tren harga di masa depan. Prediksi menggunakan teknik machine learning bertujuan untuk memberikan gambaran yang lebih objektif tentang pergerakan harga saham Microsoft di masa mendatang.

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi harga saham Microsoft di masa depan dengan akurasi yang tinggi berdasarkan data historis?
- Indikator teknikal dan faktor apa saja yang paling berpengaruh dalam pergerakan harga saham Microsoft?

### Goals
- Mengembangkan model prediksi yang mampu memprediksi harga saham Microsoft dengan tingkat error yang minimal.
- Mengidentifikasi indikator teknikal yang memiliki korelasi kuat dengan pergerakan harga saham Microsoft.

### Solution Statements
- Membangun beberapa model machine learning untuk prediksi harga saham Microsoft, termasuk Linear Regression, Random Forest, XGBoost, Gradient Boosting, dan LSTM.
- Melakukan feature engineering untuk menciptakan indikator teknikal yang relevan dan menganalisis pengaruhnya terhadap harga saham Microsoft.

## Data Understanding
Dataset yang digunakan berisi data historis harga saham Microsoft dari tahun 1986 hingga 2025. Dataset ini mencakup informasi harian tentang harga pembukaan, tertinggi, terendah, penutupan, serta volume perdagangan.
Dataset dapat di dapat dan di akses dari [kaggle](https://www.kaggle.com/datasets/muhammadatiflatif/complete-microsoft-stock-dataset-19862025?resource=download) sesuai dengan [Yahoo Finance](https://finance.yahoo.com/quote/MSFT/history/?period1=511108200&period2=1747802492) yang merupakan sumber terpercaya untuk data pasar saham.

- Kaggle : https://www.kaggle.com/datasets/muhammadatiflatif/complete-microsoft-stock-dataset-19862025?resource=download
- Yahoo Finance : https://finance.yahoo.com/quote/MSFT/history/?period1=511108200&period2=1747802492

**Informasi Dataset:**
- **Sumber:** Yahoo Finance / Dataset Historis
- **Jumlah Baris:** 9.868 entri (harian)
- **Periode:** 13 Maret 1986 - 12 Mei 2025
- **Format File:** CSV

### Variabel-variabel pada dataset:
- **date**: Tanggal perdagangan
- **open**: Harga pembukaan saham pada hari tersebut
- **high**: Harga tertinggi saham pada hari tersebut
- **low**: Harga terendah saham pada hari tersebut
- **close**: Harga penutupan saham pada hari tersebut
- **adj_close**: Harga penutupan yang telah disesuaikan (untuk dividen, stock split, dll)
- **volume**: Volume perdagangan (jumlah saham yang diperdagangkan)

## Data Preparation
### Penanganan Missing Value
Dilakukan pemeriksaan terhadap missing value pada dataset dan ditemukan beberapa nilai yang hilang. Untuk mengatasi ini, dilakukan penghapusan baris dengan nilai yang hilang untuk memastikan kualitas data.

### Penanganan Outlier
Outlier pada data saham dapat merepresentasikan momen penting dalam pasar, namun terlalu banyak outlier dapat mengganggu performa model. Penanganan outlier dilakukan dengan metode Winsorization, yang menggantikan nilai ekstrem dengan batas atas dan bawah (Q1 - 3*IQR dan Q3 + 3*IQR).

```python
# Menggunakan IQR method untuk semua fitur numerik utama
for feature in numerical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter outlier lebih konservatif (3 IQR) untuk data keuangan
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Winsorization sebagai alternatif menghapus outlier
    df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])
```

### Feature Engineering
Untuk meningkatkan kemampuan prediksi model, dilakukan feature engineering dengan menambahkan berbagai indikator teknikal:

1. **Lag Features (1-5 hari)**: Menggunakan nilai harga dan volume dari 1-5 hari sebelumnya
2. **Moving Averages**: Menghitung rata-rata bergerak dengan window 7, 14, dan 30 hari
3. **Technical Indicators**:
   - **RSI (Relative Strength Index)**: Mengukur kecepatan dan perubahan pergerakan harga
   - **MACD (Moving Average Convergence Divergence)**: Mendeteksi perubahan momentum
   - **Bollinger Bands**: Mengukur volatilitas dan level relatif harga
4. **Volatility Features**: Menghitung volatilitas dan daily return
5. **Date-based Features**: Menambahkan fitur berdasarkan tanggal seperti hari dalam minggu, bulan, kuartal

### Normalisasi
Untuk memastikan semua fitur memiliki skala yang sama, dilakukan normalisasi menggunakan MinMaxScaler.

## Modeling
Dalam proyek ini, dilakukan perbandingan beberapa model machine learning untuk menemukan yang terbaik:

1. **Linear Regression**: Model paling sederhana yang mencari hubungan linear antara fitur dan target.
2. **Random Forest**: Ensemble model yang menggunakan multiple decision trees untuk meningkatkan akurasi dan mengurangi overfitting.
3. **XGBoost**: Implementasi gradient boosting yang dioptimasi untuk kinerja dan kecepatan.
4. **Gradient Boosting**: Ensemble model yang membangun model secara bertahap untuk meminimalkan error.
5. **LSTM (Long Short-Term Memory)**: Model neural network yang dirancang untuk data time series.

### Pemilihan Model Terbaik

Setelah melakukan evaluasi terhadap kelima model, Linear Regression terpilih sebagai model terbaik untuk prediksi harga saham Microsoft. Meskipun model ini adalah yang paling sederhana, tetapi menunjukkan performa terbaik dengan RMSE terendah dan R² tertinggi pada dataset validasi.

Linear Regression dipilih karena:

1. **Performa Unggul**: Model ini mencapai RMSE 0.0110 dan R² 0.9973 pada dataset testing, mengalahkan model yang lebih kompleks.

2. **Efisiensi Komputasi**: Memerlukan sumber daya komputasi yang jauh lebih kecil dibandingkan model kompleks seperti LSTM atau Random Forest.

3. **Interpretabilitas**: Koefisien model dapat diinterpretasikan secara langsung, memungkinkan pemahaman lebih baik tentang hubungan antara fitur dan harga saham.

4. **Generalisiasi**: Menunjukkan kapasitas generalisasi yang baik tanpa overfitting pada data training.

5. **Stabilitas**: Performa model tetap konsisten antara data validasi dan testing.

Meskipun model tree-based seperti Random Forest dan XGBoost biasanya lebih mampu menangkap hubungan non-linear dalam data, pergerakan harga saham Microsoft tampaknya memiliki komponen linear yang kuat yang dapat ditangkap dengan baik oleh Linear Regression, terutama setelah feature engineering yang komprehensif.

## Evaluation

Dalam proyek prediksi harga saham Microsoft ini, evaluasi model dilakukan dengan menggunakan beberapa metrik yang relevan untuk masalah regresi time series.

### Metrik Evaluasi yang Digunakan

1. **RMSE (Root Mean Squared Error)**
   - **Formula:** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
   - **Interpretasi:** Mengukur akar kuadrat dari rata-rata kesalahan kuadrat, memberikan penalti lebih besar pada error yang besar
   - **Mengapa digunakan:** Penting untuk prediksi finansial di mana kesalahan besar bisa berakibat fatal

2. **MAE (Mean Absolute Error)**
   - **Formula:** $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
   - **Interpretasi:** Rata-rata nilai absolut dari error, dalam unit yang sama dengan data asli (dollar)
   - **Mengapa digunakan:** Lebih mudah diinterpretasi dan tidak terpengaruh outlier sebesar RMSE

3. **MAPE (Mean Absolute Percentage Error)**
   - **Formula:** $\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$
   - **Interpretasi:** Error dalam bentuk persentase dari nilai aktual
   - **Mengapa digunakan:** Sesuai dengan perspektif investor yang sering berpikir dalam persentase

4. **R² (Coefficient of Determination)**
   - **Formula:** $\text{R}^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
   - **Interpretasi:** Proporsi variasi dalam data yang dapat dijelaskan oleh model
   - **Mengapa digunakan:** Menunjukkan kekuatan prediktif model secara keseluruhan

5. **Directional Accuracy**
   - **Interpretasi:** Persentase prediksi yang benar mengenai arah perubahan harga (naik/turun)
   - **Mengapa digunakan:** Penting untuk strategi trading yang berfokus pada arah pergerakan harga

### Hasil Evaluasi Model Terbaik (Linear Regression)

| Metrik | Nilai | Threshold | Interpretasi |
|--------|-------|-----------|--------------|
| RMSE | 0.0110 | < 0.05 (baik) | Error kuadrat rata-rata sangat rendah |
| MAE | 0.0058 | < 0.03 (baik) | Model rata-rata meleset sekitar 0.58% dari nilai sebenarnya |
| MAPE | 1.09% | < 5% (sangat baik) | Error rata-rata hanya 1.09% dari nilai aktual |
| R² | 0.9973 | > 0.9 (sangat baik) | Model menjelaskan 99.73% variabilitas data |
| Directional Accuracy | 47.44% | > 50% (baik) | Model tidak dapat memprediksi arah pergerakan secara akurat |

### Keterkaitan dengan Business Understanding

#### Analisis Terhadap Problem Statement

1. **Problem Statement 1: "Bagaimana cara memprediksi harga saham Microsoft di masa depan dengan akurasi yang tinggi berdasarkan data historis?"**
   
   Model Linear Regression yang dikembangkan berhasil menjawab problem statement ini dengan sangat baik, terbukti dari:
   - MAPE 1.09% yang jauh lebih baik dibandingkan metode prediksi tradisional (Moving Average: 3-5%, Naïve forecast: ~2%)
   - R² 0.9973 menunjukkan model dapat menjelaskan hampir seluruh variasi harga saham
   - Kombinasi preprocessing data dan feature selection yang tepat menghasilkan performa prediksi yang sangat akurat

2. **Problem Statement 2: "Indikator teknikal dan faktor apa saja yang paling berpengaruh dalam pergerakan harga saham Microsoft?"**
   
   Analisis feature importance mengungkapkan bahwa:
   - Harga penutupan sebelumnya (close) adalah prediktor paling kuat dengan koefisien 0.9993
   - Meskipun berbagai indikator teknikal (RSI, MACD, Bollinger Bands) diimplementasikan, model mendeteksi multikolinearitas tinggi di antara fitur-fitur ini
   - Harga hari sebelumnya terbukti sebagai prediktor dominan, konsisten dengan Efficient Market Hypothesis

#### Pencapaian Goals

1. **Goal 1: "Mengembangkan model prediksi yang mampu memprediksi harga saham Microsoft dengan tingkat error yang minimal."**
   
   Goal ini berhasil dicapai dengan sangat baik, dibuktikan oleh:
   - MAPE 1.09% (jauh di bawah threshold industri 5%)
   - RMSE 0.0110 menunjukkan error sangat rendah
   - Model konsisten performanya di berbagai periode pasar normal

2. **Goal 2: "Mengidentifikasi indikator teknikal yang memiliki korelasi kuat dengan pergerakan harga saham Microsoft."**
   
   Goal ini tercapai sebagian:
   - Analisis korelasi mengidentifikasi bahwa hampir semua indikator teknikal berkorelasi tinggi dengan harga saham
   - Namun, karena multikolinearitas ekstrem, feature selection menyaring semua indikator kecuali 'close'
   - Hal ini mengungkap insight penting bahwa dalam prediksi jangka pendek (1 hari), harga sebelumnya adalah prediktor terkuat

#### Efektivitas Solution Statements

1. **Solution Statement 1: "Membangun beberapa model machine learning untuk prediksi harga saham Microsoft..."**
   
   Solusi ini terbukti efektif:
   - Perbandingan 5 model berbeda memungkinkan pemilihan model optimal
   - Linear Regression mengungguli model kompleks dalam prediksi nilai absolut harga
   - Perbandingan sistematis mengungkap bahwa kompleksitas model tidak selalu berarti akurasi lebih baik

2. **Solution Statement 2: "Melakukan feature engineering untuk menciptakan indikator teknikal..."**
   
   Feature engineering berhasil:
   - 42 fitur teknikal berhasil dibuat dan dianalisis
   - Analisis korelasi dan multikolinearitas memberikan insight tentang hubungan antar indikator
   - Meskipun akhirnya hanya satu fitur yang digunakan, proses ini mengungkap sifat fundamental pasar saham

### Rekomendasi Pengembangan

Berdasarkan evaluasi komprehensif, beberapa rekomendasi untuk pengembangan model selanjutnya:

1. **Peningkatan Directional Accuracy**:
   - Mengintegrasikan data eksternal seperti sentimen pasar dan berita
   - Mengembangkan model khusus untuk klasifikasi arah pergerakan harga
   - Menerapkan ensemble methods yang fokus pada prediksi arah

2. **Peningkatan Ketahanan terhadap Volatilitas Ekstrem**:
   - Mengimplementasikan regime-switching models untuk beradaptasi dengan perubahan volatilitas
   - Mengembangkan early warning system untuk mendeteksi potensi volatilitas tinggi
   - Stratifikasi data training berdasarkan level volatilitas

3. **Eksplorasi Feature Engineering Lanjutan**:
   - Menambahkan fitur makroekonomi yang dapat mempengaruhi harga saham
   - Mengintegrasikan data sentimen dari media sosial dan berita
   - Memperluas time horizon untuk menangkap pola jangka panjang

Implementasi rekomendasi ini diharapkan dapat mengatasi keterbatasan utama model saat ini, terutama dalam memprediksi arah pergerakan harga dan meningkatkan ketahanan terhadap kondisi pasar ekstrem.
