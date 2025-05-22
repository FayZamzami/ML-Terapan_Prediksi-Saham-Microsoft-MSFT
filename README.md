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

Dalam proyek prediksi harga saham Microsoft ini, evaluasi model dilakukan dengan menggunakan beberapa metrik yang relevan untuk regresi dan prediksi time series:

### Metrik Evaluasi yang Digunakan

1. **RMSE (Root Mean Squared Error)**
   - **Definisi**: Akar kuadrat dari rata-rata error kuadrat antara nilai prediksi dan nilai aktual.
   - **Formula**: $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
   - **Interpretasi**: Nilai yang lebih rendah menunjukkan akurasi prediksi yang lebih baik.
   - **Mengapa digunakan**: Memberikan bobot lebih pada error yang besar, sangat penting untuk prediksi finansial di mana kesalahan besar bisa berakibat fatal.

2. **MAE (Mean Absolute Error)**
   - **Definisi**: Rata-rata dari nilai absolut selisih antara nilai prediksi dan nilai aktual.
   - **Formula**: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
   - **Interpretasi**: Nilai yang lebih rendah menunjukkan akurasi prediksi yang lebih baik.
   - **Mengapa digunakan**: Mengukur error rata-rata tanpa memberi bobot lebih pada outlier.

3. **MAPE (Mean Absolute Percentage Error)**
   - **Definisi**: Rata-rata dari persentase absolut error.
   - **Formula**: $MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$
   - **Interpretasi**: Nilai yang lebih rendah menunjukkan akurasi prediksi yang lebih baik, dalam bentuk persentase.
   - **Mengapa digunakan**: Memungkinkan perbandingan error di skala yang berbeda dan lebih mudah diinterpretasikan.

4. **R² (Coefficient of Determination)**
   - **Definisi**: Proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh variabel independen.
   - **Formula**: $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$
   - **Interpretasi**: Nilai antara 0 dan 1, di mana nilai lebih tinggi menunjukkan model yang lebih baik.
   - **Mengapa digunakan**: Menunjukkan seberapa baik model menjelaskan variabilitas data.

5. **Directional Accuracy**
   - **Definisi**: Persentase prediksi yang benar mengenai arah perubahan harga (naik/turun).
   - **Formula**: Persentase kasus di mana (y_t+1 - y_t) dan (ŷ_t+1 - y_t) memiliki tanda yang sama.
   - **Interpretasi**: Nilai lebih tinggi menunjukkan model lebih baik dalam memprediksi arah pergerakan harga.
   - **Mengapa digunakan**: Penting untuk strategi trading di mana arah pergerakan bisa lebih penting daripada nilai absolutnya.

### Hasil Evaluasi Model Terbaik (Linear Regression)

Berdasarkan hasil evaluasi pada dataset testing, model Linear Regression menunjukkan performa sebagai berikut:

- **RMSE**: 0.0110
- **MAPE**: 1.09%
- **Directional Accuracy**: 47.44%
- **R²**: 0.9973

Nilai RMSE yang rendah (0.0110) dan R² yang tinggi (0.9973) menunjukkan bahwa model memiliki tingkat akurasi yang sangat baik dalam memprediksi nilai harga penutupan saham Microsoft. MAPE sebesar 1.09% juga mengkonfirmasi bahwa error prediksi secara persentase sangat kecil.

Namun, Directional Accuracy sebesar 47.44% menunjukkan bahwa model hanya sedikit lebih baik dari acak dalam memprediksi arah pergerakan harga (naik/turun). Hal ini mengindikasikan bahwa meskipun model sangat baik dalam memprediksi nilai absolut harga saham, model masih memiliki keterbatasan dalam memprediksi arah pergerakannya.

### Kesimpulan Evaluasi

Model Linear Regression terbukti menjadi pilihan terbaik untuk prediksi harga saham Microsoft dengan akurasi yang tinggi dalam hal nilai absolut (RMSE rendah, R² tinggi). Namun, untuk aplikasi trading yang membutuhkan prediksi arah yang akurat, model ini masih perlu dikembangkan lebih lanjut.

Untuk pengembangan selanjutnya, peningkatan directional accuracy bisa menjadi fokus utama, misalnya dengan menambahkan fitur teknikal yang lebih spesifik untuk menangkap sinyal perubahan arah atau dengan menggunakan model hybrid yang mengkombinasikan prediksi nilai dan prediksi arah.
