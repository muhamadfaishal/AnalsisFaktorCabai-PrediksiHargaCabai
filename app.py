# ==============================================================================
#                 APLIKASI STREAMLIT GABUNGAN
#       Analisis Faktor (Random Forest) & Prediksi Harga (LSTM)
# ==============================================================================

# --- Mempersiapkan Perkakas ---
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import plot_tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os
import random
import io

# ==============================================================================
#                         KONFIGURASI HALAMAN & JUDUL
# ==============================================================================
st.set_page_config(
    page_title="Analisis & Prediksi Harga Cabai",
    page_icon="🌶️",
    layout="wide"
)

st.title("🌶️ Analisis Faktor dan Prediksi Harga Cabai")
st.write("""
**Oleh: Muhamad Faishal Khalfani | NRP: 152021170**

Aplikasi ini merupakan implementasi dari penelitian tugas akhir yang berjudul: **"Analisis Faktor-Faktor yang Mempengaruhi Harga Cabai dan Prediksi Harga Menggunakan Random Forest dan LSTM"**.
Gunakan navigasi di sidebar untuk memilih metode analisis yang ingin Anda jalankan.
""")

# ==============================================================================
#                              NAVIGASI SIDEBAR
# ==============================================================================
st.sidebar.title("⚙️ Menu Navigasi")
analysis_choice = st.sidebar.radio(
    "Pilih Metode Analisis:",
    (
        "Analisis Faktor Pengaruh (Random Forest)",
        "Prediksi Runtut Waktu (LSTM)"
    )
)
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini menggabungkan dua metode untuk memberikan analisis yang komprehensif.")


# ==============================================================================
#             BAGIAN 1: ANALISIS FAKTOR (RANDOM FOREST) - REVISI
# ==============================================================================
def run_random_forest_analysis():
    """Fungsi untuk menjalankan seluruh pipeline analisis Random Forest."""

    st.header("🌳 Analisis Faktor Pengaruh Menggunakan Random Forest")
    st.write("""
    Metode **Random Forest** digunakan untuk menganalisis dan memeringkat faktor-faktor mana yang paling signifikan dalam mempengaruhi fluktuasi harga cabai. Metode ini unggul dalam menangani hubungan non-linear dan memberikan skor kepentingan (importance score) untuk setiap variabel.
    """)

    if st.button("🚀 Mulai Analisis Faktor", key="rf_button", use_container_width=True):

        # --- Langkah 1-3: Memuat Data dan Mendefinisikan Variabel ---
        with st.expander("Langkah 1-3: Memuat Data & Mendefinisikan Variabel", expanded=True):
            try:
                st.write(f"Memuat data dari `gabungan_dataset_final.xlsx`...")
                df = pd.read_excel('gabungan_dataset_final.xlsx')
                df.dropna(inplace=True)

                y = df['Harga']
                independent_vars = [
                    'RR', 'is_idul_fitri_period', 'is_idul_adha_period', 'is_nataru_period',
                    'is_payday_week', 'Hari_Senin', 'Hari_Selasa', 'Hari_Rabu', 'Hari_Kamis',
                    'Hari_Jumat', 'Hari_Sabtu', 'Hari_Minggu', 'Inflasi'
                ]
                X = df[independent_vars]
                st.success("✅ Data berhasil dimuat dan variabel didefinisikan.")
                st.dataframe(df.head())

            except Exception as e:
                st.error(f"❌ GAGAL: Terjadi error saat memuat data. Error: {e}")
                return

        # --- Langkah 4: Membagi Data ---
        with st.expander("Langkah 4: Membagi Data menjadi Data Latih dan Data Uji", expanded=True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.info(f"-> Jumlah data latih : **{len(X_train)} baris** (80%)")
            st.info(f"-> Jumlah data uji   : **{len(X_test)} baris** (20%)")
            st.success("✅ Data telah berhasil dibagi.")

        # --- Langkah 5: Melatih Model ---
        with st.expander("Langkah 5: Melatih Model Random Forest", expanded=True):
            with st.spinner("Model sedang dilatih menggunakan data latih..."):
                model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model_rf.fit(X_train, y_train)
            st.success("✅ Model telah selesai dilatih.")

        # --- Langkah 6: Menguji Model pada Data Uji (Menghitung Error) ---
        with st.expander("Langkah 6: Menguji Performa Model & Menghitung Error", expanded=True):
            predictions = model_rf.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, predictions)

            st.subheader("Tingkat Akurasi Model pada Data Uji:")
            st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2%}",
                      help="Rata-rata persentase error prediksi. Semakin rendah semakin baik.")
            st.success("✅ Pengujian performa model selesai.")

        # --- Langkah 7 & 8: Menganalisis & Membuat Grafik Faktor Penting ---
        with st.expander("Langkah 7 & 8: Menganalisis & Membuat Grafik Faktor Penting", expanded=True):
            importances = model_rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Faktor': independent_vars,
                'Skor Kepentingan': importances
            }).sort_values(by='Skor Kepentingan', ascending=False)

            # >> REVISI: Menambahkan kolom persentase
            feature_importance_df['Kontribusi (%)'] = feature_importance_df['Skor Kepentingan'] * 100

            st.subheader("Tabel Peringkat Faktor Paling Berpengaruh:")
            # >> REVISI: Menampilkan kolom persentase yang sudah diformat
            st.dataframe(
                feature_importance_df[['Faktor', 'Kontribusi (%)']].style.format({'Kontribusi (%)': '{:.2f}%'}))

            st.subheader("Grafik Peringkat Faktor:")
            fig, ax = plt.subplots(figsize=(10, 6))
            # >> REVISI: Menggunakan kolom persentase untuk grafik
            sns.barplot(x='Kontribusi (%)', y='Faktor', data=feature_importance_df, palette='viridis', ax=ax)
            ax.set_title('Peringkat Faktor yang Mempengaruhi Harga Cabai', fontsize=16, fontweight='bold')
            ax.set_xlabel('Kontribusi Kepentingan (%)', fontsize=12)  # Label sumbu-x diubah
            ax.set_ylabel('Faktor-Faktor', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            st.success("✅ Analisis dan visualisasi faktor selesai.")

        # ... Sisa kode (langkah 9 & 10) ...
        with st.expander("Langkah 9: Visualisasi Contoh Pohon Keputusan", expanded=False):
            st.write(
                "Di bawah ini adalah visualisasi dari **satu pohon acak** dari 100 pohon yang ada di dalam model 'hutan'.")
            fig_tree, ax_tree = plt.subplots(figsize=(30, 15))
            contoh_pohon = model_rf.estimators_[5]
            plot_tree(
                contoh_pohon,
                feature_names=independent_vars,
                filled=True,
                rounded=True,
                max_depth=2,
                fontsize=12,
                precision=0,
                ax=ax_tree
            )
            ax_tree.set_title("Contoh Visualisasi Satu Pohon Keputusan dari Random Forest", fontsize=20)
            st.pyplot(fig_tree)

        # --- Langkah 10: Kesimpulan Akhir ---
        st.subheader("🎯 Kesimpulan Analisis Random Forest")
        faktor_teratas = feature_importance_df.iloc[0]['Faktor']
        # >> REVISI: Mengambil nilai dari kolom persentase
        kontribusi_teratas = feature_importance_df.iloc[0]['Kontribusi (%)']

        st.markdown(f"""
        Berdasarkan analisis, faktor yang paling berpengaruh adalah **'{faktor_teratas}'** dengan kontribusi sebesar **{kontribusi_teratas:.2f}%**.
        Model ini mampu menjelaskan hubungan antar faktor dengan tingkat akurasi (MAPE) sebesar **{mape:.2%}** pada data uji.
        """)


# ==============================================================================
#             BAGIAN 2: PREDIKSI RUNTUT WAKTU (LSTM)
# ==============================================================================
def run_lstm_prediction():
    # ... (Kode untuk fungsi LSTM tetap sama seperti sebelumnya, tidak ada perubahan) ...
    st.header("📈 Prediksi Runtut Waktu Menggunakan LSTM")
    st.write("""
    Metode **Long Short-Term Memory (LSTM)** adalah jenis jaringan syaraf tiruan yang dirancang khusus untuk data sekuensial atau runtut waktu (*time series*). LSTM digunakan di sini untuk memprediksi pergerakan harga di masa depan berdasarkan pola historis dari harga itu sendiri dan fitur lainnya.
    """)

    if st.button("🚀 Mulai Proses Prediksi", key="lstm_button", use_container_width=True):
        st.warning(
            "Proses ini akan melatih ulang model LSTM dari awal dan mungkin memakan waktu beberapa menit. Harap bersabar.",
            icon="⏳")

        # --- Langkah 0: Pengaturan Seed ---
        with st.expander("Langkah 0: Pengaturan Seed", expanded=True):
            seed_value = 42
            os.environ['PYTHONHASHSEED'] = str(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)
            tf.random.set_seed(seed_value)
            st.success(f"✅ Random seed telah diatur ke **{seed_value}** untuk hasil yang konsisten.")

        # --- Langkah 1-2: Muat Data (disesuaikan untuk LSTM) ---
        with st.expander("Langkah 1-2: Memuat & Mempersiapkan Data Time Series", expanded=True):
            try:
                df = pd.read_excel('gabungan_dataset_final.xlsx')
                if 'tanggal' in df.columns:
                    df.rename(columns={'tanggal': 'Tanggal'}, inplace=True)
                df['Tanggal'] = pd.to_datetime(df['Tanggal'])
                df.set_index('Tanggal', inplace=True)
                df.sort_index(inplace=True)
                df.dropna(inplace=True)
                st.success("✅ Data berhasil dimuat.")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ GAGAL: Terjadi error. Error: {e}")
                return

        # --- Langkah 3: Normalisasi ---
        with st.expander("Langkah 3: Normalisasi Data", expanded=True):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df)
            df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            st.success("✅ Data berhasil dinormalisasi ke rentang 0-1.")
            st.dataframe(df_scaled.head())

        # --- Langkah 4: Membuat Sekuens ---
        with st.expander("Langkah 4: Membuat Sekuens Data untuk LSTM", expanded=True):
            timesteps = 30
            training_size = int(len(df_scaled) * 0.8)
            train_data = df_scaled[:training_size]
            test_data = df_scaled[training_size:]

            def create_sequences(data, timesteps):
                X, y = [], []
                target_col_index = data.columns.get_loc('Harga')
                for i in range(len(data) - timesteps):
                    X.append(data.iloc[i:(i + timesteps)].values)
                    y.append(data.iloc[i + timesteps, target_col_index])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train_data, timesteps)
            X_test, y_test = create_sequences(test_data, timesteps)
            st.success(f"✅ Data sekuens dengan `timesteps={timesteps}` berhasil dibuat.")
            st.code(f"Bentuk X_train: {X_train.shape}\nBentuk y_train: {y_train.shape}", language="text")

        # --- Langkah 5: Membangun Model LSTM ---
        with st.expander("Langkah 5: Membangun Arsitektur Model LSTM", expanded=True):
            model = Sequential([
                LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.1),
                LSTM(units=32),
                Dense(units=1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            st.code("\n".join(stringlist), language="text")
            st.success("✅ Arsitektur model LSTM berhasil dibangun.")

        # --- Langkah 6: Melatih Model ---
        with st.expander("Langkah 6: Melatih Model LSTM", expanded=True):
            with st.spinner("Model LSTM sedang dilatih (maks. 200 epoch)..."):
                early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                history = model.fit(
                    X_train, y_train,
                    epochs=200,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=0
                )
            st.success("✅ Model LSTM berhasil dilatih.")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(history.history['loss'], label='Loss Training')
            ax.plot(history.history['val_loss'], label='Loss Validasi')
            ax.set_title('Grafik Model Loss Selama Training', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # --- Langkah 7: Forecasting & Validasi ---
        with st.expander("Langkah 7: Forecasting dan Validasi Akhir", expanded=True):
            with st.spinner("Melakukan forecasting 30 hari ke depan dan validasi..."):
                last_sequence = df_scaled.values[-timesteps:]
                current_sequence = last_sequence.reshape(1, timesteps, df_scaled.shape[1])
                future_predictions_scaled = []
                for _ in range(30):
                    pred = model.predict(current_sequence, verbose=0)[0]
                    future_predictions_scaled.append(pred)
                    new_row = np.append(pred, last_sequence[-1, 1:])
                    new_row_reshaped = new_row.reshape(1, 1, df_scaled.shape[1])
                    current_sequence = np.append(current_sequence[:, 1:, :], new_row_reshaped, axis=1)

                dummy_array = np.zeros((len(future_predictions_scaled), df.shape[1]))
                dummy_array[:, 0] = np.array(future_predictions_scaled).flatten()
                future_predictions_actual = scaler.inverse_transform(dummy_array)[:, 0]
                forecast_df = pd.DataFrame({'Harga Prediksi': future_predictions_actual},
                                           index=pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30))

                df_aktual_januari = pd.read_excel('harga_cabai_januari_2025_cleaned.xlsx')
                df_aktual_januari['Tanggal'] = pd.to_datetime(df_aktual_januari['Tanggal'])
                df_aktual_januari.set_index('Tanggal', inplace=True)
                comparison_df = forecast_df.join(df_aktual_januari, how='inner').dropna()
                comparison_df.rename(columns={'Harga': 'Harga Aktual'}, inplace=True)

            st.success("✅ Forecasting dan validasi selesai.")
            st.subheader("Tabel Perbandingan Prediksi vs Aktual")
            st.dataframe(comparison_df.style.format({'Harga Prediksi': "Rp {:,.0f}", 'Harga Aktual': "Rp {:,.0f}"}))

            y_true = comparison_df['Harga Aktual']
            y_pred = comparison_df['Harga Prediksi']
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape_lstm = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            st.subheader("Metrik Evaluasi Prediksi")
            col1, col2 = st.columns(2)
            col1.metric("RMSE Prediksi", f"Rp {rmse:,.2f}")
            col2.metric("MAPE Prediksi", f"{mape_lstm:.2f}%")

            st.subheader("Grafik Validasi Prediksi")
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(df.index[-90:], df['Harga'][-90:], label='Data Historis')
            ax.plot(comparison_df.index, y_pred, label='Hasil Prediksi (Forecast)', color='red', linestyle='--')
            ax.plot(comparison_df.index, y_true, label='Harga Aktual', color='green')
            ax.set_title('Validasi Prediksi LSTM vs Harga Aktual', fontsize=16, fontweight='bold')
            ax.set_ylabel('Harga (Rp)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)


# ==============================================================================
#                      LOGIKA UTAMA UNTUK MENJALANKAN APP
# ==============================================================================
if analysis_choice == "Analisis Faktor Pengaruh (Random Forest)":
    run_random_forest_analysis()
elif analysis_choice == "Prediksi Runtut Waktu (LSTM)":
    run_lstm_prediction()