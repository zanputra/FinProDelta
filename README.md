# **HOTEL BOOKING DEMAND ANALYSIS: Optimizing Revenue and Minimize Profit Loss**

**Tools:** Python<br>
**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)

**Outline:**

1. Business Problem Understanding
2. Data Understanding
3. Data Cleaning
4. EDA (Exploratory Data Analysis)
4. Pembuatan Model Machine Learning
7. Conclusion & Recommendation

---

# **Business Problem Understanding**

---

## Context

Hotel merupakan jenis bisnis hospitality yang dalam menjalankannya terdapat tantangan tertentu, yaitu adalah **kecenderungan customer untuk membatalkan reservasi**. Pada tahun 2022, shrgroup.com mengungkapkan secara global hotel megalami revenue loss sebesar 20% akibat pembatalan reservasi, terdapat peningkatan 33% dibandingkan tahun 2019 untuk periode yang sama.

**Pembatalan berarti hotel kehilangan pendapatan/revenue yang seharusnya bisa diperoleh**, hal ini berdampak pada revenue loss atau pencapaian revenue yang tidak maksimal. Revenue loss akan meningkat ketika sebuah hotel mengalami cancellation rate yang tinggi, sedangkan apabila sebuah hotel dapat meminimalisir cancellation rate maka potensi revenue maksimal dapat tercapai.

Ada banyak alasan atau faktor yang dapat menyebabkan adanya pembatalan reservasi tersebut, hal ini perlu ditangani oleh pihak hotel dengan mengetahui `customer mana yang cenderung membatalkan reservasi`, `memfokuskan target pemasaran` (fokus pada customer yang cenderung tidak membatalkan resrvasi), dan `menerapkan kebijakan pembatalan yang optimal dan peningkatan layanan` (targetted pada layanan hotel yang cenderung mendapatkan pembatalan booking).

Referensi:
https://shrgroup.com/2023/06/21/we-need-to-talk-about-cancellations/#:~:text=In%202019%2C%20we%20found%20that,2019%20for%20the%20same%20period

## Problem Statement

- **Untuk Machine Learning**: Hotel belum bisa memprediksi secara akurat customer yang membatalkan booking, sehingga penerapan strategi promosi untuk mencegah pembatalan (seperti diskon) belum tepat sasaran pada customer yang melakukan pembatalan. Hotel ingin mengetahui customer mana yang cenderung membatalkan booking untuk mencegah terjadinya kehilangan revenue.

- **Untuk Analisis Data**: Pemasaran dan kebijakan pembatalan booking masih belum sesuai target. Hotel ingin menentukan ulang target pemasaran, fokus pada customer yang cenderung tidak membatalkan booking serta menargetkan kebijakan pembatalan yang efisien pada customer yang cenderung membatalkan booking.

## Goals

- **Untuk machine learning**: Perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan seorang customer membatalkan booking atau tidak, sehingga dapat menerapkan strategi promosi tepat sasaran pada customer yang membatalkan booking (seperti diskon untuk mencegah customer membatalkan booking).

- **Untuk data analytics**: Perusahaan ingin mengetahui faktor/alasan yang membuat seorang customer membatalkan booking atau tidak, sehingga dapat menyusun kebijakan pembatalan yang optimal dan meningkatkan layanan hotel yang kurang sesuai dengan preferensi customer.

## Analytic Approach

**Melakukan EDA (Exploratory Data Analysis)**

- Mengetahui karakteristik customer dan hotel yang akan dijadikan fokus target pemasaran (customer yang tidak membatalkan reservasi).
- Mengetahui karakteristik customer dan hotel yang akan dijadikan fokus penerapan kebijakan pembatalan (customer yang membatalkan reservasi).
- Mengetahui demografi customer untuk menyesuaikan layanan yang perlu ditingkatkan oleh pihak hotel.
- Menghitung persentase opportunity cost berdasarkan adr untuk mengetahui adr yang hilang karena pembatalan booking.

**Melakukan Pembuatan Model Prediksi**

Membangun model klasifikasi yang akan membantu pihak hotel untuk dapat memprediksi kecenderungan customer membatalkan reservasi atau tidak, sehingga hotel dapat menerapkan strategi promosi untuk mencegah pembatalan yang sesuai target customer.

## Metrics Evaluation

- Type 1 error : `False Negative (FN)` (customer membatalkan reservasi, tetapi diprediksi tidak)

Konsekuensi: Kehilangan revenue potensial yang bisa didapatkan dari customer melakukan cancel, sehingga hotel menanggung **kerugian sebesar biaya kamar** yang dibooking. jika rata-rata harga sewa perkamar adalah 100 € maka kita akan kehilangan 100 EUR per tamu yang melakukan cancel

- Type 2 error : `False Positive (FP)` (customer tidak membatalkan reservasi, tetapi diprediksi membatalkan)

Konsekuensi: Strategi promosi untuk mencegah pembatalan tidak sesuai target, sehingga hotel harus menanggung **kerugian sebesar biaya promosi** yang telah ditentukan. jika biaya diskon untuk mengakuisisi kembali tamu yang akan cancel adalah 10 €, maka kita hanya akan kehilangan 10 EUR per tamu yang salah sasaran untuk diberi diskon.

Berdasarkan konsekuensinya, **kerugian sebesar biaya kamar (FN) > kerugian sebesar biaya promosi (FP)**, sehingga FN perlu dikurangi. Metric yang dapat mengurangi FN adalah RECALL, maka metric yang akan digunakan dalam pembuatan model prediksi pembatalan booking hotel adalah **RECALL**.

---

# **Data Understanding**

---

## Context

Raw data yang akan dilakukan cleaning:

https://github.com/zanputra/FinProDelta/blob/main/hotel_bookings.csv

Dataset ini terdiri dari **32 kolom** dan **119390 baris** yang setiap barisnya menjelaskan tentang karakteristik pemesan kamar hotel. Pada dataset terdapat satu kolom target atau label `is_canceled` yang menjelaskan apakah pemesan kamar membatalkan pesanannya atau tidak membatalkan pesanannya.

Target :

0 : Booking was not canceled (Reservasi tidak dibatalkan)

1 : Booking was canceled (Reservasi dibatalkan)

## Columns


| Attribute                      | Data Type  | Description |
|--------------------------------|------------|-------------|
| hotel                            | object      | Hotel (H1 = Resort Hotel, H2 = City Hotel) |
| is_canceled                      | int64       | Value indicating if the booking was canceled (1) or not (0) |
| lead_time                        | int64       | Number of days between the booking entry date and the arrival date |
| arrival_date_year                | int64       | Year of arrival date |
| arrival_date_month               | object      | Month of arrival date |
| arrival_date_week_number         | int64       | Week number of the year for arrival date |
| arrival_date_day_of_month        | int64       | Day of arrival date |
| stays_in_weekend_nights          | int64       | Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay |
| stays_in_week_nights             | int64       | Number of weeknights (Monday to Friday) the guest stayed or booked to stay |
| adults                           | int64       | Number of adults |
| children                         | float64     | Number of children |
| babies                           | int64       | Number of babies |
| meal                             | object      | Type of meal booked (e.g., SC, BB, HB, FB) |
| country                          | object      | Country of origin (ISO 3155–3:2013 format) |
| market_segment                   | object      | Market segment designation (e.g., TA = Travel Agents, TO = Tour Operators) |
| distribution_channel             | object      | Booking distribution channel (e.g., TA = Travel Agents, TO = Tour Operators) |
| is_repeated_guest                | int64       | Value indicating if the booking name was from a repeated guest (1) or not (0) |
| previous_cancellations           | int64       | Number of previous bookings canceled by the customer |
| previous_bookings_not_canceled   | int64       | Number of previous bookings not canceled by the customer |
| reserved_room_type               | object      | Code of room type reserved (anonymized) |
| assigned_room_type               | object      | Code of room type assigned (may differ from reserved) |
| booking_changes                  | int64       | Number of changes/amendments made to the booking |
| deposit_type                     | object      | Indication of deposit type (No Deposit, Non Refund, Refundable) |
| agent                            | float64     | ID of the travel agency that made the booking |
| company                          | float64     | ID of the company/entity responsible for booking payment |
| days_in_waiting_list             | int64       | Number of days the booking was on the waiting list before confirmation |
| customer_type                    | object      | Type of booking (Contract, Group, Transient, Transient-party) |
| adr                              | float64     | Average Daily Rate (ADR) calculated as the sum of lodging transactions divided by total staying nights |
| required_car_parking_spaces      | int64       | Number of car parking spaces required by the customer |
| total_of_special_requests        | int64       | Number of special requests made by the customer |
| reservation_status               | object      | Reservation last status (Canceled, Check-Out, No-Show) |
| reservation_status_date          | object      | Date when the last status was set |

---

# **Data Cleaning**

---

## Handling Missing Value

Terdapat beberapa kolom dengan missing value seperti pada kolom ***children, country, agent, dan company***
- Kita akan mengisi kolom ***children dan country*** dengan pendekatan statistik yaitu dengan mencari nilai yang paling sering muncul (modus).
- Sedangkan untuk kolom ***agent dan juga company*** kita akan drop kedua kolom tersebut karena kolom tersebut adalah keterangan nomor ID agen atau perusahaan yang terafiliasi dengan hotel untuk keperluan marketing dalam mencari tamu hotel dan tidak akan kita gunakan dalam proses analisa.
- Value "Undefined" pada kolom ***meal*** dapat dianggap missing value --> perlu dilakukan pengisian menggunakan modus.
- Value "Undefined" pada kolom ***distribution_channel*** dapat dianggap missing value --> perlu dilakukan pengisian menggunakan modus.
- Value "Undefined" pada kolom ***market_segment*** dapat dianggap missing value --> perlu dilakukan pengisian menggunakan modus.

## Removing Data Duplicates

Pada tahap ini kita akan mengecek baris yang memiliki duplikat satu sama lain. Karena pada dataset tidak ada kolom yang menunjukan informasi tentang ID atau sesuatu yang bersifat unique sebagai penanda antara tiap baris agar tidak terjadi duplikat, maka kita akan mengecek value pada seluruh kolom untuk tiap-tiap baris. Baris yang akan dihilangkan adalah baris yang memiliki data duplikat di seluruh kolomnya.

## Handling Irrelevant Values and White Spaces

Irrelevant value adalah data yang tidak benar atau tidak masuk akal, sehingga perlu dilakukan pengecekan apakah terdapat value data yang irrelevant, jika ada maka akan dilakukan penanganan, seperti:

- Data-data dengan kondisi ***adults*** = 0,***children*** dan ***babies*** != 0
- Pada kolom **is_repeated_guest** perlu dipastikan jika **previous_booking_not_cancel** == 0 maka **is_repeated_guest** harus == 0, karena tamu tersebut belum pernah sekalipun bermalam di hotel. Jika ditemukan value data **is_repeated_guest** == 0 tetapi kolom **previous_booking_not_cancel** != 0, maka kita harus mengubah kolom **is_repeated_guest** menjadi 1, karena tamu tersebut sudah pernah bermalam di hotel.

## Handling Irrelevant Columns

- ***arrival_date_week_number***: kolom ini menjelaskan minggu keberapa dalam satu tahun para tamu yang sudah booking akan tiba di hotel yang mereka pesan. Kita sudah memiliki kolom lead time yang menjelaskan berapa lama waktu para tamu sampai di hotel dihitung sejak tamu tersebut membuat booking pertama kali. Sehingga kita bisa membuangnya.
- ***reserved_room_type***: Kita tidak akan menghapus kolom ini karena sudah memiliki kolom ***assigned_room_type*** untuk melihat ruangan yang ditempati oleh para tamu hotel.
- ***agent dan company***: kolom agents dan company merupakan kolom yang menjelaskan tentang nomor ID agen atau perusahaan yang terafiliasi dengan hotel saat melakukan marketing untuk mencari calon customer. Karena tidak ada dataset lain yang menjelaskan secara detail tentang agen dan juga perusahaan afiliasi tersebut, maka kita akan membuang kolom ini.
- ***reservation_status***: kolom ini menjelaskan tentang keterangan reservasi dari tamu apakah tamu melakukan cancel, atau jika tidak cancel booking maka keteranganya berubah menjadi check out. Kita tidak akan menggunakan kolom ini karena kita sudah memiliki kolom is_canceled yang menjelaskan apakah tamu tersebut jadi memesan kamar di hotel atau tidak.
- ***reservation_status_date***: Kolom ini menjelaskan tentang kapan terakhir kali penggantian status pemesanan dilakukan. Kita tidak akan menggunakan kolom ini untuk melakukan analisa karena kita sudah memiliki tanggal kapan tamu melakukan booking pertama kali dan sudah bisa di analisa trend nya.


## Data Adjustment

Pada tahapan ini dilakukan pembuatan kolom baru (hasil penggabungan beberapa kolom untuk lebih ringkas tanpa menghilangkan informasi penting) dan penghapusan kolom yang tidak dipakai pada tahapan EDA dan pembuatan model.

- Kolom baru dibuat dengan nama ***arrival_date*** yang merupakan gabungan dari kolom ***arrival_date_year***, ***arrival_date_month***,
dan ***arrival_date_day_of_month***
- Kolom baru dibuat dengan nama ***total_cust*** yang merupakan gabungan antara kolom ***adults***, ***children***, dan ***babies***.
- Kolom ***is_repeated_guest*** akan di mapping dari berbentuk biner (1 dan 0) menjadi bentuk "Ya" dan "Tidak".
- Kolom **'reserved_room_type'** didrop atau dihapus karena yang akan digunakan pada tahapan EDA dan pembuatan model adalah **'assigned_room_type'**.

## Handling Outliers

Tidak semua outlier akan dihilangkan pada saat melakukan analisis dan membuat model prediksi, **hanya outlier ekstrim yang akan dihilangkan**

---

# **EDA**

---

**Booking Cancellation**

Booking cancellation analysis adalah hasil analisa untuk mengetahui banyaknya tamu yang melakukan cancel dan tidak berdasarkan dari tipe hotel yang dipengaruhi oleh berbagai macam faktor seperti market segment, customer type, lead time, dan juga days in waiting list.
1. Cancellation by Hotel Type
2. Cancellation by Deposit Type
3. Cancellation by Lead Time
4. Cancellation by days in waiting list
5. Cancellation by Market Segment
6. Cancellation by Customer Type

**Customer Behavior**

Customer behavior analysis adalah analisa untuk mengetahui karakteristik dari tamu yang melakukan booking berdasarkan informasi demografis maupun pemilihan service hotel di city maupun resort hotel.
1. Analisa Demografi
2. Tipe Kamar Paling Populer
3. Preferensi Layanan Makan

**Tren Pemesanan, Pembatalan Pesan, serta Analisis Opportunity Cost.**

1. Booking trends over time: Analisis tren volume pemesanan dan tingkat pembatalan di setiap bulan dari tahun ke tahun.
2. Cancellation rate trends over time: Analisis tren pembatalan booking di setiap bulan dari tahun ke tahun.
3. Opportunity cost bedasarkan ADR: opportunity cost dihitung berasal dari ADR customer yang melakukan pembatalan booking dengan metode refundable.


Visualisasi dashboard yang telah dibuat dapat dilihat sebagai berikut:
https://public.tableau.com/app/profile/iqbal.latief/viz/shared/NS9DK2DKZ

---

# **Feature Selection**

---

## Numerical Features
**'lead_time', 'required_car_parking_space', 'adr', 'previous_cancellations', 'total_of_special_request', 'booking_changes', 'total_cust', dan 'previous_booking_not_cancelled'** adalah fitur numerik yang akan digunakan.

## Categorical Features
Seluruh fitur kategorik akan digunakan dalam tahapan pembuatan model.

Dataset yang telah dilakukan cleaning dan digunakan untuk pembuatan model prediksi adalah:
https://github.com/zanputra/FinProDelta/blob/main/mainData.csv
https://github.com/zanputra/FinProDelta/blob/main/mainData.xlsx

---

# **Feature Engineering**

---

1. **Encoding** akan dilakukan dengan dua metode yaitu One Hot Encoding untuk fitur yang memiliki nunique < 5 dan Binary Encoding untuk fitur yang memiliki nunique > 5
- **One Hot Encoding**: hotel, meal, distribution_channel, is_repeated_guest, deposit_type, customer_type
- **Binary Encoding**: country, market_segment, reserved_room_type, assigned_room_type

2. **Scaling** akan dilakukan untuk kolom fitur dengan tipe data numerikal (float dan int). Meskipun kita sudah menghapus ekstrem outliers, pada scaling kali ini kita akan tetap menggunakan **Robust Scaler** karena sebetulnya masih terdapat data poin yang bisa dianggap outliers tetapi tidak esktrem.

---

# **Machine Learning Model**

---

## Model Benchmarking
Model yang akan digunakan adalah:
1. Base Model (KNeighbors Classifier, Decision Tree Classifier, dan Logistic Regression)
2. Ensemble (Soft Voting, Hard Voting, Stacking - KNN, Stacking - DT, dan Stacking - Logistic Regression)
3. Bagging (Bagging Classifier dan Random Forest Classifier)
4. Boosting (AdaBoost Classifier, Gradient Boosting Classifier, dan XGBoost Classifier)

## Resampling (Oversampling & Undersampling)
Metode oversampling yang akan digunakan adalah:

- Random Over Sampler
- ADASYN
- SMOTE
- Borderline SMOTE

Metode undersampling yang akan digunakan adalah:

- Random Under Sampler
- Near Miss
- Tomek Links

## Hyperparameter Tuning
Hyperparameter tuning dilakukan pada 3 model terbaik (nilai recall tertinggi) dengan pasangan resamplingnya.

## Compare Actual and Predicted Value
Pada tahapan ini dilakukan perbandingan hasil prediksi dengan model terpilih dengan hasil aktual (is_canceled), selanjutnya dapat digunakan untuk menentukan batasan model / menentukan karakteristik data yang dapat diprediksi secara akurat oleh model akhir.

## Feature Importance (SHAP)
Pada tahapan ini dilakukan analisis SHAP untuk menentukan fitur yang paling berpengaruh terhadap hasil prediksi.

## Impact of Model (Simulasi Finansial)
Pada tahap ini kita akan mencoba untuk menerapkan hasil prediksi kita ke data yang sudah diprediksi dan akan dibuatkan simulasi profit dan juga opporunity cost yang dihasilkan. 

## Save Model
---
Model terpilih yang telah disimpan:
https://github.com/zanputra/FinProDelta/blob/main/best_model.pkl

---

# **Conclusion & Recommendation**

---

Kesimpulan dan rekomendasi yang diberikan adalah berdasarkan hasil temuan EDA dan pembuatan model prediksi. Untuk proses pengerjaan secara end-to-end dapat dilihat pada:
https://github.com/zanputra/FinProDelta/blob/main/FINPRO_Delta_Team%20FINAL.ipynb

---


