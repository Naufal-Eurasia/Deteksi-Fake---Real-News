# Deteksi-Fake-Real-News
UAS Kecerdasan Komputasional (TI22A)

```python
# Import library yang dibutuhkan
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import numpy as np
```
```python
# Unggah file CSV ke Colab sebelum menjalankan sel ini
from google.colab import files
uploaded = files.upload()
```
```python
# Load dataset
df = pd.read_csv('WELFake_Dataset.csv')
df = df[['text', 'label']].dropna().head(100)
df.head()
```
```python
from matplotlib import pyplot as plt
_df_0['label'].plot(kind='hist', bins=20, title='label')
plt.gca().spines[['top', 'right',]].set_visible(False)
```
```python
# Preprocessing dan tokenisasi
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(df['label'])

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

X = padded_sequences
y = labels_encoded
```
```python
# Split data dan bangun model CNN
X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=200),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=4)
```
```python
# Fungsi prediksi teks baru
def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200, padding='post')
    pred = model.predict(pad)[0][0]
    return "Real" if pred > 0.5 else "Fake", pred

# Contoh prediksi
predict_news("Pemerintah umumkan bantuan dana langsung bagi seluruh warga")
```
Dokumentasi :
![image](https://github.com/user-attachments/assets/c7bea77b-0fae-4b98-a6e1-7b0c2864675a)
![image](https://github.com/user-attachments/assets/18e13a05-53e0-4b53-ae15-bd77c6f2ad11)
![image](https://github.com/user-attachments/assets/2c1f6b3e-b7ca-4a9d-8314-294d32dbaeba)
![image](https://github.com/user-attachments/assets/a4df711a-44dc-4aae-a311-483efdcd802e)
![image](https://github.com/user-attachments/assets/f2139e7b-09c6-4864-9ae1-54f8032e93dc)
![image](https://github.com/user-attachments/assets/f1825c5f-1d1b-430a-98eb-fe1dc0843313)




