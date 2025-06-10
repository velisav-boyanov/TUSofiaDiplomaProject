import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Зареждане на аудио файл (може да използвате ваш)
filename = librosa.example('trumpet')  # или път към ваш .wav файл
signal, sr = librosa.load(filename, sr=None, duration=2.0)

# == Метод 1: Преобразуване на Фурие ==
fft_features = np.abs(np.fft.fft(signal))[:len(signal)//2]  # половината спектър
fft_features = fft_features[:500]  # Намаляване на размерността

# == Метод 2: Извличане на MFCC + Линеен модел ==
mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
mfcc_features = np.mean(mfcc.T, axis=0)

# Подготовка на фиктивни данни (пример с две класови хипотези)
X = np.vstack([fft_features, mfcc_features])
y = np.array([0, 1])  # етикети за двата метода

# Маскиране на различни размерности чрез padding
max_len = max(len(x) for x in X)
X_padded = np.array([np.pad(x, (0, max_len - len(x))) for x in X])

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_padded)

# Разделяне на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

# == Класификация: Метод 1 – FFT + KNN ==
model_knn = KNeighborsClassifier(n_neighbors=1)
model_knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)

# == Класификация: Метод 2 – MFCC + Logistic Regression ==
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)

# == Резултати ==
print(f"Точност с FFT + KNN: {acc_knn * 100:.2f}%")
print(f"Точност с MFCC + Logistic Regression: {acc_lr * 100:.2f}%")

# Визуализация
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(fft_features)
plt.title("FFT спектър")

plt.subplot(1, 2, 2)
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC характеристики")
plt.tight_layout()
plt.show()
