import numpy as np
import os
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

Fs = 50
window_size = Fs * 5
data_path = 'data/'

X_train, y_train, X_test, y_test = [], [], [], []

# ECG bandpass filter
def filter_ecg(ecg):
    b, a = butter(4, [0.5, 20], btype='bandpass', fs=Fs)
    return filtfilt(b, a, ecg)

# Smooth ACC with moving average
def smooth_acc(signal, window=25):
    return np.convolve(signal, np.ones(window) / window, mode='same')

# Loop through subjects
for subject in range(1, 11):
    file = os.path.join(data_path, f"mHealth_subject{subject}.log")
    if not os.path.exists(file):
        print(f"Missing: {file}")
        continue

    data = np.loadtxt(file)
    acc_x = smooth_acc(data[:, 0])
    acc_y = smooth_acc(data[:, 1])
    acc_z = smooth_acc(data[:, 2])
    ecg = filter_ecg(data[:, 3])
    labels = data[:, 23]

    # Binary label: 0 = calm, 1 = stress
    binary_labels = -1 * np.ones_like(labels)
    binary_labels[np.isin(labels, [1, 2, 3])] = 0
    binary_labels[np.isin(labels, [5, 9, 10, 11])] = 1

    for i in range(0, len(labels) - window_size, window_size):
        chunk = binary_labels[i:i + window_size]
        calm_ratio = np.mean(chunk == 0)
        stress_ratio = np.mean(chunk == 1)

        if calm_ratio >= 0.8:
            label = 0
        elif stress_ratio >= 0.8:
            label = 1
        else:
            continue

        winX = acc_x[i:i + window_size]
        winY = acc_y[i:i + window_size]
        winZ = acc_z[i:i + window_size]
        winECG = ecg[i:i + window_size]

        # Features
        f1 = np.mean(winX)
        f2 = np.mean(winY)
        f3 = np.mean(winZ)
        f4 = np.mean(np.sqrt(winX ** 2 + winY ** 2 + winZ ** 2))  # SMA
        f5 = np.mean(winECG)
        f6 = np.std(winECG)

        peaks, _ = find_peaks(winECG, height=0.3, distance=25)
        rr = np.diff(peaks) / Fs
        f7 = np.std(rr) if len(rr) >= 2 else 0

        features = [f1, f2, f3, f4, f5, f6, f7]

        if subject <= 8:
            X_train.append(features)
            y_train.append(label)
        else:
            X_test.append(features)
            y_test.append(label)

# Check sample counts
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
if len(X_train) == 0 or len(X_test) == 0:
    import sys
    print("❌ No data found. Check your files and labels.")
    sys.exit()

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(np.array(X_train))
X_test = scaler.transform(np.array(X_test))

# Train Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Results
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Calm", "Stress"]))
import pandas as pd

# Save metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'Model': ['Random Forest'],  # or 'Decision Tree'
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Precision': [precision_score(y_test, y_pred)],
    'Recall': [recall_score(y_test, y_pred)],
    'F1 Score': [f1_score(y_test, y_pred)]
}

df = pd.DataFrame(metrics)
df.to_csv('model_results_rf.csv', index=False)  # or model_results_dt.csv
print(" Saved: model_results_rf.csv")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Calm", "Stress"])
plt.title("Confusion Matrix - Random Forest")  # or Decision Tree
plt.savefig("conf_matrix_rf.png", dpi=300, bbox_inches='tight')  # or conf_matrix_dt.png
plt.close()
print(" Confusion matrix saved.")