import numpy as np
import os
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

Fs = 50
window_size = Fs * 5  # 5 seconds
data_path = 'data/'

X_train, y_train, X_test, y_test = [], [], [], []

# Simple ECG bandpass filter (0.5–20 Hz)
def filter_ecg(ecg):
    b, a = butter(4, [0.5, 20], btype='bandpass', fs=Fs)
    return filtfilt(b, a, ecg)

# Process all subjects
for subject in range(1, 11):
    file = os.path.join(data_path, f"mHealth_subject{subject}.log")
    if not os.path.exists(file):
        print(f"Missing: {file}")
        continue

    data = np.loadtxt(file)
    acc_x, acc_y, acc_z = data[:, 0], data[:, 1], data[:, 2]
    ecg_raw = data[:, 3]
    ecg = filter_ecg(ecg_raw)
    labels = data[:, 23]

    # Create binary labels
    binary_labels = -1 * np.ones_like(labels)
    binary_labels[np.isin(labels, [1, 2, 3])] = 0  # calm
    binary_labels[np.isin(labels, [5, 9, 10, 11])] = 1  # stress

    for i in range(0, len(labels) - window_size, window_size):
        chunk_label = binary_labels[i:i+window_size]
        calm_ratio = np.mean(chunk_label == 0)
        stress_ratio = np.mean(chunk_label == 1)

        if calm_ratio >= 0.8:
            label = 0
        elif stress_ratio >= 0.8:
            label = 1
        else:
            continue  # mixed or undefined

        # Extract signals
        win_x = acc_x[i:i+window_size]
        win_y = acc_y[i:i+window_size]
        win_z = acc_z[i:i+window_size]
        win_ecg = ecg[i:i+window_size]

        # Features
        f1 = np.mean(win_x)
        f2 = np.mean(win_y)
        f3 = np.mean(win_z)
        f4 = np.mean(np.sqrt(win_x**2 + win_y**2 + win_z**2))  # SMA
        f5 = np.mean(win_ecg)
        f6 = np.std(win_ecg)

        peaks, _ = find_peaks(win_ecg, height=0.3, distance=25)
        rr = np.diff(peaks) / Fs
        f7 = np.std(rr) if len(rr) >= 2 else 0

        features = [f1, f2, f3, f4, f5, f6, f7]

        if subject <= 8:
            X_train.append(features)
            y_train.append(label)
        else:
            X_test.append(features)
            y_test.append(label)

# Check that data exists
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
if len(X_train) == 0 or len(X_test) == 0:
    print("❌ No valid data. Check your .log files and label rules.")
    exit()

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(np.array(X_train))
X_test = scaler.transform(np.array(X_test))

# Train decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Results
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Calm", "Stress"]))