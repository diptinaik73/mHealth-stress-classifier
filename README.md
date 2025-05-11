# mHealth Stress Classifier

This project detects stress vs calm states using the [mHealth dataset](https://archive.ics.uci.edu/ml/datasets/mhealth+dataset), combining ECG and accelerometer features.

## 💡 Features Used

- Accelerometer: X, Y, Z mean, Signal Magnitude Area (SMA)
- ECG: mean, standard deviation
- HRV: standard deviation of RR intervals

## 🧠 Models Implemented

- Decision Tree
- Random Forest

## 🗃️ Dataset

- 10 subjects
- Activities manually grouped into:
  - Calm: standing, sitting, lying
  - Stress: stairs, cycling, jogging, running

## 📁 Files

| File                   | Purpose                    |
|------------------------|----------------------------|
| `decision_tree_model.py` | 5-second window classifier |
| `random_forest_model.py` | Ensemble classifier        |
| `requirements.txt`     | Python dependencies        |

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/diptinaik73/mHealth-stress-classifier.git
   cd mHealth-stress-classifier
