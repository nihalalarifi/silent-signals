# 🧠 Silent Signals — Depression Prediction via Search Engine Behavior

> Deep learning models that predict depression case counts in Riyadh, Saudi Arabia, by analyzing Google Trends search data — offering a non-invasive, data-driven early warning system for mental health deterioration.

**King Saud University — Computer Science Graduation Project (CSC 497)**  
Supervised by Dr. Mai Alzamel | Second Semester 1446/1447

---

## 💡 Why We Built This

Over 50% of major depressive disorder (MDD) cases go undiagnosed globally, often because people first turn to search engines — not doctors — when they feel distress. We asked: *what if we could read those signals?*

By correlating Google Trends search volumes for depression-related keywords (Depression, Insomnia, Weight Loss, Anxiety, Sad) with actual recorded depression cases in Riyadh, we built models that can predict emerging mental health trends **before they appear in clinical data** — giving health authorities a head start on early intervention.

---

## 🚀 Features

- **Four deep learning models compared** — LSTM, BiLSTM, GRU, and Transformer, each trained and evaluated on two data sources (Google Web Search & YouTube Search)
- **Anomaly detection** — identifies abnormal spikes in search behavior that may signal emerging mental health crises
- **Temporal shift analysis** — evaluates whether search trends can predict depression cases weeks or months ahead
- **Comprehensive evaluation** — MSE, RMSE, MAE, sMAPE, F1-score, Pearson correlation, and confusion matrices across train/validation/test splits
- **Dual data sources** — models tested on both Google Web search trends and YouTube search trends independently

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Deep learning model training |
| scikit-learn | Preprocessing, metrics, data splitting |
| Pandas & NumPy | Data manipulation and sequence generation |
| Matplotlib | Visualization of predictions and anomalies |
| SciPy | Pearson correlation analysis |
| Google Trends (pytrends) | Data source — search volume time series |
| Jupyter Notebook | Interactive experimentation environment |

---

## 📁 Project Structure

```
silent-signals/
├── MODELS/
│   ├── LSTM.ipynb          # Long Short-Term Memory model
│   ├── BiLSTM.ipynb        # Bidirectional LSTM model
│   ├── GRU.ipynb           # Gated Recurrent Unit model
│   └── Transformer.ipynb   # Transformer-based model
├── Datasets/
│   ├── dataset_web_eradah.csv      # Google Web search trends + actual cases
│   └── dataset_youtube_eradah.csv  # YouTube search trends + actual cases
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/silent-signals.git
cd silent-signals

# 2. Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib scipy jupyter

# 3. Launch Jupyter
jupyter notebook

# 4. Open any model notebook from the MODELS/ folder and run all cells
#    (datasets are referenced relatively — no path changes needed)
```

---

## 📖 Usage

Each notebook in `MODELS/` is self-contained and follows the same pipeline:

1. **Load dataset** from `../Datasets/` (web or YouTube)
2. **Preprocess** — MinMaxScaler normalization, sequence generation (12-month lookback window)
3. **Split** — 55% train / 20% validation / 25% test (chronological, no shuffling)
4. **Train model** with EarlyStopping and ModelCheckpoint callbacks
5. **Evaluate** forecasting performance (MSE, RMSE, MAE, sMAPE)
6. **Run anomaly detection** — threshold-based binary classification with temporal shift analysis
7. **Visualize** — actual vs. predicted plots, anomaly markers, confusion matrices, correlation scatter plots

---

## 📊 Key Results

| Model | Test MSE | Test RMSE | Test MAE | Test sMAPE |
|---|---|---|---|---|
| **Transformer** ✅ | **0.01** | **0.10** | **0.09** | **23.29%** |
| LSTM | 0.02 | 0.14 | 0.11 | 29.76% |
| BiLSTM | 0.09 | — | — | — |
| GRU | 0.07 | — | — | — |

**Anomaly Detection (Best F1-scores):**

| Model | Train F1 | Test F1 |
|---|---|---|
| BiLSTM | 0.88 (zero shift) | 0.71 |
| Transformer | 0.67 | **0.73** (3-month shift) |
| LSTM | 0.74 | 0.47 |
| GRU | 0.71 | 0.67 |

---

## 🧠 Technical Challenge

The hardest problem we faced was **anomaly detection in a small, noisy dataset.**

With only 81 monthly data points, standard fixed thresholds either flagged everything or nothing as anomalies. Our solution was a **grid search over threshold multipliers (k-values)** applied to rolling statistics of both actual and predicted series, combined with **temporal shift analysis** — testing whether shifting predictions by 1–3 months improved alignment with real case spikes.

This revealed something clinically meaningful: search behavior predicts depression case surges **2–3 months before they peak in hospital data**, which is exactly the lead time needed for early intervention programs. The Transformer model captured this lag most effectively, achieving F1=0.73 on the test set at a 3-month temporal shift.

---

## 📋 Dataset

- **Source:** Google Trends (via manual export) & Ministry of Health depression case records for Riyadh
- **Time range:** January 2018 – September 2024 (81 months)
- **Features:** Monthly Relative Search Volume (RSV) for 5 keywords — Depression, Tired, Weight Loss, Anxiety, Sad
- **Target:** Actual recorded depression cases in Riyadh per month
- **Privacy:** All data is publicly available and fully anonymized — no personally identifiable information

---

## 👥 Authors

| Name | Student ID |
|---|---|
| Nihal Alarifi | 441200967 |
| Sara Alharbi | 442202335 |
| Jumana bin Zaid | 443200778 |

**Supervisor:** Dr. Mai Alzamel  
College of Computer and Information Sciences — King Saud University

---

## 📄 License

MIT License — free to use, adapt, and build upon with attribution.
