# ************************************************************
# README / HEADER COMMENTS
# ************************************************************
# Project Title: Integrating NLP and Deep Learning for
#                Regulatory and Sociopolitical Risk Forecasting:
#                A Case Study of First Mining Gold Corp. (FFMGF)
#
#
# HOW TO RUN:
#   1. Install dependencies:
#      pip install yfinance pandas numpy scikit-learn xgboost
#          transformers torch matplotlib seaborn statsmodels
#          requests beautifulsoup4 imbalanced-learn
#   2. Run the script top-to-bottom:
#      python final2.py
#   3. Outputs:
#      - Console: model metrics, summaries
#      - Files: sentiment_data.csv, results_summary.csv,
#               figures/ directory with all plots
#
# SECTIONS:
#   Section 0  - Imports & Configuration
#   Section 1  - Data Collection (yfinance, gold prices)
#   Section 2  - Data Wrangling & Feature Engineering
#   Section 3  - Sentiment Analysis with FinBERT
#   Section 4  - Exploratory Data Analysis (EDA)
#   Section 5  - Imbalanced Classification Setup (SMOTE)
#   Section 6  - Ensemble Models (Random Forest, XGBoost)
#   Section 7  - Deep Learning Model (LSTM)
#   Section 8  - Model Evaluation & Metrics
#   Section 9  - Time-Series Anomaly Detection
#   Section 10 - Results Export
# ************************************************************


# ************************************************************
# SECTION 0: Imports & Configuration
# ************************************************************
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Financial data
import yfinance as yf

# Machine learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# XGBoost
from xgboost import XGBClassifier

# Deep learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# NLP / Transformers
from transformers import pipeline as hf_pipeline

# Stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# Output directory for figures
os.makedirs("figures", exist_ok=True)
print("Section 0 complete: Imports loaded.")


# ************************************************************
# SECTION 1: Data Collection
# ************************************************************
# 1a. Download FFMGF stock price data via yfinance
print("\nSection 1: Downloading market data...")

TICKER = "FFMGF"
GOLD_TICKER = "GC=F"   # Gold futures (proxy for spot price)
START_DATE = "2020-01-01"
END_DATE   = "2026-03-27"

ffmgf_raw = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
gold_raw  = yf.download(GOLD_TICKER, start=START_DATE, end=END_DATE, progress=False)

# Flatten MultiIndex columns if present
if isinstance(ffmgf_raw.columns, pd.MultiIndex):
    ffmgf_raw.columns = ffmgf_raw.columns.get_level_values(0)
if isinstance(gold_raw.columns, pd.MultiIndex):
    gold_raw.columns = gold_raw.columns.get_level_values(0)

print(f"  FFMGF rows: {len(ffmgf_raw)}")
print(f"  Gold rows : {len(gold_raw)}")

# 1b. Merge on date index
df = ffmgf_raw[["Open","High","Low","Close","Volume"]].copy()
df.columns = ["FFMGF_Open","FFMGF_High","FFMGF_Low","FFMGF_Close","FFMGF_Volume"]
df["Gold_Close"] = gold_raw["Close"].reindex(df.index, method="ffill")
df.dropna(inplace=True)
print(f"  Merged dataset shape: {df.shape}")


# ************************************************************
# SECTION 2: Data Wrangling & Feature Engineering
# ************************************************************
print("\nSection 2: Feature engineering...")

# Daily log returns
df["FFMGF_Return"] = np.log(df["FFMGF_Close"] / df["FFMGF_Close"].shift(1))
df["Gold_Return"]  = np.log(df["Gold_Close"]  / df["Gold_Close"].shift(1))

# Rolling volatility (20-day)
df["FFMGF_Vol20"] = df["FFMGF_Return"].rolling(20).std()

# Moving averages
df["MA10"]  = df["FFMGF_Close"].rolling(10).mean()
df["MA50"]  = df["FFMGF_Close"].rolling(50).mean()
df["MA200"] = df["FFMGF_Close"].rolling(200).mean()

# Relative Strength Index (RSI-14)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

df["RSI14"] = compute_rsi(df["FFMGF_Close"])

# MACD
ema12 = df["FFMGF_Close"].ewm(span=12, adjust=False).mean()
ema26 = df["FFMGF_Close"].ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Bollinger Bands
bb_mid = df["FFMGF_Close"].rolling(20).mean()
bb_std = df["FFMGF_Close"].rolling(20).std()
df["BB_Upper"] = bb_mid + 2 * bb_std
df["BB_Lower"] = bb_mid - 2 * bb_std
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid

# Gold-to-FFMGF correlation (rolling 30-day)
df["Gold_FFMGF_Corr"] = (
    df["FFMGF_Return"].rolling(30)
    .corr(df["Gold_Return"])
)

# Crash label: 1 if next-day return < -5th percentile threshold
threshold = df["FFMGF_Return"].quantile(0.05)
df["Crash_Label"] = (df["FFMGF_Return"].shift(-1) < threshold).astype(int)

df.dropna(inplace=True)
print(f"  Features engineered. Dataset shape: {df.shape}")
print(f"  Crash threshold (5th pct): {threshold:.4f}")
print(f"  Crash events: {df['Crash_Label'].sum()} / {len(df)}")


# ************************************************************
# SECTION 3: Sentiment Analysis with FinBERT
# ************************************************************
print("\nSection 3: Sentiment analysis (FinBERT)...")

# Simulated regulatory/news headlines for FFMGF
# In production: replace with scraped news, SEC filings, EA documents
SAMPLE_HEADLINES = [
    "First Mining Gold receives environmental assessment approval for Springpole project",
    "Indigenous community raises concerns over Springpole Gold Project water rights",
    "Ontario government delays permitting decision for First Mining Gold Springpole",
    "First Mining Gold Corp reports strong Q3 results amid rising gold prices",
    "Regulatory setback: Environmental review extended for Springpole Gold Project",
    "First Mining Gold secures key partnership with local Indigenous groups",
    "Protest halts survey work at Springpole Gold Project site",
    "Gold prices surge; First Mining Gold shares rally on positive outlook",
    "Environmental groups file legal challenge against Springpole mine approval",
    "First Mining Gold completes updated feasibility study for Springpole project",
    "Federal regulators impose new conditions on Springpole environmental assessment",
    "First Mining Gold announces $50M financing round for project development",
    "Community opposition grows over Springpole Gold Project expansion plans",
    "First Mining Gold Corp stock drops 12% following permitting delay announcement",
    "Positive Indigenous consultation outcome boosts First Mining Gold shares",
]

# Load FinBERT sentiment pipeline
# Note: requires internet access on first run to download model weights
try:
    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )
    sentiments = finbert(SAMPLE_HEADLINES)
    sentiment_df = pd.DataFrame({
        "Headline": SAMPLE_HEADLINES,
        "Sentiment": [s["label"] for s in sentiments],
        "Score":     [round(s["score"], 4) for s in sentiments],
    })

    # Map to numeric: positive=1, neutral=0, negative=-1
    sent_map = {"positive": 1, "neutral": 0, "negative": -1}
    sentiment_df["Sentiment_Numeric"] = sentiment_df["Sentiment"].map(sent_map)
    print(sentiment_df.to_string(index=False))
    sentiment_df.to_csv("sentiment_data.csv", index=False)
    print("  Sentiment data saved to sentiment_data.csv")
    avg_sentiment = sentiment_df["Sentiment_Numeric"].mean()
    print(f"  Average sentiment score: {avg_sentiment:.3f}")
except Exception as e:
    print(f"  FinBERT unavailable ({e}). Using simulated sentiment scores.")
    np.random.seed(42)
    sentiment_df = pd.DataFrame({
        "Headline": SAMPLE_HEADLINES,
        "Sentiment": np.random.choice(["positive","neutral","negative"], 15, p=[0.4,0.3,0.3]),
        "Score": np.round(np.random.uniform(0.6, 0.99, 15), 4),
    })
    sent_map = {"positive": 1, "neutral": 0, "negative": -1}
    sentiment_df["Sentiment_Numeric"] = sentiment_df["Sentiment"].map(sent_map)
    sentiment_df.to_csv("sentiment_data.csv", index=False)
    avg_sentiment = sentiment_df["Sentiment_Numeric"].mean()
    print(f"  Simulated average sentiment: {avg_sentiment:.3f}")

# Add aggregate sentiment as a feature (broadcast to all rows as proxy)
df["Avg_Sentiment"] = avg_sentiment


# ************************************************************
# SECTION 4: Exploratory Data Analysis (EDA)
# ************************************************************
print("\nSection 4: EDA & visualizations...")

# --- Figure 1: FFMGF Price History with Moving Averages ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(df.index, df["FFMGF_Close"], label="FFMGF Close", color="steelblue", lw=1.5)
axes[0].plot(df.index, df["MA10"],  label="MA-10",  color="orange",  lw=1, ls="--")
axes[0].plot(df.index, df["MA50"],  label="MA-50",  color="green",   lw=1, ls="--")
axes[0].plot(df.index, df["MA200"], label="MA-200", color="red",     lw=1, ls="--")
axes[0].set_ylabel("Price (USD)")
axes[0].set_title("Figure 1: FFMGF Stock Price with Moving Averages (2020â€“2024)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].bar(df.index, df["FFMGF_Return"], color=np.where(df["FFMGF_Return"]>=0,"green","red"), alpha=0.6, width=1)
axes[1].set_ylabel("Log Return")
axes[1].set_title("Daily Log Returns")
axes[1].axhline(threshold, color="black", ls="--", lw=1, label=f"Crash threshold ({threshold:.3f})")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

axes[2].plot(df.index, df["FFMGF_Vol20"], color="purple", lw=1.5)
axes[2].set_ylabel("Rolling Volatility (20d)")
axes[2].set_title("20-Day Rolling Volatility")
axes[2].grid(alpha=0.3)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("figures/fig1_price_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig1_price_history.png")

# --- Figure 2: Correlation Heatmap ---
feature_cols = [
    "FFMGF_Return","Gold_Return","FFMGF_Vol20",
    "RSI14","MACD","BB_Width","Gold_FFMGF_Corr","Crash_Label"
]
corr_matrix = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Figure 2: Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/fig2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig2_correlation_heatmap.png")

# --- Figure 3: Sentiment Distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
colors = {"positive":"green","neutral":"gray","negative":"red"}
sentiment_df["Sentiment"].value_counts().plot(
    kind="bar", ax=ax,
    color=[colors.get(x,"blue") for x in sentiment_df["Sentiment"].value_counts().index]
)
ax.set_title("Figure 3: FinBERT Sentiment Distribution of FFMGF News Headlines")
ax.set_xlabel("Sentiment Class")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("figures/fig3_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig3_sentiment_distribution.png")

# --- Figure 4: Gold vs FFMGF Price (Normalized) ---
fig, ax = plt.subplots(figsize=(12, 5))
norm_ffmgf = df["FFMGF_Close"] / df["FFMGF_Close"].iloc[0] * 100
norm_gold  = df["Gold_Close"]  / df["Gold_Close"].iloc[0]  * 100
ax.plot(df.index, norm_ffmgf, label="FFMGF (normalized)", color="steelblue")
ax.plot(df.index, norm_gold,  label="Gold Futures (normalized)", color="gold", lw=1.5)
ax.set_title("Figure 4: Normalized Price Comparison ” FFMGF vs. Gold (Base=100)")
ax.set_ylabel("Normalized Price")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_normalized_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig4_normalized_comparison.png")


# ************************************************************
# SECTION 5: Imbalanced Classification Setup (SMOTE)
# ************************************************************
print("\nSection 5: Preparing classification dataset with SMOTE...")

FEATURE_COLS = [
    "FFMGF_Return","Gold_Return","FFMGF_Vol20",
    "RSI14","MACD","MACD_Signal","BB_Width",
    "Gold_FFMGF_Corr","Avg_Sentiment"
]

X = df[FEATURE_COLS].values
y = df["Crash_Label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
print(f"  Before SMOTE ” Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
print(f"  After  SMOTE ” Class 0: {(y_train_res==0).sum()}, Class 1: {(y_train_res==1).sum()}")


# ************************************************************
# SECTION 6: Ensemble Models â€” Random Forest & XGBoost
# ************************************************************
print("\nSection 6: Training ensemble models...")

# --- Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)
rf_pred  = rf_model.predict(X_test_sc)
rf_proba = rf_model.predict_proba(X_test_sc)[:, 1]

rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, zero_division=0)
rf_auc  = roc_auc_score(y_test, rf_proba)
print(f"  Random Forest ” Acc: {rf_acc:.4f} | F1: {rf_f1:.4f} | AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf_pred, zero_division=0))

# --- XGBoost ---
scale_pos = (y_train_res == 0).sum() / max((y_train_res == 1).sum(), 1)
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train_res, y_train_res)
xgb_pred  = xgb_model.predict(X_test_sc)
xgb_proba = xgb_model.predict_proba(X_test_sc)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1  = f1_score(y_test, xgb_pred, zero_division=0)
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"  XGBoost ” Acc: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | AUC: {xgb_auc:.4f}")
print(classification_report(y_test, xgb_pred, zero_division=0))

# --- Figure 5: Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, title in zip(axes, [rf_pred, xgb_pred], ["Random Forest", "XGBoost"]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Crash","Crash"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Figure 5: {title} Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/fig5_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig5_confusion_matrices.png")

# --- Figure 6: Feature Importance (Random Forest) ---
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Figure 6: Random Forest Feature Importances")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("figures/fig6_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig6_feature_importance.png")


# ************************************************************
# SECTION 7: Deep Learning  LSTM Model
# ************************************************************
print("\nSection 7: Training LSTM model...")

SEQ_LEN = 20  # 20-day lookback window

def create_sequences(X_arr, y_arr, seq_len):
    Xs, ys = [], []
    for i in range(len(X_arr) - seq_len):
        Xs.append(X_arr[i : i + seq_len])
        ys.append(y_arr[i + seq_len])
    return np.array(Xs), np.array(ys)

# Use scaled full dataset for LSTM (train/test split by time)
X_all_sc = scaler.transform(X)
X_seq, y_seq = create_sequences(X_all_sc, y, SEQ_LEN)

split_idx = int(len(X_seq) * 0.8)
X_tr_seq, X_te_seq = X_seq[:split_idx], X_seq[split_idx:]
y_tr_seq, y_te_seq = y_seq[:split_idx], y_seq[split_idx:]

# Convert to tensors
X_tr_t = torch.FloatTensor(X_tr_seq)
y_tr_t = torch.FloatTensor(y_tr_seq)
X_te_t = torch.FloatTensor(X_te_seq)
y_te_t = torch.FloatTensor(y_te_seq)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# LSTM architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMClassifier(input_size=len(FEATURE_COLS)).to(device)

# Class-weighted loss for imbalance
pos_weight = torch.tensor([(y_tr_seq == 0).sum() / max((y_tr_seq == 1).sum(), 1)]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 30
train_losses = []
for epoch in range(EPOCHS):
    lstm_model.train()
    epoch_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = lstm_model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(train_dl)
    train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")

# Evaluate LSTM
lstm_model.eval()
with torch.no_grad():
    lstm_proba_raw = lstm_model(X_te_t.to(device)).cpu().numpy()
lstm_proba = lstm_proba_raw
lstm_pred  = (lstm_proba >= 0.5).astype(int)

lstm_acc = accuracy_score(y_te_seq, lstm_pred)
lstm_f1  = f1_score(y_te_seq, lstm_pred, zero_division=0)
try:
    lstm_auc = roc_auc_score(y_te_seq, lstm_proba)
except Exception:
    lstm_auc = float("nan")
print(f"  LSTM Acc: {lstm_acc:.4f} | F1: {lstm_f1:.4f} | AUC: {lstm_auc:.4f}")

# --- Figure 7: LSTM Training Loss ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, EPOCHS+1), train_losses, color="steelblue", marker="o", ms=3)
ax.set_title("Figure 7: LSTM Training Loss Over Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig7_lstm_training_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig7_lstm_training_loss.png")


# ************************************************************
# SECTION 8: Model Evaluation & Comparison
# ************************************************************
print("\nSection 8: Model comparison summary...")

results = pd.DataFrame({
    "Model":    ["Random Forest", "XGBoost", "LSTM"],
    "Accuracy": [rf_acc, xgb_acc, lstm_acc],
    "F1_Score": [rf_f1, xgb_f1, lstm_f1],
    "ROC_AUC":  [rf_auc, xgb_auc, lstm_auc],
})
print(results.to_string(index=False))
results.to_csv("results_summary.csv", index=False)
print("  Results saved to results_summary.csv")

# --- Figure 8: Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results["Model"]))
width = 0.25
ax.bar(x - width, results["Accuracy"], width, label="Accuracy", color="steelblue")
ax.bar(x,         results["F1_Score"], width, label="F1 Score",  color="orange")
ax.bar(x + width, results["ROC_AUC"],  width, label="ROC-AUC",   color="green")
ax.set_xticks(x)
ax.set_xticklabels(results["Model"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Figure 8: Model Performance Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig8_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig8_model_comparison.png")


# ************************************************************
# SECTION 9: Time-Series Anomaly Detection (ADF + Decomposition)
# ************************************************************
print("\nSection 9: Time-series analysis...")

# Augmented Dickey-Fuller test for stationarity
adf_result = adfuller(df["FFMGF_Return"].dropna())
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  p-value:       {adf_result[1]:.4f}")
print(f"  Stationary:    {adf_result[1] < 0.05}")

# Seasonal decomposition of closing price
decomp = seasonal_decompose(df["FFMGF_Close"], model="multiplicative", period=252)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], color="steelblue"); axes[0].set_ylabel("Observed")
decomp.trend.plot(ax=axes[1], color="orange");       axes[1].set_ylabel("Trend")
decomp.seasonal.plot(ax=axes[2], color="green");     axes[2].set_ylabel("Seasonal")
decomp.resid.plot(ax=axes[3], color="red");          axes[3].set_ylabel("Residual")
axes[0].set_title("Figure 9: Seasonal Decomposition of FFMGF Closing Price")
plt.tight_layout()
plt.savefig("figures/fig9_seasonal_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig9_seasonal_decomposition.png")

# Z-score anomaly detection on returns
df["Return_ZScore"] = (df["FFMGF_Return"] - df["FFMGF_Return"].mean()) / df["FFMGF_Return"].std()
anomalies = df[df["Return_ZScore"].abs() > 2.5]
print(f"  Anomalous return days (|Z|>2.5): {len(anomalies)}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["Return_ZScore"], color="steelblue", lw=0.8, label="Z-Score")
ax.scatter(anomalies.index, anomalies["Return_ZScore"], color="red", zorder=5, s=20, label="Anomaly (|Z|>2.5)")
ax.axhline(2.5,  color="orange", ls="--", lw=1)
ax.axhline(-2.5, color="orange", ls="--", lw=1)
ax.set_title("Figure 10: Z-Score Anomaly Detection on FFMGF Daily Returns")
ax.set_ylabel("Z-Score")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig10_anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig10_anomaly_detection.png")


# ************************************************************
# SECTION 10: Results Export
# ************************************************************
print("\nSection 10: Exporting final dataset...")
df.to_csv("ffmgf_processed_data.csv")
print("  Full processed dataset saved to ffmgf_processed_data.csv")
print("\n=== Pipeline Complete ===")
print("Output files:")
print("  - sentiment_data.csv")
print("  - results_summary.csv")
print("  - ffmgf_processed_data.csv")
print("  - figures/ (10 PNG figures)")