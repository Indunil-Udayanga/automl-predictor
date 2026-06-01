import os, json, uuid, warnings, pickle, base64, traceback, re
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.sparse import issparse, hstack as sp_hstack, csr_matrix

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

#  Optional NLTK (graceful fallback) 
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for _pkg in ("stopwords", "wordnet", "punkt", "omw-1.4"):
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

# ── Paths 
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR   = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── FastAPI App 
app = FastAPI(title="AutoML Studio", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ── Model Registries 
CLASSIFICATION_MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees":          ExtraTreesClassifier(n_estimators=100, random_state=42),
    "AdaBoost":             AdaBoostClassifier(n_estimators=50, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5),
    "SVM":                  SVC(probability=True, random_state=42),
    "Naive Bayes":          GaussianNB(),
    "MLP Neural Network":   MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

# NLP-specific classifiers — used when dataset_type is text_only or mixed
NLP_CLASSIFICATION_MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42, C=5.0),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees":          ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Multinomial NB":       MultinomialNB(alpha=0.1),
    "Complement NB":        ComplementNB(alpha=0.1),
    "SVM (Linear)":         SVC(kernel="linear", probability=True, random_state=42),
    "MLP Neural Network":   MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

REGRESSION_MODELS = {
    "Linear Regression":    LinearRegression(),
    "Ridge Regression":     Ridge(alpha=1.0),
    "Lasso Regression":     Lasso(alpha=1.0, max_iter=5000),
    "ElasticNet":           ElasticNet(alpha=1.0, max_iter=5000),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Extra Trees":          ExtraTreesRegressor(n_estimators=100, random_state=42),
    "AdaBoost":             AdaBoostRegressor(n_estimators=50, random_state=42),
    "Decision Tree":        DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors":  KNeighborsRegressor(n_neighbors=5),
    "SVR":                  SVR(),
    "MLP Neural Network":   MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}


#  TEXT COLUMN DETECTION

TEXT_COLUMN_HINTS = {
    "text", "content", "message", "body", "description",
    "review", "comment", "tweet", "post", "article",
    "subject", "title", "headline", "email", "mail",
    "summary", "feedback", "note", "news",
}

def _avg_words(series: pd.Series, sample: int = 200) -> float:
    s = series.dropna().astype(str).sample(min(sample, len(series)), random_state=0)
    return s.str.split().apply(len).mean()

def detect_text_columns(df: pd.DataFrame, target_col: str) -> list:
    text_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype not in [object, "string"]:
            continue
        col_lower = col.lower()
        # (a) name hint
        if any(hint in col_lower for hint in TEXT_COLUMN_HINTS):
            text_cols.append(col)
            continue
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        # (b) avg word count > 6
        if _avg_words(sample) > 6:
            text_cols.append(col)
            continue
        # (c) median char length > 40
        if sample.str.len().median() > 40:
            text_cols.append(col)
    return text_cols

def detect_dataset_type(df: pd.DataFrame, target_col: str) -> str:
    """Returns 'text_only', 'mixed', or 'tabular'."""
    text_cols = detect_text_columns(df, target_col)
    if not text_cols:
        return "tabular"
    non_target = [c for c in df.columns if c != target_col]
    non_text   = [c for c in non_target if c not in text_cols]
    return "text_only" if not non_text else "mixed"


#  TEXT CLEANING

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+",   " ", text)   # URLs
    text = re.sub(r"<[^>]+>",            " ", text)   # HTML tags
    text = re.sub(r"\S+@\S+",            " ", text)   # emails
    text = re.sub(r"\d+",                " ", text)   # numbers
    text = re.sub(r"[^a-z\s]",           " ", text)   # punctuation
    text = re.sub(r"\s+",                " ", text).strip()

    if not NLTK_OK:
        return text

    tokens     = text.split()
    stop_words = set(stopwords.words("english"))
    tokens     = [t for t in tokens if t not in stop_words and len(t) > 1]
    lemmatizer = WordNetLemmatizer()
    tokens     = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(clean_text)


#  HELPERS

def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#080d18", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


TARGET_KEYWORDS = [
    "target", "label", "class", "output", "result", "price",
    "sale", "churn", "survived", "diagnosis", "outcome", "y",
    "default", "fraud", "spam", "revenue", "profit", "cost",
    "risk", "score", "grade", "quality", "rating", "Category",
    "sentiment",
]

def detect_target(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in TARGET_KEYWORDS:
        if kw in cols_lower:
            return cols_lower[kw]
    return df.columns[-1]

def detect_problem(series: pd.Series) -> str:
    if series.dtype == object or series.dtype.name == "category":
        return "classification"
    unique_ratio = series.nunique() / max(len(series), 1)
    if series.nunique() <= 15 or unique_ratio < 0.05:
        return "classification"
    return "regression"

def compute_dataset_stats(df: pd.DataFrame, target_col: str) -> dict:
    missing     = df.isnull().sum().sum()
    missing_pct = round(missing / (df.shape[0] * df.shape[1]) * 100, 1)
    num_cols    = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols    = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {
        "total_rows":           len(df),
        "total_cols":           len(df.columns),
        "numeric_features":     len([c for c in num_cols if c != target_col]),
        "categorical_features": len([c for c in cat_cols if c != target_col]),
        "missing_values":       int(missing),
        "missing_pct":          missing_pct,
        "duplicates":           int(df.duplicated().sum()),
        "target_unique":        int(df[target_col].nunique()),
    }


#  UNIFIED PREPROCESS  (tabular + text + mixed)

def preprocess(df: pd.DataFrame, target_col: str, problem_type: str):
    """
    Auto-detects dataset type and applies the right preprocessing:

    ┌──────────────┬────────────────────────────────────────────────────┐
    │ tabular      │ original pipeline (median impute + one-hot)        │
    ├──────────────┼────────────────────────────────────────────────────┤
    │ text_only    │ clean_text → TF-IDF (unigrams+bigrams, 15k feats)  │
    ├──────────────┼────────────────────────────────────────────────────┤
    │ mixed        │ TF-IDF for text cols + tabular pipeline for rest,  │
    │              │ then sparse-hstack both parts                      │
    └──────────────┴────────────────────────────────────────────────────┘

    Returns: (X, y, feature_names, class_names, le, meta)
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Drop columns with >40% missing
    thresh = int(0.6 * len(df))
    df.dropna(axis=1, thresh=thresh, inplace=True)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # ── Detect dataset type 
    dataset_type = detect_dataset_type(df, target_col)
    text_cols    = detect_text_columns(df, target_col)

    # ── Target encoding 
    y_raw = df[target_col].copy()
    y_raw = y_raw.fillna(y_raw.mode()[0] if not y_raw.mode().empty else 0)

    le, class_names = None, []
    if problem_type == "classification":
        le          = LabelEncoder()
        y           = le.fit_transform(y_raw.astype(str))
        class_names = list(le.classes_)
    else:
        y = y_raw.values.astype(float)

    X_parts        = []
    all_feat_names = []
    tfidf_vecs     = {}
    tab_scaler     = None

    # ══ A. TEXT columns → clean + TF-IDF 
    if text_cols:
        for col in text_cols:
            cleaned = clean_series(df[col])
            vec = TfidfVectorizer(
                max_features  = 15_000,
                ngram_range   = (1, 2),
                min_df        = 2,
                sublinear_tf  = True,
                strip_accents = "unicode",
                analyzer      = "word",
                token_pattern = r"\b[a-z]{2,}\b",
            )
            mat = vec.fit_transform(cleaned)
            tfidf_vecs[col] = vec
            X_parts.append(mat)
            all_feat_names.extend([f"{col}__tfidf__{t}" for t in vec.get_feature_names_out()])

    # ══ B. TABULAR columns → original pipeline 
    tab_cols = [c for c in df.columns if c != target_col and c not in text_cols]

    if tab_cols:
        X_tab = df[tab_cols].copy()

        # Drop near-constant and high-cardinality ID columns
        to_drop = (
            [c for c in X_tab.columns if X_tab[c].nunique() <= 1] +
            [c for c in X_tab.columns
             if X_tab[c].dtype == object and X_tab[c].nunique() / len(X_tab) > 0.9]
        )
        X_tab.drop(columns=to_drop, inplace=True)

        
        if X_tab.shape[1] == 0:
            pass  
        else:
            num_cols = X_tab.select_dtypes(include=np.number).columns.tolist()
            cat_cols = X_tab.select_dtypes(include=["object", "category"]).columns.tolist()

            # Impute numerics → median
            for c in num_cols:
                X_tab[c] = X_tab[c].fillna(X_tab[c].median())

            # Impute categoricals → mode, cap cardinality at 30, one-hot
            for c in cat_cols:
                mode_val = X_tab[c].mode()
                X_tab[c] = X_tab[c].fillna(mode_val[0] if not mode_val.empty else "Unknown")
                top = X_tab[c].value_counts().nlargest(30).index
                X_tab[c] = X_tab[c].where(X_tab[c].isin(top), "Other")

            X_tab = pd.get_dummies(X_tab, drop_first=True)

            # ── FIX: guard against empty frame after get_dummies 
            if X_tab.shape[1] == 0:
                pass  # get_dummies produced nothing — skip
            else:
                X_arr = X_tab.values.astype(np.float32)

                # Scale numeric columns only (RobustScaler)
                if num_cols:
                    num_idxs   = [i for i, c in enumerate(X_tab.columns) if c in num_cols]
                    tab_scaler = RobustScaler()
                    X_arr[:, num_idxs] = tab_scaler.fit_transform(X_arr[:, num_idxs])

                X_parts.append(csr_matrix(X_arr))
                all_feat_names.extend(list(X_tab.columns))

    # ══ Combine sparse parts 
    if not X_parts:
        raise ValueError("No usable feature columns found after preprocessing.")

    X_final = sp_hstack(X_parts, format="csr") if len(X_parts) > 1 else X_parts[0]

    meta = {
        "dataset_type": dataset_type,
        "text_cols":    text_cols,
        "tfidf_vecs":   tfidf_vecs,
        "tab_scaler":   tab_scaler,
    }

    return X_final, y, all_feat_names, class_names, le, meta


#  TRAIN ALL MODELS  (sparse-aware)

def train_all_models(X_tr, X_te, y_tr, y_te, problem: str, selected: list, dataset_type: str = "tabular"):
    import sklearn.base as skbase

    # Choose model pool
    if problem == "classification":
        pool = NLP_CLASSIFICATION_MODELS if dataset_type in ("text_only", "mixed") else CLASSIFICATION_MODELS
    else:
        pool = REGRESSION_MODELS

    pool = {k: skbase.clone(v) for k, v in pool.items()}

    if selected and selected != ["auto"]:
        pool = {k: v for k, v in pool.items() if k in selected}

    # Scaling: skip for sparse TF-IDF matrices (already log-normalised)
    if issparse(X_tr):
        X_tr_s, X_te_s = X_tr, X_te
        scaler = None
    else:
        scaler   = RobustScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

    results, trained = [], {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) \
         if problem == "classification" \
         else KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in pool.items():
        try:
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_te_s)

            if problem == "classification":
                acc = round(accuracy_score(y_te, pred) * 100, 2)
                f1  = round(f1_score(y_te, pred, average="weighted", zero_division=0) * 100, 2)

                try:
                    prob      = model.predict_proba(X_te_s)
                    n_classes = len(np.unique(y_te))
                    auc = round(
                        roc_auc_score(y_te, prob[:, 1]) if n_classes == 2
                        else roc_auc_score(y_te, prob, multi_class="ovr", average="weighted"),
                        4
                    )
                except Exception:
                    auc = None

                cv_scores = cross_val_score(model, X_tr_s, y_tr, cv=cv, scoring="accuracy")
                cv_mean   = round(cv_scores.mean() * 100, 2)
                cv_std    = round(cv_scores.std()  * 100, 2)

                results.append({
                    "model": name, "accuracy": acc, "f1_score": f1,
                    "auc_roc": auc, "cv_score": cv_mean, "cv_std": cv_std,
                    "primary": acc,
                })
            else:
                rmse = round(np.sqrt(mean_squared_error(y_te, pred)), 4)
                mae  = round(mean_absolute_error(y_te, pred), 4)
                r2   = round(r2_score(y_te, pred) * 100, 2)

                cv_scores = cross_val_score(model, X_tr_s, y_tr, cv=cv, scoring="r2")
                cv_mean   = round(cv_scores.mean() * 100, 2)
                cv_std    = round(cv_scores.std()  * 100, 2)

                results.append({
                    "model": name, "r2_score": r2, "rmse": rmse,
                    "mae": mae, "cv_score": cv_mean, "cv_std": cv_std,
                    "primary": r2,
                })

            trained[name] = (model, scaler)

        except Exception as e:
            results.append({"model": name, "error": str(e), "primary": -9999})

    results.sort(key=lambda x: x["primary"], reverse=True)
    return results, trained


#  CHARTS
# 

def make_charts(results, best_name, problem, X_te, y_te, trained, class_names):
    charts = {}

    plt.rcParams.update({
        "text.color":      "#c8d6f0",
        "axes.labelcolor": "#c8d6f0",
        "xtick.color":     "#7a8bb0",
        "ytick.color":     "#7a8bb0",
        "axes.edgecolor":  "#1a2540",
        "grid.color":      "#111a2e",
        "font.family":     "DejaVu Sans",
    })
    BG   = "#080d18"
    SURF = "#0d1525"

    valid  = [r for r in results if "error" not in r]
    names  = [r["model"] for r in valid]
    metric = "accuracy" if problem == "classification" else "r2_score"
    vals   = [r.get(metric, 0) for r in valid]
    colors = ["#00f5c4" if n == best_name else "#3d6aff" for n in names]

    # 1. Bar chart
    fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.55 + 1)), facecolor=BG)
    ax.set_facecolor(SURF)
    bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
    lbl  = metric.replace("_", " ").title() + " (%)"
    ax.set_xlabel(lbl, fontsize=11, labelpad=8)
    ax.set_title("Model Comparison", fontsize=15, fontweight="bold", color="#e0ebff", pad=14)
    ax.grid(axis="x", alpha=0.2, linewidth=0.8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#c8d6f0")
    ax.set_xlim(0, (max(vals) * 1.15) if vals else 100)
    fig.tight_layout(pad=1.5)
    charts["bar"] = fig_to_b64(fig)

    # 2. CV Score chart
    cv_vals = [r.get("cv_score", 0) for r in valid]
    cv_stds = [r.get("cv_std",   0) for r in valid]
    fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.55 + 1)), facecolor=BG)
    ax.set_facecolor(SURF)
    ax.barh(names, cv_vals, xerr=cv_stds, color="#00c897", edgecolor="none",
            height=0.6, error_kw={"ecolor": "#ffd166", "linewidth": 1.5, "capsize": 4})
    ax.set_xlabel("CV Score (%)", fontsize=11, labelpad=8)
    ax.set_title("5-Fold Cross-Validation Scores", fontsize=15, fontweight="bold",
                 color="#e0ebff", pad=14)
    ax.grid(axis="x", alpha=0.2, linewidth=0.8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout(pad=1.5)
    charts["cv"] = fig_to_b64(fig)

    # 3. Confusion matrix / Actual vs Predicted
    if best_name in trained:
        clf, scaler = trained[best_name]
        X_te_s = scaler.transform(X_te) if scaler is not None else X_te
        pred   = clf.predict(X_te_s)

        if problem == "classification":
            cm     = confusion_matrix(y_te, pred)
            labels = class_names[:cm.shape[0]] if class_names else [str(i) for i in range(cm.shape[0])]
            sz     = max(5, min(len(labels), 12))
            fig, ax = plt.subplots(figsize=(sz, sz - 1), facecolor=BG)
            ax.set_facecolor(SURF)
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={"shrink": 0.75}, linewidths=0.4, linecolor="#1a2540",
                        annot_kws={"size": 10 if len(labels) < 10 else 8})
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual",    fontsize=11)
            ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13,
                         fontweight="bold", color="#e0ebff", pad=10)
            fig.tight_layout(pad=1.5)
            charts["cm"] = fig_to_b64(fig)
        else:
            n   = min(400, len(y_te))
            idx = np.random.choice(len(y_te), n, replace=False)
            pred_idx = pred[idx]
            y_te_idx = y_te[idx]
            fig, ax  = plt.subplots(figsize=(8, 6), facecolor=BG)
            ax.set_facecolor(SURF)
            ax.scatter(y_te_idx, pred_idx, alpha=0.55, color="#3d6aff",
                       s=25, edgecolors="none", label="Samples")
            mn = min(y_te.min(), pred.min())
            mx = max(y_te.max(), pred.max())
            ax.plot([mn, mx], [mn, mx], "--", color="#00f5c4", linewidth=1.8, label="Perfect fit")
            ax.set_xlabel("Actual", fontsize=11); ax.set_ylabel("Predicted", fontsize=11)
            ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=13,
                         fontweight="bold", color="#e0ebff", pad=10)
            ax.legend(facecolor=SURF, edgecolor="#1a2540", fontsize=9)
            ax.grid(alpha=0.2, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout(pad=1.5)
            charts["scatter"] = fig_to_b64(fig)

    # 4. Radar chart (classification)
    if problem == "classification" and len(valid) >= 3:
        cats   = ["accuracy", "f1_score", "cv_score"]
        top5   = valid[:min(5, len(valid))]
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True}, facecolor=BG)
        ax.set_facecolor(SURF)
        ax.spines["polar"].set_color("#1a2540")
        palette = ["#00f5c4", "#3d6aff", "#00c897", "#ff6d00", "#d500f9"]
        for i, r in enumerate(top5):
            v = [min(r.get(c, 0), 100) for c in cats] + [min(r.get(cats[0], 0), 100)]
            ax.plot(angles, v, "o-", linewidth=2, color=palette[i], label=r["model"])
            ax.fill(angles, v, alpha=0.1, color=palette[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Accuracy", "F1 Score", "CV Score"], color="#c8d6f0", size=10)
        ax.set_ylim(0, 105)
        ax.tick_params(colors="#7a8bb0")
        ax.grid(color="#1a2540", linewidth=0.8)
        ax.set_title("Top Models Radar", fontsize=13, fontweight="bold", color="#e0ebff", pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15),
                  facecolor=SURF, edgecolor="#1a2140", fontsize=8)
        fig.tight_layout(pad=1.5)
        charts["radar"] = fig_to_b64(fig)

    # 5. Score distribution
    if len(valid) >= 4:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        ax.set_facecolor(SURF)
        score_data = [r.get(metric, 0) for r in valid]
        ax.boxplot(score_data, vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#3d6aff", color="#3d6aff", alpha=0.7),
                   medianprops=dict(color="#00f5c4", linewidth=2.5),
                   whiskerprops=dict(color="#7a8bb0"),
                   capprops=dict(color="#7a8bb0"),
                   flierprops=dict(markerfacecolor="#ffd166", marker="o", markersize=5))
        ax.scatter(score_data, [1] * len(score_data), color="#00f5c4",
                   alpha=0.7, s=40, zorder=5, edgecolors="none")
        ax.set_xlabel(lbl, fontsize=11)
        ax.set_title("Score Distribution Across Models", fontsize=13,
                     fontweight="bold", color="#e0ebff", pad=12)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks([])
        fig.tight_layout(pad=1.5)
        charts["dist"] = fig_to_b64(fig)

    return charts


# 
#  ROUTES
# 

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents     = await file.read()
        df           = pd.read_csv(BytesIO(contents))
        target       = detect_target(df)
        problem      = detect_problem(df[target])
        stats        = compute_dataset_stats(df, target)
        dataset_type = detect_dataset_type(df, target)
        text_cols    = detect_text_columns(df, target)
        return {
            **stats,
            "columns":          list(df.columns),
            "target_col":       target,
            "detected_problem": problem,
            "dataset_type":     dataset_type,
            "text_columns":     text_cols,
            "dtypes":           {c: str(df[c].dtype) for c in df.columns},
            "target_values":    df[target].value_counts().head(10).to_dict()
                                if problem == "classification" else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train(
    file:         UploadFile = File(...),
    problem_type: str        = Form("auto"),
    target_col:   str        = Form(""),
    models:       str        = Form("[]"),
    test_size:    float      = Form(0.2),
):
    try:
        contents   = await file.read()
        df         = pd.read_csv(BytesIO(contents))
        sel_models = json.loads(models)

        if not target_col:
            target_col = detect_target(df)
        if problem_type == "auto":
            problem_type = detect_problem(df[target_col])

        raw_stats = compute_dataset_stats(df, target_col)

        # ── Smart preprocess (handles tabular / text_only / mixed) 
        X, y, feature_names, class_names, le, meta = preprocess(df, target_col, problem_type)
        dataset_type = meta["dataset_type"]

        stratify = y if problem_type == "classification" and len(np.unique(y)) < 50 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        results, trained = train_all_models(
            X_tr, X_te, y_tr, y_te,
            problem_type,
            sel_models or ["auto"],
            dataset_type=dataset_type,
        )

        best = next((r for r in results if "error" not in r), None)
        if not best:
            raise HTTPException(status_code=500, detail="All models failed to train.")

        best_name = best["model"]
        charts    = make_charts(results, best_name, problem_type, X_te, y_te, trained, class_names)

        # ── Feature importance chart 
        clf_obj, scaler_obj = trained[best_name]
        if hasattr(clf_obj, "feature_importances_"):
            imp     = clf_obj.feature_importances_
            top_idx = np.argsort(imp)[::-1][:20]
            top_names = [feature_names[i] for i in top_idx]
            top_vals  = [imp[i] for i in top_idx]

            BG, SURF = "#080d18", "#0d1525"
            fig, ax  = plt.subplots(figsize=(10, max(5, len(top_names) * 0.45 + 1)), facecolor=BG)
            ax.set_facecolor(SURF)
            colors_fi = plt.cm.YlGnBu(np.linspace(0.4, 0.9, len(top_vals)))
            ax.barh(top_names[::-1], top_vals[::-1], color=colors_fi, edgecolor="none", height=0.65)
            ax.set_xlabel("Importance", fontsize=11, labelpad=8)
            ax.set_title(f"Feature Importance — {best_name}", fontsize=14,
                         fontweight="bold", color="#e0ebff", pad=12)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.grid(axis="x", alpha=0.2, linewidth=0.8)
            fig.tight_layout(pad=1.5)
            charts["importance"] = fig_to_b64(fig)

        # ── Top TF-IDF tokens chart (text datasets) 
        if dataset_type in ("text_only", "mixed") and hasattr(clf_obj, "coef_"):
            try:
                tfidf_vecs = meta["tfidf_vecs"]
                if tfidf_vecs:
                    first_col = list(tfidf_vecs.keys())[0]
                    vec       = tfidf_vecs[first_col]
                    terms     = vec.get_feature_names_out()

                    coef = clf_obj.coef_
                    if coef.ndim == 1:
                        top_pos = np.argsort(coef)[::-1][:15]
                        top_neg = np.argsort(coef)[:15]
                        combined_idx   = np.concatenate([top_neg, top_pos])
                        combined_vals  = coef[combined_idx]
                        combined_names = [terms[i] if i < len(terms) else f"feat_{i}" for i in combined_idx]

                        BG, SURF = "#080d18", "#0d1525"
                        fig, ax  = plt.subplots(figsize=(10, 7), facecolor=BG)
                        ax.set_facecolor(SURF)
                        bar_colors = ["#ff4d6d" if v < 0 else "#00f5c4" for v in combined_vals]
                        ax.barh(combined_names, combined_vals, color=bar_colors, edgecolor="none", height=0.65)
                        ax.axvline(0, color="#7a8bb0", linewidth=1)
                        ax.set_xlabel("Coefficient Weight", fontsize=11)
                        ax.set_title(f"Top Predictive Words — {best_name}", fontsize=14,
                                     fontweight="bold", color="#e0ebff", pad=12)
                        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
                        ax.grid(axis="x", alpha=0.2, linewidth=0.8)
                        fig.tight_layout(pad=1.5)
                        charts["top_words"] = fig_to_b64(fig)
            except Exception:
                pass  # non-critical chart, skip silently

        # ── Save best model + full pipeline 
        run_id     = str(uuid.uuid4())[:8]
        model_path = os.path.join(MODEL_DIR, f"best_model_{run_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":         clf_obj,
                "scaler":        scaler_obj,
                "feature_names": feature_names,
                "class_names":   class_names,
                "label_encoder": le,
                "problem_type":  problem_type,
                "target_col":    target_col,
                "dataset_type":  dataset_type,
                "text_cols":     meta["text_cols"],
                "tfidf_vecs":    meta["tfidf_vecs"],
                "tab_scaler":    meta["tab_scaler"],
            }, f)

        return {
            "run_id":        run_id,
            "target_col":    target_col,
            "problem_type":  problem_type,
            "dataset_type":  dataset_type,
            "text_cols":     meta["text_cols"],
            "n_features":    len(feature_names),
            "train_samples": len(y_tr),
            "test_samples":  len(y_te),
            "best_model":    best_name,
            "best_score":    best["primary"],
            "results":       results,
            "charts":        charts,
            "raw_stats":     raw_stats,
            "class_names":   class_names,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.get("/check/{run_id}")
async def check_model(run_id: str):
    if not run_id.replace("-", "").isalnum() or len(run_id) > 36:
        raise HTTPException(status_code=400, detail="Invalid run ID")
    path = os.path.join(MODEL_DIR, f"best_model_{run_id}.pkl")
    return {"exists": os.path.isfile(path)}


@app.get("/download/{run_id}")
async def download_model(run_id: str):
    if not run_id.replace("-", "").isalnum() or len(run_id) > 36:
        raise HTTPException(status_code=400, detail="Invalid run ID")
    model_path = os.path.realpath(os.path.join(MODEL_DIR, f"best_model_{run_id}.pkl"))
    real_dir   = os.path.realpath(MODEL_DIR)
    if not model_path.startswith(real_dir + os.sep):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for run ID: {run_id}")
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"automl_best_model_{run_id}.pkl",
    )
