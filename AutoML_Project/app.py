import os, json, uuid, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, base64
from io import BytesIO
import pickle

from flask import Flask, request, jsonify, send_file, render_template, abort
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

#  App setup 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)
app.config['UPLOAD_FOLDER']       = UPLOAD_DIR
app.config['MODEL_FOLDER']        = MODEL_DIR
app.config['MAX_CONTENT_LENGTH']  = 50 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

#  Model registries 
CLASSIFICATION_MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Extra Trees':         ExtraTreesClassifier(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=50, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(probability=True, random_state=42),
    'Naive Bayes':         GaussianNB(),
    'MLP Neural Network':  MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
}

REGRESSION_MODELS = {
    'Linear Regression':   LinearRegression(),
    'Ridge Regression':    Ridge(alpha=1.0),
    'Lasso Regression':    Lasso(alpha=1.0, max_iter=5000),
    'ElasticNet':          ElasticNet(alpha=1.0, max_iter=5000),
    'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Extra Trees':         ExtraTreesRegressor(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostRegressor(n_estimators=50, random_state=42),
    'Decision Tree':       DecisionTreeRegressor(random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'SVR':                 SVR(),
    'MLP Neural Network':  MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
}

#  Helpers 
def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#080c14', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def detect_target(df):
    keywords = ['target','label','class','output','result','price','sale',
                'churn','survived','diagnosis','outcome','y','default','fraud','spam']
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in keywords:
        if kw in cols_lower:
            return cols_lower[kw]
    return df.columns[-1]


def detect_problem(series):
    if series.dtype == object or series.dtype.name == 'category':
        return 'classification'
    if series.nunique() <= 20 or series.nunique() / len(series) < 0.05:
        return 'classification'
    return 'regression'


def preprocess(df, target_col):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    thresh = int(0.4 * len(df))
    df.dropna(axis=1, thresh=thresh, inplace=True)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col].copy()
    X     = df.drop(columns=[target_col])

    # Drop high-cardinality string columns
    to_drop = [c for c in X.columns if X[c].nunique() / len(X) > 0.95 and X[c].dtype == object]
    X.drop(columns=to_drop, inplace=True)

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    for c in num_cols:
        X[c].fillna(X[c].median(), inplace=True)
    for c in cat_cols:
        X[c].fillna(X[c].mode()[0] if not X[c].mode().empty else 'Unknown', inplace=True)
        top = X[c].value_counts().nlargest(20).index
        X[c] = X[c].where(X[c].isin(top), 'Other')

    X = pd.get_dummies(X, drop_first=True)
    y_raw.fillna(y_raw.mode()[0] if not y_raw.mode().empty else 0, inplace=True)

    if y_raw.dtype == object or y_raw.dtype.name == 'category':
        enc = pd.get_dummies(y_raw, drop_first=False)
        y   = np.argmax(enc.values, axis=1) if enc.shape[1] > 1 else enc.iloc[:, 0].values
        class_names = list(enc.columns)
    else:
        y = y_raw.values
        class_names = sorted(y_raw.unique().tolist()) if y_raw.nunique() <= 20 else []

    return X.values.astype(np.float32), y, list(X.columns), class_names


def train_models(X_tr, X_te, y_tr, y_te, problem, selected):
    pool = CLASSIFICATION_MODELS if problem == 'classification' else REGRESSION_MODELS
    if selected and selected != ['auto']:
        pool = {k: v for k, v in pool.items() if k in selected}

    scaler   = RobustScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)
    results, trained = [], {}

    for name, clf in pool.items():
        try:
            clf.fit(X_tr_s, y_tr)
            pred = clf.predict(X_te_s)

            if problem == 'classification':
                acc = round(accuracy_score(y_te, pred) * 100, 2)
                f1  = round(f1_score(y_te, pred, average='weighted', zero_division=0) * 100, 2)
                try:
                    prob = clf.predict_proba(X_te_s)
                    auc  = round(
                        roc_auc_score(y_te, prob, multi_class='ovr', average='weighted')
                        if prob.shape[1] > 1 else roc_auc_score(y_te, prob[:, 1]), 4)
                except Exception:
                    auc = None
                cv = round(cross_val_score(clf, X_tr_s, y_tr, cv=3, scoring='accuracy').mean() * 100, 2)
                results.append({'model': name, 'accuracy': acc, 'f1_score': f1,
                                'auc_roc': auc, 'cv_score': cv, 'primary': acc})
            else:
                rmse = round(np.sqrt(mean_squared_error(y_te, pred)), 4)
                mae  = round(mean_absolute_error(y_te, pred), 4)
                r2   = round(r2_score(y_te, pred) * 100, 2)
                cv   = round(cross_val_score(clf, X_tr_s, y_tr, cv=3, scoring='r2').mean() * 100, 2)
                results.append({'model': name, 'r2_score': r2, 'rmse': rmse,
                                'mae': mae, 'cv_score': cv, 'primary': r2})
            trained[name] = (clf, scaler)
        except Exception as e:
            results.append({'model': name, 'error': str(e), 'primary': -999})

    results.sort(key=lambda x: x['primary'], reverse=True)
    return results, trained


def make_charts(results, best_name, problem, X_te, y_te, trained, class_names):
    charts = {}
    plt.rcParams.update({
        'text.color':   '#c8d4f0', 'axes.labelcolor': '#c8d4f0',
        'xtick.color':  '#7a8bb0', 'ytick.color':     '#7a8bb0',
        'axes.edgecolor': '#1e2a40', 'grid.color':    '#141e30',
    })
    bg, surface = '#080c14', '#0e1622'

    valid  = [r for r in results if 'error' not in r]
    names  = [r['model'] for r in valid]
    metric = 'accuracy' if problem == 'classification' else 'r2_score'
    vals   = [r.get(metric, 0) for r in valid]
    colors = ['#00e5ff' if n == best_name else '#2d5afe' for n in names]

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=bg)
    ax.set_facecolor(surface)
    bars = ax.barh(names, vals, color=colors, edgecolor='none', height=0.55)
    ax.set_xlabel(metric.replace('_', ' ').title() + ' (%)', fontsize=11)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold', color='#e8f0ff', pad=12)
    ax.grid(axis='x', alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9, color='#c8d4f0')
    ax.set_xlim(0, (max(vals) * 1.12) if vals else 100)
    fig.tight_layout()
    charts['bar'] = fig_to_b64(fig)

    # CV chart
    cv_vals = [r.get('cv_score', 0) for r in valid]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=bg)
    ax.set_facecolor(surface)
    ax.barh(names, cv_vals, color='#00c897', edgecolor='none', height=0.55)
    ax.set_xlabel('CV Score (%)', fontsize=11)
    ax.set_title('Cross-Validation Scores (3-Fold)', fontsize=14, fontweight='bold', color='#e8f0ff', pad=12)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    charts['cv'] = fig_to_b64(fig)

    # Confusion matrix / scatter
    if best_name in trained:
        clf, scaler = trained[best_name]
        pred = clf.predict(scaler.transform(X_te))

        if problem == 'classification':
            cm     = confusion_matrix(y_te, pred)
            labels = class_names[:cm.shape[0]] if class_names else [str(i) for i in range(cm.shape[0])]
            fig, ax = plt.subplots(
                figsize=(max(5, len(labels)), max(4, len(labels) - 1)), facecolor=bg)
            ax.set_facecolor(surface)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='#1e2a40')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13,
                         fontweight='bold', color='#e8f0ff', pad=10)
            fig.tight_layout()
            charts['cm'] = fig_to_b64(fig)
        else:
            idx = np.random.choice(len(y_te), min(300, len(y_te)), replace=False)
            fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg)
            ax.set_facecolor(surface)
            ax.scatter(y_te[idx], pred[idx], alpha=0.5, color='#2d5afe', s=22, edgecolors='none')
            mn = min(y_te.min(), pred.min()); mx = max(y_te.max(), pred.max())
            ax.plot([mn, mx], [mn, mx], '--', color='#00e5ff', linewidth=1.5, label='Perfect fit')
            ax.set_xlabel('Actual', fontsize=11); ax.set_ylabel('Predicted', fontsize=11)
            ax.set_title(f'Actual vs Predicted — {best_name}', fontsize=13,
                         fontweight='bold', color='#e8f0ff', pad=10)
            ax.legend(facecolor=surface, edgecolor='#1e2a40')
            ax.grid(alpha=0.25)
            fig.tight_layout()
            charts['scatter'] = fig_to_b64(fig)

    # Radar
    if problem == 'classification' and valid:
        cats   = ['accuracy', 'f1_score', 'cv_score']
        top5   = valid[:5]
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True}, facecolor=bg)
        ax.set_facecolor(surface)
        ax.spines['polar'].set_color('#1e2a40')
        palette = ['#00e5ff', '#2d5afe', '#00c897', '#ff6d00', '#d500f9']
        for i, r in enumerate(top5):
            v = [r.get(c, 0) for c in cats] + [r.get(cats[0], 0)]
            ax.plot(angles, v, 'o-', linewidth=1.5, color=palette[i], label=r['model'])
            ax.fill(angles, v, alpha=0.08, color=palette[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Accuracy', 'F1', 'CV Score'], color='#c8d4f0', size=10)
        ax.set_ylim(0, 105)
        ax.set_title('Top-5 Radar', fontsize=13, fontweight='bold', color='#e8f0ff', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
                  facecolor=surface, edgecolor='#1e2a40', fontsize=8)
        fig.tight_layout()
        charts['radar'] = fig_to_b64(fig)

    return charts


#  Routes 
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    try:
        df       = pd.read_csv(request.files['file'])
        target   = detect_target(df)
        prob     = detect_problem(df[target])
        return jsonify({
            'rows': len(df), 'cols': len(df.columns),
            'columns': list(df.columns),
            'target_col': target,
            'detected_problem': prob,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    f            = request.files['file']
    problem_type = request.form.get('problem_type', 'auto')
    target_col   = request.form.get('target_col', '')
    sel_models   = json.loads(request.form.get('models', '[]'))
    test_size    = float(request.form.get('test_size', 0.2))

    try:
        df = pd.read_csv(f)
        if not target_col:
            target_col = detect_target(df)
        if problem_type == 'auto':
            problem_type = detect_problem(df[target_col])

        X, y, feature_names, class_names = preprocess(df, target_col)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if problem_type == 'classification' and len(np.unique(y)) < 50 else None
        )

        results, trained = train_models(X_tr, X_te, y_tr, y_te,
                                        problem_type, sel_models or ['auto'])

        best = next((r for r in results if 'error' not in r), None)
        if not best:
            return jsonify({'error': 'All models failed to train.'}), 500

        best_name = best['model']
        charts    = make_charts(results, best_name, problem_type, X_te, y_te, trained, class_names)

        # Save model
        run_id     = str(uuid.uuid4())[:8]
        model_path = os.path.join(MODEL_DIR, f'best_model_{run_id}.pkl')
        clf_obj, scaler_obj = trained[best_name]
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model':         clf_obj,
                'scaler':        scaler_obj,
                'feature_names': feature_names,
                'problem_type':  problem_type,
                'target_col':    target_col,
            }, f)

        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file could not be saved.'}), 500

        return jsonify({
            'run_id':        run_id,
            'target_col':    target_col,
            'problem_type':  problem_type,
            'n_features':    len(feature_names),
            'train_samples': len(X_tr),
            'test_samples':  len(X_te),
            'best_model':    best_name,
            'best_score':    best['primary'],
            'results':       results,
            'charts':        charts,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/check/<run_id>')
def check_model(run_id):
    if not run_id.replace('-', '').isalnum() or len(run_id) > 36:
        return jsonify({'exists': False}), 400
    path = os.path.join(MODEL_DIR, f'best_model_{run_id}.pkl')
    return jsonify({'exists': os.path.isfile(path)})


@app.route('/download/<run_id>')
def download_model(run_id):
    if not run_id.replace('-', '').isalnum() or len(run_id) > 36:
        abort(400)

    model_path = os.path.join(MODEL_DIR, f'best_model_{run_id}.pkl')
    real_path  = os.path.realpath(model_path)
    real_dir   = os.path.realpath(MODEL_DIR)

    if not real_path.startswith(real_dir + os.sep):
        abort(403)
    if not os.path.isfile(real_path):
        return jsonify({'error': f'Model not found for run ID: {run_id}. Re-train to regenerate.'}), 404

    return send_file(
        real_path,
        as_attachment=True,
        download_name=f'automl_best_model_{run_id}.pkl',  # <-- .pkl here
        mimetype='application/octet-stream',
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)