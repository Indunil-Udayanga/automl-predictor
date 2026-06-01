'use strict';

/* ── Model lists ─────────────────────────────────────────────────────────── */
const CLS_MODELS = [
  'Logistic Regression','Random Forest','Gradient Boosting',
  'Extra Trees','AdaBoost','Decision Tree',
  'K-Nearest Neighbors','SVM','Naive Bayes','MLP Neural Network',
];
const REG_MODELS = [
  'Linear Regression','Ridge Regression','Lasso Regression',
  'ElasticNet','Random Forest','Gradient Boosting',
  'Extra Trees','AdaBoost','Decision Tree',
  'K-Nearest Neighbors','SVR','MLP Neural Network',
];
const LOAD_STEPS = [
  'Reading dataset…',
  'Detecting target column…',
  'Preprocessing features…',
  'Encoding categorical variables…',
  'Handling missing values…',
  'Splitting train/test sets…',
  'Training models…',
  'Running 5-fold cross-validation…',
  'Ranking all models…',
  'Generating visualisations…',
  'Saving best model…',
];

/* ── State ───────────────────────────────────────────────────────────────── */
let _file   = null;
let _runId  = null;
let _lastData = null;

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const qs  = sel => document.querySelector(sel);
const qsa = sel => document.querySelectorAll(sel);

/* ── Boot ────────────────────────────────────────────────────────────────── */
initToggles();
bindUpload();
buildModelChips(CLS_MODELS);

/* ── Option card toggles ──────────────────────────────────────────────────── */
function initToggles() {
  qsa('.option-card').forEach(card => {
    card.addEventListener('click', () => {
      const group = card.dataset.group;
      const val   = card.dataset.val;
      qsa(`.option-card[data-group="${group}"]`).forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      if (group === 'problem') {
        buildModelChips(val === 'regression' ? REG_MODELS : CLS_MODELS);
      }
      if (group === 'mode') {
        $('modelChips').style.display = val === 'custom' ? 'grid' : 'none';
      }
    });
  });
}

function getToggleVal(group) {
  const el = qs(`.option-card[data-group="${group}"].active`);
  return el ? el.dataset.val : null;
}

/* ── Model chip grid ──────────────────────────────────────────────────────── */
function buildModelChips(models) {
  const grid = $('modelChips');
  grid.innerHTML = '';
  models.forEach(m => {
    const chip = document.createElement('div');
    chip.className     = 'mc on';
    chip.dataset.model = m;
    chip.textContent   = m;
    chip.addEventListener('click', () => chip.classList.toggle('on'));
    grid.appendChild(chip);
  });
}

function getSelectedModels() {
  return Array.from(qsa('#modelChips .mc.on')).map(c => c.dataset.model);
}

/* ── Upload ───────────────────────────────────────────────────────────────── */
function bindUpload() {
  const zone = $('uploadZone');
  const inp  = $('fileInput');

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.csv')) onFile(f);
    else toast('Please drop a .csv file', 'err');
  });
  inp.addEventListener('change', e => { if (e.target.files[0]) onFile(e.target.files[0]); });
}

function onFile(file) {
  _file  = file;
  _runId = null;

  const zone = $('uploadZone');
  zone.classList.add('has-file');
  const info = $('fileInfo');
  info.textContent = `✓ ${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
  info.classList.add('visible');
  $('trainBtn').disabled = false;
  setStatus('Analysing…', false);

  const fd = new FormData();
  fd.append('file', file);

  fetch('/analyze', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (data.detail) { toast(data.detail, 'err'); setStatus('Error', false); return; }
      populateTargetSelect(data.columns, data.target_col);
      showDatasetChips(data);
      if (data.detected_problem) {
        const card = qs(`.option-card[data-group="problem"][data-val="${data.detected_problem}"]`);
        if (card) card.click();
      }
      setStatus('Ready · ' + data.total_rows.toLocaleString() + ' rows', true);
      toast(`Dataset loaded — target: "${data.target_col}"`, 'inf');
    })
    .catch(() => { setStatus('Error', false); toast('Failed to analyse file', 'err'); });
}

function populateTargetSelect(columns, detected) {
  const sel = $('targetSel');
  sel.innerHTML = `<option value="">Auto detect (${detected})</option>`;
  columns.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col; opt.textContent = col;
    if (col === detected) opt.selected = true;
    sel.appendChild(opt);
  });
}

function showDatasetChips(data) {
  const wrap = $('datasetChips');
  wrap.innerHTML = [
    ['Rows',    data.total_rows.toLocaleString()],
    ['Cols',    data.total_cols],
    ['Num',     data.numeric_features],
    ['Cat',     data.categorical_features],
    ['Missing', data.missing_pct + '%'],
    ['Dups',    data.duplicates],
  ].map(([k, v]) => `<div class="dc"><span>${k}</span><strong>${v}</strong></div>`).join('');
}

/* ── Training ──────────────────────────────────────────────────────────────── */
function startTraining() {
  if (!_file) { toast('Upload a CSV first', 'err'); return; }

  const problem = getToggleVal('problem') || 'auto';
  const mode    = getToggleVal('mode')    || 'auto';
  const target  = $('targetSel').value;
  const split   = parseInt($('splitRange').value) / 100;

  let models = [];
  if (mode === 'custom') {
    models = getSelectedModels();
    if (!models.length) { toast('Select at least one model', 'err'); return; }
  }

  _runId = null;
  $('trainBtn').disabled = true;
  showPane('loading');
  setStatus('Training…', false);
  startProgress();

  const fd = new FormData();
  fd.append('file',         _file);
  fd.append('problem_type', problem);
  fd.append('target_col',   target);
  fd.append('models',       JSON.stringify(models));
  fd.append('test_size',    split);

  fetch('/train', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      stopProgress(100);
      setTimeout(() => {
        $('trainBtn').disabled = false;
        if (data.detail || data.error) {
          showPane('empty');
          toast('Error: ' + (data.detail || data.error), 'err');
          setStatus('Error', false);
        } else {
          _runId    = data.run_id;
          _lastData = data;
          renderResults(data);
          showPane('results');
          toast(`Done! Best: ${data.best_model} · ${data.best_score.toFixed(1)}%`, 'ok');
          setStatus(`Best: ${data.best_model}`, true);
        }
      }, 400);
    })
    .catch(() => {
      stopProgress(0);
      $('trainBtn').disabled = false;
      showPane('empty');
      toast('Network error — is the server running?', 'err');
      setStatus('Error', false);
    });
}

/* Progress animation */
let _progIv = null;
let _step   = 0;

function startProgress() {
  _step = 0;
  $('loadStepsList').innerHTML = '';
  updateRing(0);
  $('loadStep').textContent = LOAD_STEPS[0];

  _progIv = setInterval(() => {
    if (_step < LOAD_STEPS.length - 1) {
      _step++;
      const pct = Math.round(_step / LOAD_STEPS.length * 88);
      updateRing(pct);
      $('loadStep').textContent = LOAD_STEPS[_step];

      const li = document.createElement('div');
      li.className = 'ls-item done';
      li.style.animationDelay = '0s';
      li.textContent = LOAD_STEPS[_step - 1];
      $('loadStepsList').appendChild(li);
      $('loadStepsList').scrollTop = 9999;
    }
  }, 700);
}

function stopProgress(pct) {
  clearInterval(_progIv);
  updateRing(pct);
  $('ringPct').textContent = pct + '%';
}

function updateRing(pct) {
  const circumference = 276;
  const offset = circumference - (pct / 100) * circumference;
  $('ringProgress').style.strokeDashoffset = offset;
  $('ringPct').textContent = pct + '%';
}

/* ── Render results ───────────────────────────────────────────────────────── */
function renderResults(data) {
  const isClf  = data.problem_type === 'classification';
  const valid  = data.results.filter(r => !r.error);
  const best   = valid[0];
  if (!best) return;

  /* Best banner */
  const pLabel = isClf ? 'Accuracy' : 'R² Score';
  const pVal   = isClf ? best.accuracy + '%' : best.r2_score + '%';
  $('bestBanner').innerHTML = `
    <span class="best-tag">🏆 BEST MODEL</span>
    <div>
      <div class="best-model-name">${data.best_model}</div>
      <div class="best-model-sub">${pLabel}: ${pVal} &nbsp;·&nbsp; CV: ${best.cv_score}% ± ${best.cv_std ?? 0}%</div>
    </div>
    <div class="best-score-big">${pVal}</div>
    <span class="best-problem-badge">${data.problem_type.toUpperCase()}</span>
    <button class="btn-dl" id="dlBtn" onclick="downloadModel()">
      <div class="dl-spin"></div>
      <span class="dl-text">⬇ Download .pkl</span>
    </button>
  `;

  /* Metric strip */
  const metrics = isClf
    ? [
        { key: 'ACCURACY',       val: best.accuracy + '%',        c: '#00c897' },
        { key: 'F1 SCORE',       val: best.f1_score + '%',        c: '#00e5ff' },
        { key: 'AUC-ROC',        val: best.auc_roc ?? 'N/A',     c: '#a855f7' },
        { key: 'CV SCORE',       val: best.cv_score + '%',        c: '#ffd166' },
        { key: 'CV STD',         val: '±' + (best.cv_std ?? 0) + '%', c: '#5a6e90' },
        { key: 'MODELS TRAINED', val: data.results.length,         c: '#3d6aff' },
      ]
    : [
        { key: 'R² SCORE', val: best.r2_score + '%', c: '#00c897' },
        { key: 'RMSE',     val: best.rmse,            c: '#ff6b6b' },
        { key: 'MAE',      val: best.mae,             c: '#ffd166' },
        { key: 'CV SCORE', val: best.cv_score + '%',  c: '#00e5ff' },
        { key: 'CV STD',   val: '±' + (best.cv_std ?? 0) + '%', c: '#5a6e90' },
        { key: 'MODELS',   val: data.results.length,  c: '#3d6aff' },
      ];

  $('metricStrip').innerHTML = metrics.map(m => `
    <div class="metric-card" style="--c:${m.c}">
      <div class="mc-val">${m.val}</div>
      <div class="mc-key">${m.key}</div>
    </div>
  `).join('');

  /* Dataset card */
  const s = data.raw_stats || {};
  $('datasetCard').innerHTML = `
    <div class="card-hdr">📦 Dataset Overview</div>
    <div class="info-grid">
      <div class="info-cell"><div class="ic-val">${(data.train_samples + data.test_samples).toLocaleString()}</div><div class="ic-lbl">Total Rows</div></div>
      <div class="info-cell"><div class="ic-val">${data.n_features}</div><div class="ic-lbl">Processed Features</div></div>
      <div class="info-cell"><div class="ic-val">${data.train_samples.toLocaleString()}</div><div class="ic-lbl">Train Samples</div></div>
      <div class="info-cell"><div class="ic-val">${data.test_samples.toLocaleString()}</div><div class="ic-lbl">Test Samples</div></div>
      <div class="info-cell"><div class="ic-val">${data.target_col}</div><div class="ic-lbl">Target Column</div></div>
      <div class="info-cell"><div class="ic-val">${data.problem_type.toUpperCase()}</div><div class="ic-lbl">Problem Type</div></div>
      <div class="info-cell"><div class="ic-val">${s.missing_pct ?? '—'}%</div><div class="ic-lbl">Missing Data</div></div>
      <div class="info-cell"><div class="ic-val">${s.duplicates ?? '—'}</div><div class="ic-lbl">Duplicates</div></div>
    </div>
  `;

  renderTable(data.results, isClf);
  renderCharts(data.charts);
  renderAnalysis(data, isClf);
}

/* ── Table ─────────────────────────────────────────────────────────────────── */
function renderTable(results, isClf) {
  const valid  = results.filter(r => !r.error);
  const maxP   = valid.length ? Math.max(...valid.map(r => r.primary)) : 1;
  const hdrs   = isClf
    ? ['#', 'Model', 'Accuracy', 'F1 Score', 'AUC-ROC', 'CV Score', 'CV Std']
    : ['#', 'Model', 'R² Score', 'RMSE', 'MAE', 'CV Score', 'CV Std'];

  let tbody = '';
  results.forEach((r, i) => {
    if (r.error) {
      tbody += `<tr>
        <td class="rank-num">${i+1}</td>
        <td>${r.model}</td>
        <td colspan="${hdrs.length - 2}"><span class="err-tag">⚠ ${r.error.slice(0, 80)}</span></td>
      </tr>`;
      return;
    }
    const ratio  = maxP > 0 ? (r.primary / maxP * 100) : 0;
    const colors = ['#00f5c4', '#00c8e0', '#3d6aff'];
    const color  = colors[i] || '#5a6e90';
    const icons  = ['🥇', '🥈', '🥉'];
    const icon   = icons[i] || (i + 1);

    const cells = isClf
      ? `<td>
           <div class="score-wrap">
             <span class="mono-val" style="color:${color}">${r.accuracy}%</span>
             <div class="score-bar"><div class="score-bar-fill" style="width:${ratio}%;background:${color}"></div></div>
           </div>
         </td>
         <td class="mono-val">${r.f1_score}%</td>
         <td class="mono-val">${r.auc_roc ?? '—'}</td>
         <td class="mono-val">${r.cv_score}%</td>
         <td class="mono-val">±${r.cv_std ?? 0}%</td>`
      : `<td>
           <div class="score-wrap">
             <span class="mono-val" style="color:${color}">${r.r2_score}%</span>
             <div class="score-bar"><div class="score-bar-fill" style="width:${Math.max(0,ratio)}%;background:${color}"></div></div>
           </div>
         </td>
         <td class="mono-val">${r.rmse}</td>
         <td class="mono-val">${r.mae}</td>
         <td class="mono-val">${r.cv_score}%</td>
         <td class="mono-val">±${r.cv_std ?? 0}%</td>`;

    tbody += `<tr class="${i === 0 ? 'row-best' : ''}">
      <td class="rank-num">${icon}</td>
      <td style="font-weight:${i===0?700:400};color:${i===0?'#00f5c4':'inherit'}">${r.model}</td>
      ${cells}
    </tr>`;
  });

  $('tblWrap').innerHTML = `
    <div class="card-hdr" style="padding:0 0 14px">
      📋 Model Rankings
      <span class="card-badge">${results.length} models</span>
    </div>
    <table>
      <thead><tr>${hdrs.map(h => `<th>${h}</th>`).join('')}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
}

/* ── Charts ────────────────────────────────────────────────────────────────── */
const CHART_META = [
  { key: 'bar',        title: 'Model Comparison',       full: true  },
  { key: 'cv',         title: '5-Fold Cross-Validation', full: true  },
  { key: 'cm',         title: 'Confusion Matrix',        full: false },
  { key: 'scatter',    title: 'Actual vs Predicted',     full: false },
  { key: 'radar',      title: 'Top Models Radar',        full: false },
  { key: 'dist',       title: 'Score Distribution',      full: false },
  { key: 'importance', title: 'Feature Importance',      full: true  },
];

function renderCharts(charts) {
  $('chartsGrid').innerHTML = CHART_META
    .filter(d => charts[d.key])
    .map(d => `
      <div class="chart-box ${d.full ? 'full' : ''}">
        <div class="chart-box-hdr"><span class="ch-dot"></span>${d.title}</div>
        <img src="data:image/png;base64,${charts[d.key]}" alt="${d.title}" loading="lazy">
      </div>
    `).join('');
}

/* ── Analysis Tab ──────────────────────────────────────────────────────────── */
function renderAnalysis(data, isClf) {
  const valid = data.results.filter(r => !r.error);
  const best  = valid[0];
  if (!best) return;

  const second = valid[1];
  const gap    = second
    ? Math.abs(best.primary - second.primary).toFixed(2)
    : '—';

  const overfitScore = best.cv_score != null
    ? Math.abs(best.primary - best.cv_score).toFixed(2)
    : null;

  function overfitClass(v) {
    if (v == null) return '';
    if (v < 3) return 'good'; if (v < 8) return 'warn'; return 'bad';
  }

  function scoreClass(v) {
    if (v >= 85) return 'good'; if (v >= 65) return 'warn'; return 'bad';
  }

  const bestScore = isClf ? best.accuracy : best.r2_score;

  /* Top 5 list */
  const maxScore = valid.length ? Math.max(...valid.map(r => r.primary)) : 1;
  const topItems = valid.slice(0, 5).map((r, i) => {
    const s    = isClf ? r.accuracy : r.r2_score;
    const bar  = Math.round((r.primary / maxScore) * 100);
    const icon = ['🥇','🥈','🥉'][i] || `#${i+1}`;
    return `<div class="top-item">
      <span class="top-rank">${icon}</span>
      <span class="top-name">${r.model}</span>
      <div class="top-bar"><div class="top-bar-fill" style="width:${bar}%"></div></div>
      <span class="top-score">${s}%</span>
    </div>`;
  }).join('');

  const metricName = isClf ? 'Accuracy' : 'R² Score';

  $('analysisWrap').innerHTML = `
    <div class="analysis-section">
      <div class="as-title">🎯 Training Verdict</div>
      <div class="verdict-grid">
        <div class="verdict-item">
          <div class="vi-label">Best ${metricName}</div>
          <div class="vi-val ${scoreClass(bestScore)}">${bestScore}%</div>
        </div>
        <div class="verdict-item">
          <div class="vi-label">Margin over #2</div>
          <div class="vi-val">${gap}%</div>
        </div>
        <div class="verdict-item">
          <div class="vi-label">Overfit Gap (test vs CV)</div>
          <div class="vi-val ${overfitClass(overfitScore)}">${overfitScore != null ? overfitScore + '%' : 'N/A'}</div>
        </div>
        <div class="verdict-item">
          <div class="vi-label">CV Stability (std)</div>
          <div class="vi-val ${overfitClass(best.cv_std ?? 0)}">±${best.cv_std ?? 0}%</div>
        </div>
        <div class="verdict-item">
          <div class="vi-label">Models Evaluated</div>
          <div class="vi-val">${data.results.length}</div>
        </div>
        <div class="verdict-item">
          <div class="vi-label">Models Failed</div>
          <div class="vi-val ${data.results.filter(r=>r.error).length > 0 ? 'warn' : 'good'}">
            ${data.results.filter(r=>r.error).length}
          </div>
        </div>
      </div>
    </div>
    <div class="analysis-section">
      <div class="as-title">🏅 Top 5 Models</div>
      <div class="top-list">${topItems}</div>
    </div>
    <div class="analysis-section">
      <div class="as-title">💡 Recommendations</div>
      ${buildRecommendations(best, second, data.problem_type, overfitScore)}
    </div>
  `;
}

function buildRecommendations(best, second, problem, overfitScore) {
  const tips = [];

  if (overfitScore != null && parseFloat(overfitScore) > 8) {
    tips.push('⚠️ <strong>High overfit gap detected.</strong> Consider adding regularisation, reducing model complexity, or gathering more training data.');
  }
  if (best.cv_std > 5) {
    tips.push('📊 <strong>High CV variance.</strong> The model performance is unstable across folds — try ensemble methods or hyperparameter tuning.');
  }
  if (best.primary < 60 && problem === 'classification') {
    tips.push('📉 <strong>Low accuracy.</strong> Consider feature engineering, more data, or check if the target column is correct.');
  }
  if (best.primary < 50 && problem === 'regression') {
    tips.push('📉 <strong>Low R² score.</strong> The relationship may be non-linear. Try Gradient Boosting or add more relevant features.');
  }
  if (second && Math.abs(best.primary - second.primary) < 1) {
    tips.push(`🔁 <strong>Close tie with ${second.model}.</strong> Both models perform similarly — compare inference speed and interpretability.`);
  }
  if (['Random Forest','Extra Trees','Gradient Boosting'].includes(best.model)) {
    tips.push(`🌲 <strong>${best.model} won.</strong> Tree ensembles excel at capturing non-linear patterns. Feature importance is available in the Charts tab.`);
  }
  if (best.model === 'Logistic Regression' || best.model === 'Linear Regression') {
    tips.push(`📐 <strong>Linear model won.</strong> Data appears linearly separable — this is great for interpretability and fast deployment.`);
  }
  if (tips.length === 0) {
    tips.push(`✅ <strong>Excellent results!</strong> ${best.model} achieves ${best.primary.toFixed(1)}% with stable cross-validation. Ready to deploy.`);
  }

  return tips.map(t => `<div style="padding:10px 14px;background:var(--surface3);border-radius:8px;border:1px solid var(--border);font-size:0.83rem;line-height:1.6;margin-bottom:8px">${t}</div>`).join('');
}

/* ── Download ──────────────────────────────────────────────────────────────── */
async function downloadModel() {
  if (!_runId) { toast('No model — train first', 'err'); return; }

  const btn    = $('dlBtn');
  const dlText = btn.querySelector('.dl-text');
  btn.disabled = true;
  btn.classList.add('loading');
  dlText.textContent = 'Checking…';

  try {
    const chk = await fetch(`/check/${_runId}`);
    if (!chk.ok) throw new Error(`Server ${chk.status}`);
    const { exists } = await chk.json();

    if (!exists) {
      toast('Model file not found on server — please re-train.', 'err');
      dlText.textContent = '⬇ Download .pkl';
      return;
    }

    dlText.textContent = 'Downloading…';
    const resp = await fetch(`/download/${_runId}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const blob   = await resp.blob();
    const url    = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href     = url;
    anchor.download = `automl_best_model_${_runId}.pkl`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);

    toast('Model downloaded successfully!', 'ok');
    dlText.textContent = '⬇ Download .pkl';

  } catch (err) {
    toast('Download failed: ' + err.message, 'err');
    dlText.textContent = '⬇ Download .pkl';
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}

/* ── Tab switching ─────────────────────────────────────────────────────────── */
function switchTab(name, el) {
  qsa('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  ['rankings','charts','analysis'].forEach(n => {
    $('tab' + n.charAt(0).toUpperCase() + n.slice(1)).style.display = n === name ? 'block' : 'none';
  });
}

/* ── Pane switching ────────────────────────────────────────────────────────── */
function showPane(name) {
  const map = { empty: 'emptyState', loading: 'loadingState', results: 'resultsPane' };
  Object.entries(map).forEach(([k, id]) => {
    $(id).style.display = k === name ? 'flex' : 'none';
  });
  if (name === 'results') $(map.results).style.display = 'flex';
}

/* ── Status pill ───────────────────────────────────────────────────────────── */
function setStatus(msg, ok) {
  const pill = $('statusPill');
  pill.querySelector('.status-dot').style.background = ok ? 'var(--accent)' : 'var(--accent5)';
  pill.lastChild.textContent = msg;
}

/* ── Reset ─────────────────────────────────────────────────────────────────── */
function resetAll() {
  _file = null; _runId = null; _lastData = null;
  $('fileInput').value      = '';
  $('fileInfo').classList.remove('visible');
  $('fileInfo').textContent = '';
  $('uploadZone').classList.remove('has-file', 'drag');
  $('datasetChips').innerHTML = '';
  $('targetSel').innerHTML    = '<option value="">Auto detect</option>';
  $('trainBtn').disabled      = true;
  showPane('empty');
  setStatus('Ready', true);
  toast('Reset complete', 'inf');
}

/* ── Toast ─────────────────────────────────────────────────────────────────── */
let _toastTimer = null;
function toast(msg, type = 'ok') {
  const el   = $('toast');
  el.textContent = msg;
  el.className   = `toast t-${type} show`;
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 4500);
}