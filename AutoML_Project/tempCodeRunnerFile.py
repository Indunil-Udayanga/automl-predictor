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