# Machine-Learning Roadmap

The app currently supports:

- scikit-learn HistGradientBoosting models
- optional XGBoost models
- command-line training via `scripts/train_models.py`
- Colab training via `notebooks/Setka_ML_Training_Colab.ipynb`

## Optional tools from the resource list

| Tool | Use | File/support |
|---|---|---|
| scikit-learn | Baselines, metrics, fallback gradient boosting | Active in `src/ml_pipeline.py` |
| XGBoost | Strong tabular gradient boosting | Active optional in `src/ml_pipeline.py` |
| LightGBM | Alternative fast gradient boosting | Add experiments after `pip install -r requirements-optional.txt` |
| CatBoost | Alternative boosting, categorical features | Add experiments after optional install |
| Optuna | Hyperparameter optimization | `scripts/tune_winner_model.py` |
| SHAP | Explainability | `src/explainability.py` |
| PyTorch | Neural nets/sequence models | Optional heavy install |
| TensorFlow | Neural nets/sequence models | Optional heavy install |

## Recommended modelling phases

### Phase 1: Backtest the existing rule and ML models

- Chronological split by date
- Metrics: accuracy, ROC-AUC, Brier score, log loss
- For totals: MAE, RMSE, calibration by line buckets

### Phase 2: Add odds features

When odds are available through The Odds API/Pinnacle/Betfair:

- bookmaker implied probability
- closing line value
- odds movement
- market total line
- over/under price difference

### Phase 3: Tune models

Install optional requirements:

```bash
pip install -r requirements-optional.txt
```

Run Optuna winner-model tuning:

```bash
python scripts/tune_winner_model.py --algorithm sklearn --trials 50 --max-training-rows 50000
```

If XGBoost is installed:

```bash
python scripts/tune_winner_model.py --algorithm xgboost --trials 50 --max-training-rows 50000
```

### Phase 4: Explain predictions

Use `src/explainability.py` to generate:

- native feature importance where available
- permutation importance fallback
- SHAP importance when `shap` is installed

### Phase 5: Deploy

Keep Streamlit deployment lightweight:

- default `requirements.txt` for the app
- optional requirements only for research/training
- trained `.joblib` artifacts in `models/`, but do not commit large model files unless necessary
