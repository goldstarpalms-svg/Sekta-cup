# Setka Prediction App

Interactive Streamlit app for Setka Cup/table-tennis analysis, built from the uploaded Setka match-history CSV and leaderboard CSV.

It includes:

- official Setka upcoming-match feed integration
- Setka Trading Desk with live ticker, protection mode, stop-loss, bankroll caps, and GREEN/WATCH/NO BET decisions
- live Setka prediction board with confidence filters, pick-strength labels, mobile cards, and set-count Over/Under
- Live Match Center with in-play Setka scores, set scores, active player, auto-refresh, and official viewing links
- Owner Edge Engine with GREEN/WATCH/NO BET decisions, minimum value odds, and bankroll caps
- official Setka results/live-score checker with auto-refresh option
- automatic grading for saved/uploaded Setka prediction snapshots
- enhanced winner model with recent H2H, point-difference, set-difference, calibration, and upset-risk flags
- set-count markets: Over/Under 3.5 sets and Over/Under 4.5 sets
- time-split Accuracy Lab backtesting with recommended probability filters
- smart stake calculator for table-tennis selections with edge, EV, and safer fractional-Kelly sizing
- bet slip tools for table-tennis odds, combined payout, risk level, and CSV export
- transparent rule-blend prediction
- optional scikit-learn/XGBoost ML training
- odds/API integration scaffolds for future table-tennis odds
- a data-source/research registry for Setka/table-tennis resources collected during planning

## Main features

### Home / Setka dashboard

- Table-tennis-first landing dashboard
- Quick access to Setka live predictions, Setka results, Accuracy Lab, stake calculator, bet slip tools, player stats, and H2H lookup
- Focused on Setka Cup only for now

### Accuracy Lab

- Time-split holdout testing of recent Setka matches
- Winner, total-points, first-set, and sets Over/Under accuracy
- Accuracy by probability threshold
- Recommended live filters to avoid weak picks

### Smart Stake Calc

- Decimal odds payout calculator
- Implied probability and model edge
- Expected value estimate
- Full Kelly and safer 1/4 Kelly stake suggestion

### Bet Slip Tools

- Paste multiple decimal odds for table-tennis selections
- Calculate combined odds, implied probability, payout, and risk level
- Simple split-stake planning and CSV export

### Setka Trading Desk

- Premium command center for the owner
- Live match ticker and upcoming edge scan
- Protection/Balanced/Aggressive modes
- Daily stop-loss, daily risk cap, and stake caps
- GREEN/WATCH/NO BET decisions with minimum value odds
- Designed to reduce chasing after back-to-back losses

### Live Match Center

- Fetches official Setka live widget matches
- Shows current match score, set scores, active player, tournament, and location
- Auto-refresh option for near-live tracking
- Official Setka site/schedule links for watching where available

### Owner Edge Engine

- Scans upcoming Setka matches and selects only the best market per match
- Gives strict decisions: GREEN, WATCH, or NO BET
- Calculates fair odds and minimum value odds before a pick is acceptable
- Adds suggested stake caps from bankroll and risk settings
- Designed to reject weak games instead of forcing action

### Live Predictions

- Fetches upcoming matches from the official Setka API
- Converts match times to Lagos time
- Produces winner, total-points, and first-set Over/Under predictions
- Adds confidence, H2H count, expected total points, and expected first-set points
- Includes filters to hide weak picks, CSV export, browser-session snapshots, and mobile-friendly pick cards

### Results Checker

- Fetches official Setka tournament results and live widget scores
- Shows scheduled, live, and finished matches
- Calculates set-score text, first-set total, and match total points
- Grades stored prediction snapshots against actual results
- Can upload a downloaded prediction CSV and grade it against official results

### Match Predictor

- Match winner probability
- Expected total points
- Total-points Over/Under probability
- First-set Over/Under probability, default line **18.5**
- Head-to-head summary
- Player comparison table

### ML Lab

Train four models from the historical match data:

1. Match winner classifier
2. First-set Over 18.5 classifier
3. Total-points regressor
4. First-set-points regressor

The ML pipeline uses chronological pre-match features:

- rolling Elo
- career win rate
- recent form
- first-set win tendency
- first-set Over 18.5 tendency
- point-difference history
- total-points history
- direct H2H history

The app can use:

- `xgboost` if installed and selected
- scikit-learn `HistGradientBoosting` fallback

### Live Odds page

Prepared integration for [The Odds API](https://the-odds-api.com/):

- list available odds sports/markets for your API key
- fetch odds for a chosen table-tennis/sport key
- flatten bookmaker/market/outcome odds into a table
- calculate implied probabilities for decimal or American odds
- export odds to CSV

Additional scaffold clients are included for:

- Pinnacle API
- Betfair API-NG

No API key is included. Add your own key through environment variables or Streamlit secrets.

### Data Sources page

The app includes a structured source registry for the links you provided:

- live scores and match history: Flashscore, SofaScore, LiveScore.in, BetExplorer, Scorebing
- betting odds: The Odds API, Pinnacle, Betfair
- table-tennis data: ITTF, World Table Tennis, TableTennis.Guide, Ratings Central
- ML: scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, CatBoost, Optuna, SHAP
- analysis: NumPy, Pandas, SciPy, Plotly
- training: Colab, Kaggle
- GitHub/research discovery links

Compliance note: the app does **not** blindly scrape websites. Use official APIs, licensed feeds, permitted exports, or manual imports.

### Setka Cup official link

Includes the official Setka Cup website link:

- <https://tabletennis.setkacup.com/en/>

The app currently performs a lightweight availability/status check only. If you obtain an official Setka API/feed or have permission to scrape structured data, plug it into `src/setka_live.py`.

### Colab notebook

A Google Colab starter notebook is included:

```text
notebooks/Setka_ML_Training_Colab.ipynb
```

Use it to train models in Colab, download a `.joblib` model bundle, then place it in `models/`.

## Project structure

```text
setka-prediction-app/
├── app.py
├── requirements.txt
├── requirements-optional.txt
├── requirements-deep-learning.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── data/
│   ├── Setka_June_2025_to_Now.csv
│   └── setka_leaderboard.csv
├── docs/
│   ├── DATA_SOURCES.md
│   └── ML_ROADMAP.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── Setka_ML_Training_Colab.ipynb
├── scripts/
│   ├── live_predictions.py
│   ├── check_results.py
│   ├── train_models.py
│   └── tune_winner_model.py
└── src/
    ├── __init__.py
    ├── explainability.py
    ├── external_clients.py
    ├── ml_pipeline.py
    ├── odds_api.py
    ├── setka_core.py
    ├── setka_live.py
    └── source_registry.py
```

## Data currently included

- `data/Setka_June_2025_to_Now.csv`
  - 155,715 matches
  - date range: 2025-06-01 to 2026-07-24
  - score strings parsed into total points, first-set points, sets played, etc.
- `data/setka_leaderboard.csv`
  - leaderboard rows with Elo and match counts

## Run locally

```bash
cd setka-prediction-app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## CLI helpers

```bash
# Upcoming official Setka predictions
python scripts/live_predictions.py --limit 20

# Official result/live-score check
python scripts/check_results.py --date 2026-07-25 --period all
```

## Add API keys

### Environment variables

```bash
export THE_ODDS_API_KEY="your_api_key_here"
export PINNACLE_USERNAME="your_username"
export PINNACLE_PASSWORD="your_password"
export BETFAIR_APP_KEY="your_app_key"
export BETFAIR_SESSION_TOKEN="your_session_token"
streamlit run app.py
```

### Streamlit secrets

Copy:

```text
.streamlit/secrets.toml.example
```

to:

```text
.streamlit/secrets.toml
```

Then set your keys. `secrets.toml` is ignored by Git and should not be committed.

## Train ML models from command line

Quick training with the latest 50,000 orientation rows:

```bash
python scripts/train_models.py --algorithm auto --max-training-rows 50000
```

Full training:

```bash
python scripts/train_models.py --algorithm auto --max-training-rows 0
```

Save path defaults to:

```text
models/setka_ml_bundle.joblib
```

Model artifacts are ignored by Git because they can be regenerated.

## Optional ML tools

Install optional ML/research libraries:

```bash
pip install -r requirements-optional.txt
```

Tune the winner model with Optuna:

```bash
python scripts/tune_winner_model.py --algorithm sklearn --trials 50 --max-training-rows 50000
```

Heavy deep-learning frameworks are separated:

```bash
pip install -r requirements-deep-learning.txt
```

Recommended: install deep-learning frameworks only in Colab/Kaggle or a machine with enough disk/RAM.

## Use Google Colab

1. Push this project to GitHub.
2. Open `notebooks/Setka_ML_Training_Colab.ipynb` in Colab.
3. Change `REPO_URL` to your GitHub repo URL.
4. Run the notebook.
5. Download the trained `setka_ml_bundle.joblib` artifact if needed.

## More documentation

- `docs/DATA_SOURCES.md` — integration plan and source registry
- `docs/ML_ROADMAP.md` — ML tools, tuning, explainability, deployment notes

## Push to GitHub

After reviewing the files:

```bash
cd setka-prediction-app
git init
git add .
git commit -m "Initial Setka prediction app with ML and odds integration"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

If you want me to push it directly from this workspace, send the GitHub repository URL and make sure GitHub authentication is available.

## Important disclaimer

This app provides analytical estimates from historical data. It is **not** a guarantee, betting advice, or financial advice. Always validate and backtest before using predictions for real decisions.
