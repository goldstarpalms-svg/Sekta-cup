from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class SourceItem:
    category: str
    name: str
    url: str
    use_case: str
    access_type: str
    integration_status: str
    notes: str


SOURCES: list[SourceItem] = [
    # Live scores / match history
    SourceItem(
        "Live Scores & Match History",
        "Flashscore Table Tennis",
        "https://www.flashscore.com/table-tennis/",
        "Live scores, schedules, historical result reference",
        "Website / licensed feed required",
        "Planned / manual reference",
        "Do not scrape by default. Use only permitted exports, licensed feeds, or manual research.",
    ),
    SourceItem(
        "Live Scores & Match History",
        "SofaScore Table Tennis",
        "https://www.sofascore.com/table-tennis",
        "Live scores, event pages, form reference",
        "Website / licensed feed required",
        "Planned / manual reference",
        "Useful for cross-checking player names and match timing when permission is available.",
    ),
    SourceItem(
        "Live Scores & Match History",
        "LiveScore.in Table Tennis",
        "https://www.livescore.in/table-tennis/",
        "Live scores and match history reference",
        "Website / licensed feed required",
        "Planned / manual reference",
        "Keep as a research/source link unless an approved feed is available.",
    ),
    SourceItem(
        "Live Scores & Match History",
        "BetExplorer Table Tennis",
        "https://www.betexplorer.com/table-tennis/",
        "Historical odds/results reference",
        "Website / licensed feed required",
        "Planned / manual reference",
        "Potentially useful for validating historical odds, subject to terms and access rights.",
    ),
    SourceItem(
        "Live Scores & Match History",
        "Scorebing",
        "https://www.scorebing.com/",
        "Score/result reference",
        "Website / licensed feed required",
        "Planned / manual reference",
        "Use as a discovery/reference source unless an approved API/feed is provided.",
    ),

    # Betting odds
    SourceItem(
        "Betting Odds",
        "The Odds API",
        "https://the-odds-api.com/",
        "Bookmaker odds, implied probability comparison",
        "API key",
        "Code scaffold active",
        "Integrated in src/odds_api.py and the Live Odds page. Add THE_ODDS_API_KEY to use.",
    ),
    SourceItem(
        "Betting Odds",
        "Pinnacle API",
        "https://developer.pinnacle.com/",
        "Sharp bookmaker odds, line movement, market reference",
        "Approved API credentials",
        "Connector scaffold planned",
        "Requires Pinnacle API access. Keep credentials out of Git.",
    ),
    SourceItem(
        "Betting Odds",
        "Betfair API",
        "https://api.betfair.com/",
        "Exchange prices, market liquidity, implied probability comparison",
        "App key + session token",
        "Connector scaffold planned",
        "Requires Betfair developer access and market availability in your region.",
    ),

    # Table tennis data
    SourceItem(
        "Table Tennis Data",
        "ITTF Results",
        "https://results.ittf.com/",
        "Official international match results",
        "Website / official data access",
        "Planned / manual import",
        "Good for non-Setka player/team context where permitted.",
    ),
    SourceItem(
        "Table Tennis Data",
        "World Table Tennis",
        "https://worldtabletennis.com/",
        "Official WTT schedules, rankings, results, news",
        "Website / official data access",
        "Planned / manual import",
        "Useful for broader player strength features outside Setka.",
    ),
    SourceItem(
        "Table Tennis Data",
        "TableTennis.Guide",
        "https://tabletennis.guide/",
        "Player profiles, rankings, historical reference",
        "Website / permitted data access",
        "Planned / manual import",
        "Potential identity-resolution source; check terms before automated use.",
    ),
    SourceItem(
        "Table Tennis Data",
        "Ratings Central",
        "https://www.ratingscentral.com/",
        "Independent table-tennis ratings and event records",
        "Website / permitted data access",
        "Planned / manual import",
        "Potential external rating feature source where permitted.",
    ),

    # Machine learning
    SourceItem(
        "Machine Learning",
        "scikit-learn",
        "https://scikit-learn.org/",
        "Baseline ML, gradient boosting fallback, metrics",
        "Python package",
        "Active",
        "Used by src/ml_pipeline.py.",
    ),
    SourceItem(
        "Machine Learning",
        "XGBoost",
        "https://xgboost.readthedocs.io/",
        "Gradient boosting models for winner/totals predictions",
        "Python package",
        "Active optional",
        "Used automatically when installed and algorithm='auto'.",
    ),
    SourceItem(
        "Machine Learning",
        "PyTorch",
        "https://pytorch.org/",
        "Neural-network experiments",
        "Optional Python package",
        "Optional future experiment",
        "Not in default requirements to keep deploy lightweight.",
    ),
    SourceItem(
        "Machine Learning",
        "TensorFlow",
        "https://www.tensorflow.org/",
        "Neural-network experiments",
        "Optional Python package",
        "Optional future experiment",
        "Not in default requirements to keep deploy lightweight.",
    ),
    SourceItem(
        "Machine Learning",
        "LightGBM",
        "https://lightgbm.readthedocs.io/",
        "Fast gradient boosting alternative",
        "Optional Python package",
        "Optional future experiment",
        "Listed in optional ML requirements.",
    ),
    SourceItem(
        "Machine Learning",
        "CatBoost",
        "https://catboost.ai/",
        "Gradient boosting with categorical support",
        "Optional Python package",
        "Optional future experiment",
        "Listed in optional ML requirements.",
    ),
    SourceItem(
        "Machine Learning",
        "Optuna",
        "https://optuna.org/",
        "Hyperparameter tuning",
        "Optional Python package",
        "Optional script added",
        "Use scripts/tune_winner_model.py after installing optional requirements.",
    ),
    SourceItem(
        "Machine Learning",
        "SHAP",
        "https://shap.readthedocs.io/",
        "Prediction explainability and feature importance",
        "Optional Python package",
        "Utility scaffold added",
        "Use src/explainability.py; falls back to sklearn permutation importance when SHAP is unavailable.",
    ),

    # Data analysis
    SourceItem("Data Analysis", "NumPy", "https://numpy.org/", "Numerical arrays/statistics", "Python package", "Active", "Used throughout the project."),
    SourceItem("Data Analysis", "Pandas", "https://pandas.pydata.org/", "CSV loading, feature engineering, tables", "Python package", "Active", "Core data engine."),
    SourceItem("Data Analysis", "SciPy", "https://scipy.org/", "Scientific/statistical tools", "Optional Python package", "Optional future", "Can be added for calibration/statistical testing."),
    SourceItem("Data Analysis", "Plotly Python", "https://plotly.com/python/", "Interactive charts", "Python package", "Active", "Used by the Streamlit dashboard."),

    # Training platforms
    SourceItem("Training", "Google Colab", "https://colab.research.google.com/", "Cloud notebooks/model training", "Notebook platform", "Notebook added", "See notebooks/Setka_ML_Training_Colab.ipynb."),
    SourceItem("Training", "Kaggle", "https://www.kaggle.com/", "Dataset experiments and notebooks", "Notebook platform", "Planned", "Project can be uploaded as a Kaggle dataset/notebook later."),

    # GitHub discovery
    SourceItem("GitHub", "table-tennis topic", "https://github.com/topics/table-tennis", "Open-source discovery", "Website", "Research link", "Use for inspiration and related libraries."),
    SourceItem("GitHub", "sports-prediction topic", "https://github.com/topics/sports-prediction", "Open-source model patterns", "Website", "Research link", "Use for architecture/model examples."),
    SourceItem("GitHub", "machine-learning topic", "https://github.com/topics/machine-learning", "ML project patterns", "Website", "Research link", "General ML resources."),
    SourceItem("GitHub", "xgboost topic", "https://github.com/topics/xgboost", "XGBoost examples", "Website", "Research link", "Useful for model training references."),

    # Research
    SourceItem("Research", "Google Scholar", "https://scholar.google.com/", "Academic papers", "Website", "Research link", "Search terms: table tennis prediction, Elo sports prediction, point total modelling."),
    SourceItem("Research", "arXiv", "https://arxiv.org/", "Preprints and ML papers", "Website", "Research link", "Useful for sports forecasting and calibration methods."),
    SourceItem("Research", "ResearchGate", "https://www.researchgate.net/", "Academic articles and author profiles", "Website", "Research link", "Use for literature discovery where accessible."),
]


def registry_dataframe(category: str | None = None) -> pd.DataFrame:
    rows = [asdict(item) for item in SOURCES]
    df = pd.DataFrame(rows)
    if category and category != "All":
        df = df[df["category"] == category]
    return df.reset_index(drop=True)


def categories() -> list[str]:
    return sorted({item.category for item in SOURCES})


def summary_by_category() -> pd.DataFrame:
    df = registry_dataframe()
    return (
        df.groupby("category")
        .agg(sources=("name", "count"), active_or_scaffold=("integration_status", lambda s: int(s.str.contains("Active|scaffold|added|Notebook", case=False, regex=True).sum())))
        .reset_index()
        .sort_values("category")
    )
