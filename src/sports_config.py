from __future__ import annotations

import pandas as pd

SPORTS = [
    {
        "sport": "Setka Cup Table Tennis",
        "key": "setka_table_tennis",
        "status": "Live prediction engine active",
        "coverage": "Official Setka fixtures/results + uploaded Setka history",
        "markets": "Winner, total points, first set O/U 18.5",
        "data_source": "Official Setka API + project CSV",
    },
    {
        "sport": "Football / Soccer",
        "key": "soccer",
        "status": "Odds/API scaffold ready",
        "coverage": "Requires odds/scores API key or permitted feed",
        "markets": "1X2, BTTS, O/U, double chance, handicap",
        "data_source": "The Odds API / future football-data provider",
    },
    {
        "sport": "Basketball",
        "key": "basketball",
        "status": "Odds/API scaffold ready",
        "coverage": "Requires odds/scores API key or permitted feed",
        "markets": "Moneyline, spreads, totals, team totals",
        "data_source": "The Odds API / future sports feed",
    },
    {
        "sport": "Tennis",
        "key": "tennis",
        "status": "Odds/API scaffold ready",
        "coverage": "Requires odds/scores API key or permitted feed",
        "markets": "Winner, set betting, game handicap, totals",
        "data_source": "The Odds API / future tennis feed",
    },
    {
        "sport": "Baseball",
        "key": "baseball",
        "status": "Odds/API scaffold ready",
        "coverage": "Requires odds/scores API key or permitted feed",
        "markets": "Moneyline, run line, totals",
        "data_source": "The Odds API / future sports feed",
    },
    {
        "sport": "Ice Hockey",
        "key": "ice_hockey",
        "status": "Odds/API scaffold ready",
        "coverage": "Requires odds/scores API key or permitted feed",
        "markets": "Moneyline, puck line, totals",
        "data_source": "The Odds API / future sports feed",
    },
    {
        "sport": "American Football",
        "key": "american_football",
        "status": "Odds/API scaffold ready",
        "coverage": "Requires odds/scores API key or permitted feed",
        "markets": "Moneyline, spreads, totals, props",
        "data_source": "The Odds API / future sports feed",
    },
]


REFERENCE_SCORE_SITES = [
    {
        "name": "Flashscore",
        "url": "https://www.flashscore.com/",
        "best_for": "Live scores, fixtures, form, fast result checking",
        "integration_status": "Reference link/manual cross-check unless licensed/API access is available",
    },
    {
        "name": "SofaScore",
        "url": "https://www.sofascore.com/",
        "best_for": "Live scores, stats, momentum, event pages",
        "integration_status": "Reference link/manual cross-check unless licensed/API access is available",
    },
    {
        "name": "BetExplorer",
        "url": "https://www.betexplorer.com/",
        "best_for": "Historical odds/results reference and bookmaker comparison",
        "integration_status": "Reference link/manual cross-check unless licensed/API access is available",
    },
]


def sports_dataframe() -> pd.DataFrame:
    return pd.DataFrame(SPORTS)


def reference_sites_dataframe() -> pd.DataFrame:
    return pd.DataFrame(REFERENCE_SCORE_SITES)


def supported_sport_names() -> list[str]:
    return [item["sport"] for item in SPORTS]
