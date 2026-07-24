# Data Source Registry

This project keeps external data integrations modular. Some links are API-ready, while others should be treated as research/manual-reference sources unless you have permission, a paid feed, or an official API.

## Compliance rule

Do **not** blindly scrape websites. Before automating collection, check the provider's terms, robots policy, rate limits, and licensing. Prefer official APIs, licensed feeds, CSV exports, or manual imports.

## Live scores and match history

| Source | URL | Intended use | Status |
|---|---|---|---|
| Flashscore | https://www.flashscore.com/table-tennis/ | Live scores, schedule, result cross-checking | Manual/reference unless approved feed |
| SofaScore | https://www.sofascore.com/table-tennis | Live scores, event pages, form reference | Manual/reference unless approved feed |
| LiveScore.in | https://www.livescore.in/table-tennis/ | Live scores and match history reference | Manual/reference unless approved feed |
| BetExplorer | https://www.betexplorer.com/table-tennis/ | Historical results/odds reference | Manual/reference unless approved feed |
| Scorebing | https://www.scorebing.com/ | Score/result reference | Manual/reference unless approved feed |

## Betting odds

| Source | URL | Intended use | Status |
|---|---|---|---|
| The Odds API | https://the-odds-api.com/ | Bookmaker odds, implied probabilities | Active scaffold in `src/odds_api.py` |
| Pinnacle API | https://developer.pinnacle.com/ | Sharp odds and line movement | Client scaffold in `src/external_clients.py` |
| Betfair API | https://api.betfair.com/ | Exchange prices and liquidity | Client scaffold in `src/external_clients.py` |

### Required secrets

Use environment variables or Streamlit secrets; never commit credentials.

```bash
THE_ODDS_API_KEY="..."
PINNACLE_USERNAME="..."
PINNACLE_PASSWORD="..."
BETFAIR_APP_KEY="..."
BETFAIR_SESSION_TOKEN="..."
```

## Table-tennis data

| Source | URL | Intended use | Status |
|---|---|---|---|
| ITTF Results | https://results.ittf.com/ | Official international results | Planned/manual import |
| World Table Tennis | https://worldtabletennis.com/ | Official WTT rankings, schedules, results | Planned/manual import |
| TableTennis.Guide | https://tabletennis.guide/ | Player profiles and reference | Planned/manual import |
| Ratings Central | https://www.ratingscentral.com/ | Ratings and event records | Planned/manual import |

## Integration approach

1. **Raw layer**: save any permitted downloaded/API data exactly as received.
2. **Normalized layer**: convert to common columns: `date_time`, `competition`, `player1`, `player2`, `winner`, `score`, `source`, `source_event_id`.
3. **Identity resolution**: map player names across sources to the app's canonical names.
4. **Feature layer**: build ML-ready features: Elo, form, H2H, totals, odds, implied probabilities.
5. **Model/prediction layer**: compare model probabilities to market implied probabilities.

The current app already implements the Setka CSV normalized layer and ML feature layer. External sources can be plugged in without changing the dashboard structure.
