# Project audit — 12 September 2026

The project has working foundations, but the checked-in version could not start reliably and did not expose the three requested datasets through the chatbot. This pass inspected every Python module, the three notebooks, dependency/setup files, database schema, API routes and dashboard. Verified repairs and unresolved product limitations are separated below. This is not a certification of production readiness.

## Repaired

| Finding | Result |
|---|---|
| Empty `pyproject.toml`, missing package initializer | Installable `src` package, pytest import configuration, notebook extras |
| `get_connection` imported widely but absent | Shared psycopg 3 connection factory restored |
| SQLAlchemy selected psycopg2 while requirements installed psycopg 3 | Consistent driver and safely encoded connection URL |
| Diagnostic scripts used different database defaults | All use the shared configuration |
| API import eagerly required Groq, Qdrant and downloaded embeddings | Structured analytics start independently; document services initialize only when requested and enabled |
| Chat only retrieved a single load total | Explicit date/range and dataset routing, technology filters, chart rows, summaries and sources |
| Dashboard called nonexistent data, upload and health routes | Historical and health routes implemented; dashboard now offers working chat/history/forecast pages; unsupported upload page removed |
| Forecast endpoint was a status placeholder | Actual in-memory training and recursive load forecasts, with bounded horizon |
| Missing load model artifact and no production training entry point | `--train` CLI added; API trains without requiring an artifact |
| Historical load simulation reused an all-history model | Retrains on observations strictly before the target day |
| `res_backtest.py` duplicated load forecast code | Replaced with a RES monthly evaluator |
| Row shifts treated missing observations as continuous hours | Timestamp-based 24/48/168-hour lag lookup, including the feature notebook |
| Validation accepted fractional periods, infinity and malformed values | Rejects malformed dates, numeric values, periods and empty datasets |
| Historical reads loaded entire tables | Date predicates now applied in PostgreSQL |
| PDF chunk IDs restarted at zero for every document | Content-derived UUIDs prevent different documents overwriting each other's chunks |
| RAG context used string representations rather than payload text | Uses document text and actual source metadata |
| Parser tests required absent untracked XLS file | Generated, isolated Excel fixtures for all three parsers |
| Missing dashboard/model dependencies and environment template | Added dependencies and `.env.example`; replaced outdated README with runnable instructions |
| Download interruption could leave a partial cached file | Downloads use temporary files and atomic replacement; reject empty responses |

## Verified evidence

- **28 automated tests passed**: parsers, validation, routing, date errors, chart response schema, aggregation, no-data behavior, lag gaps, recursive forecasting and historical training cutoff. Two deprecation warnings originate from installed HTTP test dependencies.
- Python compilation and `git diff --check` passed.
- Streamlit AppTest opened all four pages with zero exceptions. This is a page-render smoke test, not a full browser interaction test.
- Read-only local PostgreSQL checks succeeded:

| Table | Rows | Earliest date | Latest date |
|---|---:|---|---|
| `system_load` | 33,475 | 2023-01-01 | 2026-08-31 |
| `res_production` | 33,475 | 2023-01-01 | 2026-08-31 |
| `generation_actual` | 145,632 | 2026-01-01 | 2026-01-31 |

- Real API chat requests for 2026-01-15 returned HTTP 200: load 24 rows / 127,994 MWh; RES 24 rows / 31,289 MWh; generation 96 source-hour rows / 165,354 MWh.
- An actual Random Forest trained on 31,861 model-ready rows and returned 24 forecast rows for 2026-09-01 through the chat API. Successful execution does **not** establish forecast accuracy.
- Public ADMIE XLS samples for 2026-01-15 were downloaded outside the repository, parsed and validated for all three categories. Generation unit sums matched the source's technology subtotals. Load and generation clock-change samples for 2025-03-30 and 2025-10-26 were also inspected.
- No historical database records or saved forecasts were changed during verification.

## Subsequent year-ahead forecast change

Chat/API now support one target day up to two calendar years after the latest load observations. Requests beyond 14 days use a separate calendar model with a held-out-year evaluation and descriptive historical error bounds. This supersedes the original 14-day API limit; it does not resolve the daylight-saving or structural-change limitations below. See [YEAR_AHEAD_VALIDATION.md](YEAR_AHEAD_VALIDATION.md). The expanded automated suite contains 36 passing tests.

## Remaining limitations, in priority order

1. **Daylight-saving interpretation remains unresolved.** The 2025-10-26 load file has a genuine period 25 of 4,099 MWh. Existing queries exclude it. The generation sample exposes only 24 periods, while the spring sample contains a terminal placeholder. It would be incorrect to invent a common timezone conversion from these files alone. API responses and README disclose the nominal 24-period convention. Before exact daily totals/clock-time charts are released, verify ADMIE interval conventions, preserve the full reporting identity, and test all three datasets across clock changes. Do not simply relabel period 25 as midnight on the following day.
2. **Dataset freshness and coverage differ.** Generation only covers January 2026 locally. Load and RES end on August 31. Ingest the requested periods and add a scheduled refresh plus completeness monitoring. Min/max dates and counts alone do not prove every value is valid.
3. **Chat language coverage is bounded.** It supports documented English/Greek keywords and date forms, not arbitrary questions, multiple analytical intents, per-request hour extraction, or unrestricted conversational reasoning. Bounded follow-up context was added subsequently; supported patterns and API context are documented in README. Wind/solar are not separate verified technology categories; current grouping retains ADMIE's RES bucket. A grounded intent schema with clarification behavior is the next step for unrestricted natural-language requests.
4. **Forecast validation is incomplete.** Monthly scripts retain fixed 2026 windows and use observed lag inputs. Run and record chronological baseline comparisons, horizon-specific recursive backtests and prediction intervals before claiming predictive quality. The 500 MWh load cutoff is inherited and needs domain validation. Clock-change mapping also affects training features.
5. **Ingestion assumes a known workbook layout.** Only sampled files were verified. Overlapping revisions fail duplicate validation rather than selecting a canonical latest publication. Add revision provenance and explicit format detection before bulk ingestion of all historical revisions. Raw-file caching does not detect a provider replacing content at the same URL.
6. **Forecast requests retrain synchronously.** This is functional for local use but expensive under concurrent traffic. A scheduled model-training job, versioned artifacts and a cache keyed by data freshness are needed for a deployed service. Saved forecast tables overwrite each timestamp and do not retain forecast-origin/model history.
7. **Optional RAG was not tested live.** Groq calls, embedding downloads and Qdrant indexing were not exercised. The optional Compose profile supplies Qdrant, but there is still no document upload API. Updated document versions can leave older chunks indexed; document lifecycle handling is still needed.
8. **Reproducibility and deployment remain basic.** Dependencies are lower bounds rather than a tested lockfile. No CI workflow or application Dockerfile is present. Add a dependency lock, automated checks, readiness reporting, and deployment configuration when preparing a release. Notebook code was reviewed and obvious connection/lag issues fixed; notebooks were not executed end to end.

## References checked

The data evidence came directly from ADMIE's `getOperationMarketFilewRange` endpoint and its returned XLS links, including the [load sample](https://www.admie.gr/sites/default/files/attached-files/type-file/2026/01/20260115_RealTimeSCADASystemLoad_01.xls), [RES sample](https://www.admie.gr/sites/default/files/attached-files/type-file/2026/01/20260115_RealTimeSCADARES_01.xls) and [generation sample](https://www.admie.gr/sites/default/files/attached-files/type-file/2026/01/20260115_SystemRealizationSCADA_01.xls). The optional Compose image uses the published [Qdrant v1.17.0 release](https://github.com/qdrant/qdrant/releases/tag/v1.17.0).
