"""Publication of the model results into the web bundle.

Two layers live here:

- Pure exporters. Every ``export_*`` function takes an already executed ``Simulator`` (or its
  ``Forecaster``) and returns ``(data, frame)``: ``data`` follows the bundle schema of the same name
  and ``frame`` is its tabular twin for the CSV. They keep no state and touch no storage. The
  ``export_analysis_*``, ``export_poll_errors`` and ``export_error_data`` exporters take the summaries
  of a ``Computer`` instead (``export_analysis_herding`` returns two frames).
  ``export_backtest_table`` and ``export_backtest_meta`` convert the tables of ``backtest/results``.
- Orchestration. ``publish_forecast`` runs the model of a scope and writes an immutable run through a
  ``BundleWriter``; ``publish_analysis`` computes the houses analysis of a scope and overwrites it;
  ``publish_backtest`` converts the CSV tables of the model backtest of a scope and overwrites them;
  ``rebuild_history``, ``update_manifest``, ``point`` and ``unpublish`` move the
  pointers; ``loreg_guard`` and ``resolve_scopes`` decide what may be published. This layer reads and
  writes the bundle, queries the database (``db_stats``, ``next_event_date``, ``resolve_scopes``) and
  runs git for the provenance of each run (``provenance``).

The package has six submodules: ``common`` (constants, provenance, the publication guards and the scope helpers),
``forecast`` (the exporters of a run and ``publish_forecast``), ``analysis``, ``backtest``, ``events`` and ``pointers``
(manifest and run pointers). The public names are re-exported here.
"""
from .analysis import (
    ANALYSIS_EVENTS_COLUMNS, ANALYSIS_HE_COLUMNS, ANALYSIS_HERDING_COLUMNS, ERROR_DATA_COLUMNS,
    export_analysis_herding, export_analysis_house_effects, export_analysis_meta, export_error_data,
    export_poll_errors, POLL_ERRORS_COLUMNS, POLL_ERRORS_PARAMS, publish_analysis
)
from .backtest import (
    BACKTEST_BOOL_COLS, BACKTEST_FILES, BACKTEST_RUN_EXCLUDED, backtest_source, export_backtest_meta,
    export_backtest_table, PROB_PREFIXES, publish_backtest
)
from .common import (
    ATTRIBUTION, db_stats, ISO_DAY_RE, latest_entry, loreg_guard, next_event_date, NothingToPublish, PCT_DECIMALS,
    PROB_DECIMALS, provenance, PROVENANCE_PACKAGES, PublishRefused, records, REPO_ROOT, resolve_scopes, round_cols,
    SEATS_DECIMALS
)
from .events import (
    event_dates_to_publish, EVENT_MODEL_PARTS, EVENT_POLLSTER_COLUMNS, event_summary, export_event_meta,
    export_event_model, export_event_polls, export_event_pollsters, export_event_results, publish_event
)
from .forecast import (
    backfill_days, backfill_run_at, backfill_run_id, DEFAULT_CORRECTORS, DISP_COLUMNS, export_dispersion,
    export_dist, export_districts, export_fan, export_headline, export_house_effects, export_meta, export_polls,
    export_projection, export_scenario, export_series, export_summary, export_vote, HE_COLUMNS, headline_mode,
    MODE_EXPORTERS, n_polls, PCT_COLS, POLL_COLUMNS, PROB_COLS, publish_forecast, publish_mode, SEATS_COLS,
    SUMMARY_COLS
)
from .pointers import point, read_manifest, rebuild_history, unpublish, update_manifest
