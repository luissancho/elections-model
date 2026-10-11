"""
Server-rendered web pages: routes, the Jinja2 environment, formatting filters and page contexts.

The package has one module per page (``index``, ``promedio``, ``escanos``, ``casas``, ``elecciones``, ``modelo``), plus
``common`` (routes, messages, environment and shared context) and ``filters`` (formatting filters). This facade
re-exports every public name, so callers use ``pages.<name>``.

The pages are rendered by the controllers of ``mtpy/controllers`` (``Page`` and its subclasses)
from the templates of ``web/templates``. Every context is built from the published bundle through
``webapi.site()``, reading all the parts of a page with the same run, so a page never mixes runs;
the houses page adds the live tables of the database (``webapi.query_*``) and the houses analysis
of the scope, which is not tied to a run and is read as published.
The formatting filters follow the rules of ``web/dist/js/format.js``. Like ``webapi``, this module
does not import the model nor ``publish``, so the web workers stay light.
"""
from .casas import (ANALYSIS_CSV, HERDING_KEYS, MTYPE_LABELS, PATH_KEYS, RANKING_KEYS, analysis_part, casa_context,
                    casas_context, effects_table, herding_rows, house_effects_grid, profile_rows, ranking_rows)
from .common import (MESSAGES, MODE_LABELS, NOT_FOUND, OTHERS_COLOR, PAGES, REPO_ROOT, ROUTES, SITE_TITLE,
                     TEMPLATES_DIR, Catalog, api_url, block_color, catalog, common_context, environment, parse_state,
                     part, published_scopes, query_string, resolve_scope)
from .elecciones import (EVENT_CSV, SUMMARY_KEYS, eleccion_context, elecciones_context, event_part, horizon_tables,
                         published_events, resolve_event)
from .escanos import (CSV_PARTS, block_rows, csv_links, default_coalition, district_table, escanos_context,
                      region_rows, resolve_region, seat_rows)
from .filters import DASH, MONTHS, fmt_date, fmt_datetime, fmt_int, fmt_num, fmt_pct, fmt_prob, fmt_range
from .index import index_context, majority_rows
from .modelo import backtest_part, modelo_context, project
from .promedio import TABLE_ROWS, chart_polls, promedio_context, table_rows
