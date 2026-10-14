"""
Server-rendered web pages: routes, the Jinja2 environment, formatting filters and page contexts.

The package has one module per page (``inicio``, ``promedios``, ``escanos``, ``casas``, ``elecciones``, ``metodo``,
``backtest``), plus ``common`` (routes, messages, environment and shared context) and ``filters`` (formatting
filters). This facade re-exports every public name, so callers use ``pages.<name>``.

The pages are rendered by the controllers of ``mtpy/controllers`` (``Page`` and its subclasses)
from the templates of ``web/templates``. Every context is built from the published bundle through
``webapi.site()``, reading all the parts of a page with the same run, so a page never mixes runs;
the houses page adds the live tables of the database (``webapi.query_*``) and the houses analysis
of the scope, which is not tied to a run and is read as published.
The formatting filters follow the rules of ``web/dist/js/format.js``. Like ``webapi``, this module
does not import the model nor ``publish``, so the web workers stay light.
"""
from .backtest import backtest_context, backtest_part, project
from .casas import (ANALYSIS_CSV, HERDING_KEYS, MTYPE_LABELS, PATH_KEYS, RANKING_KEYS, analysis_part, casa_context,
                    casas_context, effects_table, herding_rows, house_effects_grid, profile_rows, ranking_rows)
from .common import (EVENT_TABS, MESSAGES, NOT_FOUND, OTHERS_COLOR, PAGES, REPO_ROOT, ROUTES, SITE_TAGLINE,
                     SITE_TITLE, TEMPLATES_DIR, Catalog, api_url, block_color, catalog, common_context, days_left, days_text,
                     default_mode, environment, event_info, event_query, event_title, ink_color, mode_text, parse_state,
                     part, published_scopes, resolve_event_scope, resolve_scope, scope_query, title_of)
from .elecciones import (EVENT_CSV, SUMMARY_KEYS, eleccion_context, elecciones_context, error_rows, event_part,
                         history_headline, horizon_tables, published_events, resolve_event, with_abs)
from .escanos import (CSV_PARTS, block_rows, csv_links, default_coalition, district_table, escanos_context,
                      escanos_headline_text, region_rows, resolve_region, seat_rows)
from .filters import DASH, MONTHS, fmt_date, fmt_datetime, fmt_int, fmt_num, fmt_pct, fmt_prob, fmt_range
from .inicio import INTRO, card_headline, event_cards, inicio_context, past_events
from .metodo import (METODO_SECTIONS, NATIONAL_SCOPE, SECTIONS, method_href, metodo_context, metodo_section_context,
                     national_figures, subpages, thresholds_rows)
from .promedios import (DEFINITIVE_PREFIX, TABLE_ROWS, chart_polls, headline_text, majority_rows, poll_name,
                        pollster_options, promedios_context, round_series, table_rows)
