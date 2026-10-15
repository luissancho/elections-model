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
from .casas import (ANALYSIS_CSV, HERDING_KEYS, MTYPE_LABELS, RANKING_KEYS, RANKING_SHOWN, analysis_part, casa_context,
                    casas_context, effects_table, herding_rows, house_effects_grid, last_poll_records, poll_date,
                    profile_rows, ranking_rows, rating_path)
from .common import (EVENT_TABS, MESSAGES, NOT_FOUND, OTHERS_COLOR, PAGES, REPO_ROOT, ROUTES, SITE_TAGLINE,
                     SITE_TITLE, TABBAR, TEMPLATES_DIR, Catalog, api_url, block_color, catalog, chamber_text, common_context,
                     days_left, days_text, default_mode, environment, event_info, event_query, event_title, hemicycle_order,
                     ink_color, parse_state, part, published_scopes, resolve_event_scope, resolve_scope, scope_query, tabbar,
                     title_of)
from .elecciones import (EVENT_CSV, HISTORY_COLUMNS, LAST_KEYS, LEFT_BLOCK, RESULTS_SHOWN, SUMMARY_KEYS, chart_parties,
                         error_blocks, event_block_rows, history_events, history_rows, history_scopes, eleccion_context,
                         elecciones_context, error_rows, event_part, last_poll_rows, last_results, horizon_tables, majority_of,
                         mark_shown, party_blocks, poll_error_parts, published_events, resolve_event, shown_parties,
                         split_sides, with_abs)
from .escanos import (CSV_PARTS, block_rows, csv_links, default_coalition, district_table, has_communities, escanos_context,
                      region_rows, resolve_region, seat_rows)
from .filters import (DASH, MONTHS, bar_sides, bar_width, delta_text, fmt_date, fmt_date_short, fmt_datetime, fmt_int,
                      fmt_num, fmt_pct, fmt_prob, fmt_range, rng_cell, shade)
from .inicio import INTRO, card_headline, event_cards, inicio_context, past_events
from .metodo import (METODO_SECTIONS, NATIONAL_SCOPE, SECTIONS, method_href, metodo_context, metodo_section_context,
                     national_figures, subpages, thresholds_rows)
from .slugs import (NATIONAL_SLUG, event_href, event_segment, pollster_href, pollster_slugs, resolve_event_path,
                    resolve_pollster, scope_paths, scope_slug, slugify)
from .promedios import (BLOCK_ORDER, PAGE_SIZE, block_result_rows, chart_polls, poll_months, poll_name, polls_table,
                        pollster_options, promedios_context, result_rows, round_series, short_date)
