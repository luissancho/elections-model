// Past elections page, drawn from the data embedded by the server. The list (error history) has the chained
// filters of scope, election, house and years, whose state lives in the URL hash (`errors.js`), the figures of the
// selection, the evolution of the error by election, the histogram of the poll errors, the bullseye coloured by
// election or scope, the table by weeks before the election and the table of elections. The detail of one
// election has the hemicycle of the official result next to the results table (behind "Ver más"), the error of
// the last poll of each house (bars, the split by block and the bullseye), the histogram of its polls, the poll
// average of the cycle with the official result (`series-wire.js`, shared with Promedios), the horizon selector of
// what the model said and the table of every poll of the cycle (`polls-wire.js`).
import {parseSeriesHash} from '../blocks.js';
import {Catalog} from '../catalog.js';
import {fmtDate, fmtNum} from '../format.js';
import {
  DEFAULT_MEASURE, MEASURES, complements, errorsHash, eventKey, eventsOf, filterRows, imageTexts, meanOf,
  parseErrorsHash, restPoints, trendPoints, weeksTable, yearsText,
} from '../errors.js';
import {THEME} from '../charts/base.js';
import {renderBars} from '../charts/bars.js';
import {bullseyeOption, renderBullseye} from '../charts/bullseye.js';
import {renderDiverging} from '../charts/diverging.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {renderHistogram} from '../charts/histogram.js';
import {renderSeries, seriesImageOption} from '../charts/series.js';
import {renderTrend} from '../charts/trend.js';
import {fillComplements, fillFigures, fillSelect, filterEventsTable} from './history-dom.js';
import {BULLSEYE_IMAGE_HEIGHT, chosenLabel} from './bullseye-wire.js';
import {mapNames, markSegments, mountWhenReady, readInitial, showError, tryDraw, wireControls, wireMore} from './common.js';
import {wireDownload} from '../share.js';
import {wirePollsTable} from './polls-wire.js';
import {WINDOWS, seriesViews, wireAverage} from './series-wire.js';

// Height of the error bars: one row per house plus the axes (and the legend, in the split by block).
const BAR_ROW_HEIGHT = 22;
const BAR_EXTRA_HEIGHT = 60;
const SPLIT_EXTRA_HEIGHT = 80;

// Accent of the national scope (`es`) in the trend and in the bullseye by scope, and of the detail's bullseye.
const ACCENT = THEME.accentColor;
// The model among the pollsters in the detail's bullseye: the orange of the site (that of the regional elections)
const MODEL_COLOR = '#c0672e';
const NATIONAL_SCOPE = 'es';

// The weeks table lists the elections one by one only up to this many.
const WEEKS_MAX_EVENTS = 15;

// Colours of the elections in the bullseye of the error history, by position in the published list (newest
// first): sixteen fixed neutral tones, unrelated to the party colours, distinguishable from each other and
// readable on white, so every published election of a scope (`es` has 11) gets its own; only a scope with more
// than sixteen would cycle.
const EVENT_COLORS = [
  '#3b6e8f', '#c0672e', '#5a8a4f', '#8a5a8c', '#9a8a1f', '#a8505a', '#2f8f8f', '#6b6b6b',
  '#7a4e2d', '#5468b0', '#b0487f', '#4f7a2a', '#d08a1c', '#2c5d5d', '#8c7aa8', '#1f3f5c',
];

/**
 * Draw the histogram of a measure of some poll-errors rows: polls counted (unweighted) in one-point bins, with
 * the mean marked.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows rows with the measure
 * @param {string} measure `error_blocks_abs` or `error_avg`
 */
function drawErrorHistogram(el, rows, measure) {
  const values = rows.map((row) => row[measure]).filter(Number.isFinite);
  const mean = meanOf(rows, measure);
  tryDraw(() => renderHistogram(el, values, {
    width: 1,
    unit: 'puntos',
    label: (start) => `${start} a ${start + 1} puntos`,
    median: mean,
    markLabel: mean === null ? null : `Media: ${fmtNum(mean, 1)}`,
    share: false,
    countLabel: (count) => `${count} ${count === 1 ? 'sondeo' : 'sondeos'}`,
  }), 'the error histogram');
}

/**
 * Wire the measure radios of a histogram: show them (they only work with JavaScript, so the server renders them
 * hidden), check the current measure and call `onChange` with the new one.
 *
 * @param {HTMLElement|null} fieldset the `medida` radios
 * @param {string} measure the current measure
 * @param {Function} onChange receives the chosen measure
 */
function wireMeasure(fieldset, measure, onChange) {
  if (!fieldset) {
    return;
  }
  for (const input of fieldset.querySelectorAll('input[name=medida]')) {
    input.checked = input.value === measure;
  }
  markSegments(fieldset);
  fieldset.hidden = false;
  fieldset.addEventListener('change', (event) => {
    if (event.target.matches('input[name=medida]') && MEASURES.includes(event.target.value)) {
      onChange(event.target.value);
    }
  });
}

/**
 * Fill the table by weeks before the election (`weeksTable`): a column per week, a row per election and the row
 * of every poll; each cell is the mean of the measure and, in brackets, its polls.
 *
 * @param {HTMLTableElement} table the `#weeks-table`, with its caption, `thead` and `tbody`
 * @param {object[]} rows filtered poll-errors rows
 * @param {string} measure `error_blocks_abs` or `error_avg`
 * @param {boolean} perElection whether to keep the row of each election (otherwise only the one of every poll)
 */
function fillWeeks(table, rows, measure, perElection) {
  const {weeks, rows: all} = weeksTable(rows, measure);
  const lines = perElection ? all : all.slice(-1);  // empty without valid rows
  const cell = (tag, text, scope = null) => {
    const node = document.createElement(tag);
    node.textContent = text;
    if (scope) {
      node.scope = scope;
    }
    return node;
  };
  const head = document.createElement('tr');
  head.append(cell('th', 'Elección', 'col'), ...weeks.map((week) => cell('th', `Semana ${week}`, 'col')));
  table.tHead.replaceChildren(head);
  table.tBodies[0].replaceChildren(...lines.map((line) => {
    const tr = document.createElement('tr');
    tr.append(cell('th', line.event, 'row'), ...weeks.map((week) => {
      const value = line.cells[week];
      return cell('td', value ? `${fmtNum(value.mean, 1)} (${value.n})` : '');
    }));
    return tr;
  }));
}

/**
 * Colour of each scope name: the accent for the national scope and the neutral tones, by order of appearance,
 * for the rest.
 *
 * @param {object[]} scopes `{code, name}` of the history scopes
 * @returns {object} `{scope name: colour}`
 */
function scopeColors(scopes) {
  const tones = EVENT_COLORS.slice(1);  // the first tone is too close to the accent
  const colors = {};
  let next = 0;
  for (const {code, name} of scopes) {
    colors[name] = code === NATIONAL_SCOPE ? ACCENT : tones[next++ % tones.length];
  }
  return colors;
}

/**
 * Evolution of the error: one point per election of the rows (a series per scope), the moving average and, with
 * a house chosen, the mean of the other houses in the same elections.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows filtered rows
 * @param {object[]} others rows that pass every filter but the house
 * @param {object} state the filters and the measure
 * @param {object} context `scopes`, `colors` and `hrefs` (link of each election by key) of the page
 */
function drawTrend(el, rows, others, state, {scopes, colors, hrefs}) {
  const byScope = new Map(scopes.map(({name}) => [name, []]));
  for (const item of eventsOf(rows, state.measure).filter((entry) => entry.mean !== null)) {
    byScope.get(item.scope_name)?.push({
      date: item.event_date,
      value: item.mean,
      n: item.n,
      label: `${fmtDate(item.event_date)} · ${item.name || item.event}`,
      href: hrefs.get(eventKey(item)) ?? null,
    });
  }
  const series = [...byScope].filter(([, points]) => points.length).map(([name, points]) => ({name, points}));
  tryDraw(() => renderTrend(el, {
    series,
    trend: trendPoints(rows, state.measure),
    rest: state.pollster ? restPoints(others, state.pollster, state.measure) : null,
  }, {formatter: (x) => fmtNum(x, 1), colors, unit: 'puntos', accent: ACCENT}), 'the error trend');
}

/**
 * Bullseye of the filtered rows: coloured by election with a scope chosen and by scope with "Todos" (the
 * national scope, the general elections, drawn over the others).
 *
 * @param {object[]} rows filtered rows with their `href`
 * @param {string[]} blocks the two blocks of the axes
 * @param {(string|null)} scope the chosen scope
 * @param {object} context `events` (published list), `colors` (by scope name) and `national` (name of the
 *   national scope) of the page
 * @returns {object} the options of `bullseyeOption`
 */
function errorBullseyeOptions(rows, blocks, scope, {events, colors, national}) {
  let groupOf = (row) => row.scope_name;
  let palette = colors;
  let groupLabel = (group) => group;
  let onTop = (group) => group === national;
  if (scope !== null) {
    groupOf = eventKey;
    palette = {};
    onTop = null;
    events.filter((event) => event.scope === scope).forEach((event, i) => {
      palette[event.key] = EVENT_COLORS[i % EVENT_COLORS.length];
    });
    const labels = new Map(rows.map((row) => [eventKey(row), row.event]));
    groupLabel = (key) => labels.get(key) ?? key;
  }
  return {blocks, groupOf, colors: palette, legend: true, labels: false, groupLabel, onTop};
}

/**
 * Error history of the list: read the filters from the hash, chain the selects, draw the figures, the trend, the
 * histogram, the bullseye and the weeks table of the filtered rows, filter the table of elections and redraw
 * (rewriting the hash) on every change of a filter or of the measure.
 *
 * @param {object} initial the page's initial data (`errors`, `error_blocks`, `events`, `scopes`, `years`,
 *   `selected_scope`)
 */
function wireHistory(initial) {
  const {errors = [], error_blocks: blocks = [], events = [], scopes: allScopes = [], years = []} = initial;
  const form = document.getElementById('errors-filter');
  if (!form) {
    return;  // no analysis published
  }
  const scopes = allScopes.filter((scope) => scope.analysis);
  const pollsters = [...new Set(errors.map((row) => row.pollster))];
  const state = parseErrorsHash(window.location.hash, {
    scopes: scopes.map((scope) => scope.code), events: events.map((event) => event.key), pollsters, years,
    defaultScope: initial.selected_scope ?? 'es',
  });
  if (state.scope === null && !/(^|[#&])ambito=/.test(window.location.hash)
      && scopes.some((scope) => scope.code === initial.selected_scope)) {
    state.scope = initial.selected_scope;
  }
  const count = document.getElementById('errors-count');
  const histogramEl = document.getElementById('errors-histogram');
  const bullseyeEl = document.getElementById('errors-bullseye');
  let shown = [];  // the rows drawn, the ones the download button exports
  const trendEl = document.getElementById('trend-chart');
  const weeksEl = document.getElementById('weeks-table');
  const weeksNote = document.getElementById('weeks-note');
  const eventsTable = document.getElementById('events-table');
  const figures = document.getElementById('figures-list');
  const complementsEl = document.getElementById('complements');
  const colors = scopeColors(scopes);
  const national = scopes.find((scope) => scope.code === NATIONAL_SCOPE)?.name ?? null;
  const analysed = new Set(scopes.map((scope) => scope.code));
  // The rows carry the short code of their election (`2023-23J`); the charts and tables name it in full and
  // link its detail (the server gives the link of each election, `null` for a scope without a run).
  const names = new Map(events.map((event) => [event.key, event.name]));
  const hrefs = new Map(events.map((event) => [event.key, event.href ?? null]));
  const linked = errors.map((row) => ({
    ...row,
    name: names.get(eventKey(row)) || null,
    href: hrefs.get(eventKey(row)) ?? null,
  }));
  const figureNodes = figures && {
    polls: document.getElementById('figure-polls'),
    events: document.getElementById('figure-events'),
    mean: document.getElementById('figure-mean'),
    bias: document.getElementById('figure-bias'),
    meanLabel: document.getElementById('figure-mean-label'),
  };

  /** Whether a year is inside the chosen range (an extreme left at `null` is open; swapped bounds are ordered). */
  const inYears = (year) => {
    const swap = state.from !== null && state.to !== null && state.from > state.to;
    const [low, high] = swap ? [state.to, state.from] : [state.from, state.to];
    return (low === null || year >= low) && (high === null || year <= high);
  };

  /**
   * Rebuild the selects of election and house for the chosen scope and years; a chosen value that is no
   * longer offered becomes `null`.
   */
  function chain() {
    const inRange = filterRows(linked, {scope: state.scope, from: state.from, to: state.to});
    const offered = events.filter((event) => analysed.has(event.scope) && (state.scope === null || event.scope === state.scope)
      && inYears(Number(event.date.slice(0, 4))));
    state.event = fillSelect(form.elements.eleccion, offered.map((event) => ({
      value: event.key,
      label: [fmtDate(event.date), event.name].filter(Boolean).join(' · '),
      scope: event.scope,
    })), 'Todas', state.event);
    const houses = [...new Set(inRange.map((row) => row.pollster))].sort((a, b) => a.localeCompare(b, 'es'));
    state.pollster = fillSelect(form.elements.casa, houses.map((name) => ({value: name, label: name})), 'Todas',
      state.pollster);
  }

  /** Write the filters to the URL hash without adding a history entry. */
  function writeHash() {
    const {pathname, search} = window.location;
    // An explicit empty `ambito=` keeps "Todos" when the page would preselect a scope on its own.
    const scope = state.scope === null && initial.selected_scope ? '' : state.scope;
    window.history.replaceState(null, '', pathname + search + errorsHash({...state, scope}));
  }

  /** Redraw everything that depends on the filters. */
  function apply() {
    const rows = filterRows(linked, state);
    const others = filterRows(linked, {...state, pollster: null});
    const keys = new Set(rows.map(eventKey));
    const noun = (n, one, many) => `${n} ${n === 1 ? one : many}`;
    if (count) {
      count.textContent = `${noun(rows.length, 'sondeo', 'sondeos')} de ${noun(keys.size, 'elección', 'elecciones')}`;
    }
    if (figureNodes) {
      fillFigures(figureNodes, {
        polls: rows.length,
        events: keys.size,
        mean: meanOf(rows, state.measure),
        bias: meanOf(rows, 'error_blocks'),
        meanLabel: state.measure === 'error_avg' ? 'Error medio' : 'Error de bloques',
      });
      fillComplements(complementsEl, complements(linked, state, state.measure));
    }
    if (trendEl) {
      drawTrend(trendEl, rows, others, state, {scopes, colors, hrefs});
    }
    if (histogramEl) {
      drawErrorHistogram(histogramEl, rows, state.measure);
    }
    shown = rows;
    if (bullseyeEl && blocks.length >= 2) {
      tryDraw(() => renderBullseye(bullseyeEl, rows, errorBullseyeOptions(rows, blocks, state.scope, {events, colors, national})),
        'the errors bullseye');
    }
    if (weeksEl) {
      const perElection = keys.size <= WEEKS_MAX_EVENTS;
      fillWeeks(weeksEl, rows, state.measure, perElection);
      if (weeksNote) {
        weeksNote.hidden = perElection;
      }
    }
    if (eventsTable) {
      filterEventsTable(eventsTable, document.getElementById('events-count'), (scope, date, year) => (
        (state.scope === null || scope === state.scope)
        && (state.event === null || state.event === `${scope}:${date}`)
        && inYears(year)));
    }
  }

  /** Read the controls into the state, normalising the extreme years to `null`. */
  function readControls() {
    const {ambito, eleccion, casa, desde, hasta} = form.elements;
    state.scope = ambito.value || null;
    state.event = eleccion.value || null;
    state.pollster = casa.value || null;
    state.from = Number(desde.value) === years[0] ? null : Number(desde.value);
    state.to = Number(hasta.value) === years[years.length - 1] ? null : Number(hasta.value);
  }

  /** Put the state into the controls that the chain does not rebuild. */
  function writeControls() {
    const {ambito, desde, hasta} = form.elements;
    ambito.value = state.scope ?? '';
    desde.value = String(state.from ?? years[0]);
    hasta.value = String(state.to ?? years[years.length - 1]);
  }

  writeControls();
  chain();
  form.hidden = false;
  if (figures) {
    figures.hidden = false;
  }
  form.addEventListener('submit', (event) => event.preventDefault());
  form.addEventListener('change', (event) => {
    if (event.target.matches('input[name=medida]')) {
      return;  // handled by the measure radios
    }
    readControls();
    chain();
    apply();
    writeHash();
  });
  wireMeasure(document.getElementById('errors-measure'), state.measure, (measure) => {
    state.measure = measure;
    apply();
    writeHash();
  });
  apply();
  writeHash();
  // the bullseye as filtered, the filters chosen and the count in the subtitle and the file name
  wireDownload(document.getElementById('errors-download'),
    () => bullseyeOption(shown, errorBullseyeOptions(shown, blocks, state.scope, {events, colors, national})), showError, {
      chartHeight: BULLSEYE_IMAGE_HEIGHT,
      texts: (dataset) => imageTexts(dataset, [
        chosenLabel(form.elements.ambito), chosenLabel(form.elements.eleccion), chosenLabel(form.elements.casa),
        yearsText(state.from, state.to),
      ], count ? count.textContent : null),
    });
}

/**
 * Histogram of the polls of one election, with its measure radios (no hash: the detail has no other filter).
 *
 * @param {object[]} polls `poll_errors` of the initial data
 */
function wirePollHistogram(polls) {
  const el = document.getElementById('poll-histogram-chart');
  if (!el || !polls.length) {
    return;
  }
  let measure = DEFAULT_MEASURE;
  wireMeasure(document.getElementById('poll-measure'), measure, (chosen) => {
    measure = chosen;
    drawErrorHistogram(el, polls, measure);
  });
  drawErrorHistogram(el, polls, measure);
}

/**
 * Hemicycle of the official result of an election with its Partidos/Bloques switch (as in Promedios): the seats
 * of each party, ordered by block and catalogue, or of each block; the switch also shows the matching results
 * table and hides the other one. The absolute majority is written in the centre.
 *
 * @param {object} initial the page's initial data (`results` with `n_seats`, `majority`, `rows` and `blocks` as
 *   `{name, seats}`; `block_colors`)
 * @param {Catalog} catalog party catalogue of the page (colours, full names and order)
 */
function wireHemicycle(initial, catalog) {
  const el = document.getElementById('hemicycle');
  const {results = null, block_colors: blockColors = {}} = initial;
  if (!el || !results || !results.rows || !results.rows.length) {
    return;
  }
  const names = results.rows.map((row) => row.name);
  const blocks = (results.blocks || []).filter((row) => Number.isFinite(row.seats));
  const views = {
    partidos: {
      seats: Object.fromEntries(results.rows.map((row) => [row.name, row.seats])),
      colors: mapNames(names, (name) => catalog.color(name)),
      fullnames: mapNames(names, (name) => catalog.fullname(name)),
      order: (list) => catalog.order(list),
      label: 'Hemiciclo con los escaños de cada partido en el resultado oficial',
      card: 'results',
    },
    bloques: {
      seats: Object.fromEntries(blocks.map((row) => [row.name, row.seats])),
      colors: mapNames(blocks.map((row) => row.name), (name) => blockColors[name] || THEME.barColor),
      fullnames: {},
      order: (list) => catalog.orderBlocks(list),
      label: 'Hemiciclo con los escaños de cada bloque en el resultado oficial',
      card: 'results-blocks',
    },
  };

  /** Draw the hemicycle of `view` and show its results table, hiding the other one. */
  function draw(view) {
    const chosen = views[view];
    el.setAttribute('aria-label', chosen.label);
    tryDraw(() => renderHemicycle(el, chosen.seats, {
      colors: chosen.colors, fullnames: chosen.fullnames, order: chosen.order, nSeats: results.n_seats,
      majority: results.majority,
    }), 'the hemicycle');
    for (const [name, item] of Object.entries(views)) {
      const card = document.getElementById(item.card);
      if (card) {
        card.hidden = name !== view;
      }
    }
  }

  draw('partidos');
  const fieldset = document.getElementById('results-view');
  if (fieldset && blocks.length) {
    fieldset.hidden = false;
    fieldset.addEventListener('change', (event) => {
      if (event.target.matches('input[name=rview]') && views[event.target.value]) {
        draw(event.target.value);
        markSegments(fieldset);
      }
    });
  }
}

/**
 * Draw the poll average of the cycle (the series frozen at the shortest horizon of the backtest, the polls as
 * points and the official result as diamonds at the election) and wire its controls and its download button, as
 * Promedios does; nothing without the series or its card.
 *
 * @param {object} initial the page's initial data (`series`, `polls`, `result`, `inks`, `meta`)
 * @param {Catalog} catalog party catalogue of the page, with the `bmaps` of the curve
 */
function wireAverageCard(initial, catalog) {
  if (!initial.series || !document.getElementById('series-chart')) {
    return;
  }
  const views = seriesViews(initial, catalog);
  const when = initial.result ? initial.result.date : initial.meta.event_date;
  const state = parseSeriesHash(window.location.hash, {
    parties: views.partidos.parties, blocks: views.bloques ? views.bloques.parties : [], windows: WINDOWS,
  });
  const dataOf = (pollsOfChart, chartState) => {
    const current = views[chartState.view];
    const dates = current.series.dates;
    return {
      series: current.series, polls: pollsOfChart, projection: null, result: current.result,
      parties: current.parties, selected: chartState.selected ?? current.defaults, colors: current.colors,
      asOf: dates[dates.length - 1], asOfLabel: `A ${initial.series.horizon} días`, when,
      window: chartState.window, anchor: when, legend: 'hidden', layers: chartState.layers,
    };
  };
  const render = (el, pollsOfChart, chartState) => renderSeries(el, dataOf(pollsOfChart, chartState));
  const imageOf = (pollsOfChart, chartState, zoom) => seriesImageOption(dataOf(pollsOfChart, chartState), zoom);
  wireAverage(views, state, render, {anchor: when, inks: initial.inks || {}, imageOf});
}

/**
 * Show the horizon selector of «Lo que dijo el modelo» and switch the tables of each horizon with it (the server
 * renders them all, every one but the first hidden).
 */
function wireModelHorizons() {
  const fieldset = document.getElementById('model-horizon');
  if (!fieldset) {
    return;
  }
  fieldset.hidden = false;
  fieldset.addEventListener('change', (event) => {
    if (!event.target.matches('input[name=horizon]')) {
      return;
    }
    for (const box of document.querySelectorAll('#model .horizon[data-horizon]')) {
      box.hidden = box.dataset.horizon !== event.target.value;
    }
    markSegments(fieldset);
  });
}

/**
 * Error of the last poll of each house: the bars of the block error and, next to them, the same houses with the
 * error of each block, one growing to the left and the other to the right (`initial.split` says which).
 *
 * @param {object} initial the page's initial data (`last`, `blocks`, `split`, `block_colors`)
 */
function drawLastErrors(initial) {
  const {last = [], blocks = [], split = null, block_colors: colors = {}} = initial;
  const rows = last.filter((row) => Number.isFinite(row.error_blocks_abs));
  if (!rows.length) {
    return;
  }
  const bars = document.getElementById('error-chart');
  if (bars) {
    bars.style.height = `${rows.length * BAR_ROW_HEIGHT + BAR_EXTRA_HEIGHT}px`;
    tryDraw(() => renderBars(bars, rows.map((row) => ({name: row.pollster, error_blocks_abs: row.error_blocks_abs})),
      {value: 'error_blocks_abs', formatter: (x) => fmtNum(x, 1)}), 'the last-poll errors');
  }
  const el = document.getElementById('split-chart');
  if (el && split && blocks.length >= 2) {
    el.style.height = `${rows.length * BAR_ROW_HEIGHT + SPLIT_EXTRA_HEIGHT}px`;
    const side = (block) => ({key: block, name: block, color: colors[block] || THEME.barColor});
    tryDraw(() => renderDiverging(el, rows.map((row) => ({name: row.pollster, [split.left]: row[split.left], [split.right]: row[split.right]})),
      {left: side(split.left), right: side(split.right), formatter: (x) => fmtNum(x, 1)}), 'the block errors');
  }
}

/**
 * Draw the charts whose container is present and has data. A scope without published elections renders no
 * `#initial-data` island (and no charts), so there is nothing to draw and `readInitial` must not be called.
 */
function paint() {
  wireMore('results-more', 'results-table');
  if (!document.getElementById('initial-data')) {
    return;
  }

  const initial = readInitial();
  const {meta = {}, last = [], blocks = [], model_point: modelPoint = null} = initial;
  const catalog = new Catalog(initial.catalog || {});

  if (initial.errors) {
    wireHistory(initial);
  }

  wireHemicycle(initial, catalog);
  tryDraw(() => wireAverageCard(initial, catalog), 'the poll average');
  wireModelHorizons();
  drawLastErrors(initial);

  if (initial.poll_errors) {
    wirePollHistogram(initial.poll_errors);
  }

  const target = document.getElementById('bullseye-chart');
  if (target && last.length && blocks.length >= 2) {
    tryDraw(() => renderBullseye(
      target,
      last.map((row) => ({...row, event: row.pollster, event_date: meta.event_date})),
      // the model at its shortest horizon, scored like a pollster: the highlighted point, in orange
      {blocks, color: ACCENT, highlight: modelPoint
        ? {row: {...modelPoint, event: modelPoint.pollster, event_date: meta.event_date}, color: MODEL_COLOR}
        : null},
    ), 'the bullseye');
  }

  if (initial.polls) {
    // the rows pinned under the mean: what the model said at its shortest horizon and the official result
    const fixed = initial.model_last ? {
      modelLast: initial.model_last,
      result: initial.result,
      seats: Object.fromEntries(((initial.results && initial.results.rows) || []).map((row) => [row.name, row.seats])),
    } : null;
    tryDraw(() => wirePollsTable(initial.polls.polls, catalog, {fixed}), 'the polls table');
  }
}

wireControls();
mountWhenReady(paint);
