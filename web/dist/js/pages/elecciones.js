// Past elections page, drawn from the data embedded by the server. The list (error history) has the chained
// filters of scope, election, house and years, whose state lives in the URL hash (`errors.js`), the figures of the
// selection, the evolution of the error by election, the histogram of the poll errors, the bullseye coloured by
// election or scope, the table by weeks before the election and the table of elections. The detail of one
// election has the error of each house, the histogram of its polls and the bullseye of their last poll.
import {fmtDate, fmtNum} from '../format.js';
import {
  DEFAULT_MEASURE, MEASURES, complements, errorsHash, eventKey, eventsOf, filterRows, meanOf, parseErrorsHash,
  restPoints, trendPoints, weeksTable,
} from '../errors.js';
import {renderBars} from '../charts/bars.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderHistogram} from '../charts/histogram.js';
import {renderTrend} from '../charts/trend.js';
import {fillComplements, fillFigures, fillSelect, filterEventsTable} from './history-dom.js';
import {markSegments, mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

// Height of the error bars: one row per house plus the axes.
const BAR_ROW_HEIGHT = 22;
const BAR_EXTRA_HEIGHT = 60;

// Accent of the national scope (`es`) in the trend and in the bullseye by scope.
const ACCENT = '#1d4f86';
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
  const lines = perElection ? all : all.slice(-1);
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
 * @param {object} context `scopes`, `events` and `colors` of the page
 */
function drawTrend(el, rows, others, state, {scopes, events, colors}) {
  const names = new Map(events.map((event) => [event.key, event.name]));
  const byScope = new Map(scopes.map(({name}) => [name, []]));
  for (const item of eventsOf(rows, state.measure).filter((entry) => entry.mean !== null)) {
    const key = eventKey(item);
    byScope.get(item.scope_name)?.push({
      date: item.event_date,
      value: item.mean,
      n: item.n,
      label: [fmtDate(item.event_date), names.get(key) || item.event, item.scope_name].join(' · '),
      href: `/elecciones?scope=${encodeURIComponent(item.scope)}&date=${encodeURIComponent(item.event_date)}`,
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
 * Bullseye of the filtered rows: coloured by election with a scope chosen and by scope with "Todos".
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows filtered rows with their `href`
 * @param {string[]} blocks the two blocks of the axes
 * @param {(string|null)} scope the chosen scope
 * @param {object} context `events` (published list) and `colors` (by scope name) of the page
 */
function drawErrorBullseye(el, rows, blocks, scope, {events, colors}) {
  let groupOf = (row) => row.scope_name;
  let palette = colors;
  let groupLabel = (group) => group;
  if (scope !== null) {
    groupOf = eventKey;
    palette = {};
    events.filter((event) => event.scope === scope).forEach((event, i) => {
      palette[event.key] = EVENT_COLORS[i % EVENT_COLORS.length];
    });
    const labels = new Map(rows.map((row) => [eventKey(row), row.event]));
    groupLabel = (key) => labels.get(key) ?? key;
  }
  tryDraw(() => renderBullseye(el, rows, {blocks, groupOf, colors: palette, legend: true, labels: false, groupLabel}),
    'the errors bullseye');
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
  });
  if (state.scope === null && !/(^|[#&])ambito=/.test(window.location.hash)
      && scopes.some((scope) => scope.code === initial.selected_scope)) {
    state.scope = initial.selected_scope;
  }
  const count = document.getElementById('errors-count');
  const histogramEl = document.getElementById('errors-histogram');
  const bullseyeEl = document.getElementById('errors-bullseye');
  const trendEl = document.getElementById('trend-chart');
  const weeksEl = document.getElementById('weeks-table');
  const weeksNote = document.getElementById('weeks-note');
  const eventsTable = document.getElementById('events-table');
  const figures = document.getElementById('figures-list');
  const complementsEl = document.getElementById('complements');
  const colors = scopeColors(scopes);
  const analysed = new Set(scopes.map((scope) => scope.code));
  const linked = errors.map((row) => ({
    ...row,
    href: `/elecciones?scope=${encodeURIComponent(row.scope)}&date=${encodeURIComponent(row.event_date)}`,
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
      label: [fmtDate(event.date), event.name, event.scope_name].filter(Boolean).join(' · '),
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
      drawTrend(trendEl, rows, others, state, {scopes, events, colors});
    }
    if (histogramEl) {
      drawErrorHistogram(histogramEl, rows, state.measure);
    }
    if (bullseyeEl && blocks.length >= 2) {
      drawErrorBullseye(bullseyeEl, rows, blocks, state.scope, {events, colors});
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
 * Draw the charts whose container is present and has data. A scope without published elections renders no
 * `#initial-data` island (and no charts), so there is nothing to draw and `readInitial` must not be called.
 */
function paint() {
  if (!document.getElementById('initial-data')) {
    return;
  }

  const initial = readInitial();
  const {meta = {}, pollsters = [], bullseye = [], blocks = []} = initial;

  if (initial.errors) {
    wireHistory(initial);
  }

  const bars = document.getElementById('error-chart');
  const rows = pollsters.filter((row) => row.error_blocks_abs !== null && row.error_blocks_abs !== undefined)
    .map((row) => ({name: row.pollster, error_blocks_abs: row.error_blocks_abs}));
  if (bars && rows.length) {
    bars.style.height = `${rows.length * BAR_ROW_HEIGHT + BAR_EXTRA_HEIGHT}px`;
    tryDraw(() => renderBars(bars, rows, {value: 'error_blocks_abs', formatter: (x) => fmtNum(x, 1)}),
      'the house errors');
  }

  if (initial.poll_errors) {
    wirePollHistogram(initial.poll_errors);
  }

  const target = document.getElementById('bullseye-chart');
  if (target && bullseye.length && blocks.length >= 2) {
    tryDraw(() => renderBullseye(
      target,
      bullseye.map((row) => ({...row, event: row.pollster, event_date: meta.event_date})),
      {blocks},
    ), 'the bullseye');
  }
}

wireControls();
mountWhenReady(paint);
