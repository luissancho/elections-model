// Past elections page, drawn from the data embedded by the server. The list (error history) has the filters of
// election and house, whose state lives in the URL hash (`errors.js`), the histogram of the poll errors, the
// bullseye coloured by election, the table by weeks before the election and the sector error over time. The detail
// of one election has the error of each house, the histogram of its polls and the bullseye of their last poll.
import {fmtNum} from '../format.js';
import {DEFAULT_MEASURE, MEASURES, errorsHash, filterRows, meanOf, parseErrorsHash, weeksTable} from '../errors.js';
import {renderBars} from '../charts/bars.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderHistogram} from '../charts/histogram.js';
import {renderLine} from '../charts/line.js';
import {mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

// Height of the error bars: one row per house plus the axes.
const BAR_ROW_HEIGHT = 22;
const BAR_EXTRA_HEIGHT = 60;

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
 */
function fillWeeks(table, rows, measure) {
  const {weeks, rows: lines} = weeksTable(rows, measure);
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
 * Error history of the list: read the filters from the hash, draw the histogram, the bullseye and the weeks table
 * of the filtered rows and redraw them (rewriting the hash) on every change of a filter or of the measure.
 *
 * @param {object} initial the page's initial data (`state`, `errors`, `error_blocks`, `events`)
 */
function wireHistory(initial) {
  const {state: page, errors = [], error_blocks: blocks = [], events = []} = initial;
  const form = document.getElementById('errors-filter');
  if (!form) {
    return;  // no analysis published
  }
  const pollsters = [...new Set(errors.map((row) => row.pollster))];
  const state = parseErrorsHash(window.location.hash, {events: events.map((event) => event.date), pollsters});
  const count = document.getElementById('errors-count');
  const histogramEl = document.getElementById('errors-histogram');
  const bullseyeEl = document.getElementById('errors-bullseye');
  const weeksEl = document.getElementById('weeks-table');

  // One colour per election, by its position in the list (newest first); the bullseye groups by its label.
  const colorOfDate = new Map(events.map((event, i) => [event.date, EVENT_COLORS[i % EVENT_COLORS.length]]));
  const colors = {};
  for (const row of errors) {
    colors[row.event] = colorOfDate.get(row.event_date);
  }
  const scope = encodeURIComponent(page.scope);
  const linked = errors.map((row) => ({
    ...row, href: `/elecciones?scope=${scope}&date=${encodeURIComponent(row.event_date)}`,
  }));

  /** Write the filters to the URL hash without adding a history entry. */
  function writeHash() {
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + errorsHash(state));
  }

  /** Redraw everything that depends on the filters. */
  function apply() {
    const rows = filterRows(linked, state);
    if (count) {
      count.textContent = `${rows.length} ${rows.length === 1 ? 'sondeo' : 'sondeos'} de ${errors.length}`;
    }
    if (histogramEl) {
      drawErrorHistogram(histogramEl, rows, state.measure);
    }
    if (bullseyeEl && blocks.length >= 2) {
      tryDraw(() => renderBullseye(bullseyeEl, rows, {
        blocks, groupOf: (row) => row.event, colors, legend: true, labels: false,
      }), 'the errors bullseye');
    }
    if (weeksEl) {
      fillWeeks(weeksEl, rows, state.measure);
    }
  }

  form.elements.eleccion.value = state.event ?? '';
  form.elements.casa.value = state.pollster ?? '';
  form.hidden = false;
  form.addEventListener('submit', (event) => event.preventDefault());
  form.addEventListener('change', () => {
    state.event = form.elements.eleccion.value || null;
    state.pollster = form.elements.casa.value || null;
    apply();
    writeHash();
  });
  wireMeasure(document.getElementById('errors-measure'), state.measure, (measure) => {
    state.measure = measure;
    apply();
    writeHash();
  });
  apply();
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
  const {meta = {}, sector = [], pollsters = [], bullseye = [], blocks = []} = initial;

  const sectorEl = document.getElementById('sector-chart');
  const points = sector.filter((row) => row.error_avg !== null && row.error_avg !== undefined)
    .map((row) => [row.date, row.error_avg]);
  if (sectorEl && points.length) {
    tryDraw(() => renderLine(sectorEl, points, {formatter: (x) => fmtNum(x, 1), name: 'Error medio del sector'}),
      'the sector error');
  }

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
