// Wiring of the evolution of the updates of the model: the party and block views of the daily runs
// (`evolutionViews`), the pill legend that drives the hidden ECharts legend, the % Voto/Escaños and Partidos/Bloques
// selectors, the table of changes under the chart and the download button (`wireEvolution`). The selection lives in
// the chart alone, never in the URL hash, like the house chart of the event page.
import {blockRuns} from '../blocks.js';
import {dailyRuns, evolutionImageOption, renderEvolution} from '../charts/evolution.js';
import {chartZoom} from '../charts/series.js';
import {changesTableHtml, evolutionChanges} from '../evolution-changes.js';
import {fmtPct} from '../format.js';
import {tableImage, wireDownload} from '../share.js';
import {mapNames, markSegments, showError, tryDraw} from './common.js';
import {markLegend} from './legend-dom.js';
import {chosenNames, fillLegend} from './series-wire.js';

/** Whether a value of a headline is a number (the parts write missing values as null). */
const present = (value) => value !== null && value !== undefined;

/** Metrics of the evolution: the vote, the default one, and the seats of the central allocation. */
const METRIC_TEXT = {pct: 'el voto', seats: 'los escaños'};

/**
 * The metric drawn: what the views and the chart read of each run.
 *
 * @param {object} metrics `{key: {field, formatter, total}}`, the default metric first
 * @param {string} key the metric chosen (`pct` or `seats`)
 * @returns {{field: string, formatter: function, total: number}} that of `key`; the first one for an unknown key
 */
export function evolutionMetric(metrics, key) {
  const {field, formatter, total} = metrics[key] || Object.values(metrics)[0];
  return {field, formatter, total};
}

/**
 * Name of the file of the image of the evolution for a metric: the template writes that of the vote
 * (`…-evolucion-…`); the seats add their word (`…-evolucion-escanos-…`).
 *
 * @param {string} file the file of the button (`data-file`)
 * @param {string} key the metric shown
 * @returns {string} the file of that metric
 */
export function evolutionFile(file, key) {
  return key === 'seats' && !file.includes('-evolucion-escanos-')
    ? file.replace('-evolucion-', '-evolucion-escanos-') : file;
}

/**
 * Accessible name of the chart of the evolution for a metric.
 *
 * @param {string} key the metric shown (`pct` or `seats`)
 * @returns {string} what the chart draws
 */
export function evolutionLabel(key) {
  const what = METRIC_TEXT[key] || METRIC_TEXT.pct;
  return `Evolución de las actualizaciones: ${what} de cada partido en cada actualización`;
}

/**
 * Data of the two views of the evolution: the parties of the page, in its order (`parties`: the same names and
 * order as the legend of the average, Luis 2026-10-30; without it, the parties of the last daily run with a value
 * of `field`, in the order of the run), the ones of `bmaps.main` shown by default, and, when the scope has
 * blocks, the blocks summed in the browser (`blockRuns`), all shown by default.
 *
 * @param {object[]} runs `runs` list of the history part
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} [options] `parties` (party names, legend order), `field` (party field drawn: `pct` or
 *   `seats`) and `total` (what the blocks add up to: 100 for shares, the chamber for seats)
 * @returns {object} `partidos` and `bloques` (null without blocks): `runs` (daily), `parties`, `colors`,
 *   `fullnames` and `defaults` (names shown by default, null for all), as `fillLegend` and `chosenNames` read them
 */
export function evolutionViews(runs, catalog, {parties = null, field = 'pct', total = 100} = {}) {
  const daily = dailyRuns(runs);
  const last = daily[daily.length - 1];
  const rows = last ? (last.nowcast || {}).parties || [] : [];
  const names = parties ? [...parties] : rows.filter((row) => present(row[field])).map((row) => row.name);
  const main = catalog.bmaps.main || [];
  const defaults = names.filter((name) => main.includes(name));
  const blocks = catalog.bmaps.blocks && !Array.isArray(catalog.bmaps.blocks) ? catalog.bmaps.blocks : {};
  const blockNames = catalog.orderBlocks(Object.keys(blocks));
  return {
    partidos: {
      runs: daily,
      parties: names,
      colors: mapNames(names, (name) => catalog.color(name)),
      fullnames: mapNames(names, (name) => catalog.fullname(name)),
      defaults: defaults.length ? defaults : null,
    },
    bloques: blockNames.length ? {
      runs: blockRuns(daily, blocks, field, total),
      parties: blockNames,
      colors: mapNames(blockNames, (name) => catalog.blockColor(name)),
      fullnames: mapNames(blockNames, (name) => name),
      defaults: null,
    } : null,
  };
}

/** The metrics of a page that does not give them: the vote alone. */
const VOTE_ONLY = {pct: {field: 'pct', formatter: fmtPct, total: 100}};

/**
 * Draw the evolution (`#evolution-chart`; nothing without it) and wire its controls: the % Voto/Escaños radios
 * (`#evolution-metric`, shown only with more than one metric), the Partidos/Bloques radios (`#evolution-view`, shown
 * only when the scope has blocks), the pill legend (`#evolution-legend`), which selects the names through the hidden
 * ECharts legend, the table of changes (`#evolution-changes`), with every name of the view whatever the legend shows,
 * shown only on demand by its button (`#evolution-changes-toggle`, «Ver los datos» / «Ocultar los datos», hidden with
 * fewer than two days; Luis, 2026-11-04), and the download button (`#evolution-download`), which saves the chart as
 * shown, with the table under it, as a signed image (`share.js`), its file named after the metric
 * (`evolutionFile`). Changing the metric or the view resets the selection.
 *
 * @param {object} initial the page's initial data (`runs`, the history part, and `inks`, the text colour over
 *   the colour of each name for the pills)
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} [options] `parties` (party names, legend order: the list of the page) and `metrics`
 *   (`{key: {field, formatter, total}}`: `field` is the party field drawn, `formatter` value → text and `total` what
 *   the blocks add up to; the first one is the default; the vote alone by default)
 */
export function wireEvolution(initial, catalog, {parties = null, metrics = VOTE_ONLY} = {}) {
  const chartEl = document.getElementById('evolution-chart');
  if (!chartEl) {
    return;
  }
  const legendEl = document.getElementById('evolution-legend');
  const changesEl = document.getElementById('evolution-changes');
  const toggleEl = document.getElementById('evolution-changes-toggle');
  const viewSet = document.getElementById('evolution-view');
  const metricSet = document.getElementById('evolution-metric');
  const runs = (initial.runs || {}).runs || [];
  const keys = Object.keys(metrics);
  const cache = {};
  /** The views of a metric, built once. */
  const viewsOf = (key) => {
    cache[key] = cache[key] || evolutionViews(runs, catalog, {parties, ...evolutionMetric(metrics, key)});
    return cache[key];
  };
  const state = {metric: keys[0], view: 'partidos', selected: null, listOpen: false};
  let chart = null;

  /** Data of the chart (`evolutionOption`) in the metric, view and selection of `chartState`. */
  function dataOf(chartState) {
    const current = viewsOf(chartState.metric)[chartState.view];
    const {field, formatter} = evolutionMetric(metrics, chartState.metric);
    return {
      runs: current.runs,
      parties: current.parties,
      selected: chartState.selected ?? current.defaults,
      colors: current.colors,
      fullnames: current.fullnames,
      field,
      formatter,
      legend: 'hidden',
    };
  }

  /** Fill the table of changes of the current view and show it if open; button and table hidden without changes. */
  function fillChanges() {
    if (!changesEl) {
      return;
    }
    const current = viewsOf(state.metric)[state.view];
    const {field, formatter} = evolutionMetric(metrics, state.metric);
    const changes = evolutionChanges(current.runs, current.parties, field);
    changesEl.innerHTML = changes ? changesTableHtml(changes, current, {field, formatter}) : '';
    changesEl.hidden = !changes || !state.listOpen;
    if (toggleEl) {
      toggleEl.hidden = !changes;
      toggleEl.textContent = state.listOpen ? 'Ocultar los datos' : 'Ver los datos';
      toggleEl.setAttribute('aria-expanded', String(state.listOpen));
    }
  }

  /** Draw the chart, fill its pill legend and its table and, the first time, listen to the legend selection. */
  function draw() {
    const drawn = renderEvolution(chartEl, dataOf(state));
    if (!chart) {
      chart = drawn;
      chart.on('legendselectchanged', (event) => {
        state.selected = chosenNames(viewsOf(state.metric)[state.view], event.selected);
        if (legendEl) {
          markLegend(legendEl, event.selected);
        }
      });
    }
    fillLegend(legendEl, viewsOf(state.metric)[state.view], state.selected, chart, initial.inks || {});
    fillChanges();
  }

  for (const input of document.querySelectorAll('#evolution-view input[name=evview]')) {
    input.checked = input.value === state.view;
  }
  for (const input of document.querySelectorAll('#evolution-metric input[name=evmetric]')) {
    input.checked = input.value === state.metric;
  }
  markSegments();
  tryDraw(draw, 'the evolution');
  wireDownload(document.getElementById('evolution-download'), () => evolutionImageOption(dataOf(state), chartZoom(chart)), showError, {
    bottom: changesEl ? () => (changesEl.hidden ? null : tableImage(changesEl.querySelector('table'))) : null,
    texts: keys.length > 1 ? ({file}) => ({file: evolutionFile(file, state.metric)}) : null,
  });

  if (toggleEl && changesEl) {
    toggleEl.addEventListener('click', () => {
      state.listOpen = !state.listOpen;
      fillChanges();
    });
  }

  // The selectors only work with JavaScript, so the server renders them hidden.
  if (metricSet && keys.length > 1) {
    metricSet.hidden = false;
    metricSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=evmetric]') && metrics[event.target.value]) {
        state.metric = event.target.value;
        state.selected = null;
        chartEl.setAttribute('aria-label', evolutionLabel(state.metric));
        tryDraw(draw, 'the evolution');
      }
    });
  }
  if (viewSet && viewsOf(state.metric).bloques) {
    viewSet.hidden = false;
    viewSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=evview]') && viewsOf(state.metric)[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        tryDraw(draw, 'the evolution');
      }
    });
  }
}
