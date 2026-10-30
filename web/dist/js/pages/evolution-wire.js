// Wiring of the evolution of the publications, shared by Promedios and Escaños: the party and block views of the
// daily runs (`evolutionViews`), the pill legend that drives the hidden ECharts legend, the Partidos/Bloques
// selector and the download button (`wireEvolution`). The selection lives in the chart alone, never in the URL
// hash, like the house chart of Promedios.
import {blockRuns} from '../blocks.js';
import {dailyRuns, evolutionImageOption, renderEvolution} from '../charts/evolution.js';
import {chartZoom} from '../charts/series.js';
import {fmtPct} from '../format.js';
import {wireDownload} from '../share.js';
import {mapNames, markSegments, showError, tryDraw} from './common.js';
import {markLegend} from './legend-dom.js';
import {chosenNames, fillLegend} from './series-wire.js';

/** Whether a value of a headline is a number (the parts write missing values as null). */
const present = (value) => value !== null && value !== undefined;

/**
 * Data of the two views of the evolution: the parties of the last daily run (those with a value of `field`, in
 * catalogue order, the ones of `bmaps.main` shown by default) and, when the scope has blocks, the blocks summed
 * in the browser (`blockRuns`), all shown by default.
 *
 * @param {object[]} runs `runs` list of the history part
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} [options] `field` (party field drawn: `pct` or `seats`) and `total` (what the blocks add up
 *   to: 100 for shares, the chamber for seats)
 * @returns {object} `partidos` and `bloques` (null without blocks): `runs` (daily), `parties`, `colors`,
 *   `fullnames` and `defaults` (names shown by default, null for all), as `fillLegend` and `chosenNames` read them
 */
export function evolutionViews(runs, catalog, {field = 'pct', total = 100} = {}) {
  const daily = dailyRuns(runs);
  const last = daily[daily.length - 1];
  const rows = last ? (last.nowcast || {}).parties || [] : [];
  const names = catalog.order(rows.filter((row) => present(row[field])).map((row) => row.name));
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

/**
 * Draw the evolution (`#evolution-chart`; nothing without it) and wire its controls: the Partidos/Bloques radios
 * (`#evolution-view`, shown only when the scope has blocks), the pill legend (`#evolution-legend`), which selects
 * the names through the hidden ECharts legend, and the download button (`#evolution-download`), which saves the
 * chart as shown as a signed image (`share.js`). Changing the view resets the selection.
 *
 * @param {object} initial the page's initial data (`runs`, the history part, and `inks`, the text colour over
 *   the colour of each name for the pills)
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} [options] `field` (party field drawn), `formatter` (value → text) and `total` (what the blocks
 *   add up to)
 */
export function wireEvolution(initial, catalog, {field = 'pct', formatter = fmtPct, total = 100} = {}) {
  const chartEl = document.getElementById('evolution-chart');
  if (!chartEl) {
    return;
  }
  const legendEl = document.getElementById('evolution-legend');
  const viewSet = document.getElementById('evolution-view');
  const views = evolutionViews((initial.runs || {}).runs || [], catalog, {field, total});
  const state = {view: 'partidos', selected: null};
  let chart = null;

  /** Data of the chart (`evolutionOption`) in the view and selection of `chartState`. */
  function dataOf(chartState) {
    const current = views[chartState.view];
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

  /** Draw the chart, fill its pill legend and, the first time, listen to the legend selection. */
  function draw() {
    const drawn = renderEvolution(chartEl, dataOf(state));
    if (!chart) {
      chart = drawn;
      chart.on('legendselectchanged', (event) => {
        state.selected = chosenNames(views[state.view], event.selected);
        if (legendEl) {
          markLegend(legendEl, event.selected);
        }
      });
    }
    fillLegend(legendEl, views[state.view], state.selected, chart, initial.inks || {});
  }

  for (const input of document.querySelectorAll('#evolution-view input[name=evview]')) {
    input.checked = input.value === state.view;
  }
  markSegments();
  tryDraw(draw, 'the evolution');
  wireDownload(document.getElementById('evolution-download'), () => evolutionImageOption(dataOf(state), chartZoom(chart)), showError);

  // The selector only works with JavaScript, so the server renders it hidden.
  if (viewSet && views.bloques) {
    viewSet.hidden = false;
    viewSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=evview]') && views[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        tryDraw(draw, 'the evolution');
      }
    });
  }
}
