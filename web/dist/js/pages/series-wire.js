// Wiring of the poll average shared by Promedios and the detail of a past election: the party and block views
// of the series (`seriesViews`), the pill legend that drives the ECharts legend and the window, view and layers
// controls (`wireAverage`), whose state lives in the URL hash (`blocks.js`).
import {blockPolls, blockProjection, blockResult, blockSeries, seriesHash} from '../blocks.js';
import {chartZoom, setWindow} from '../charts/series.js';
import {WINDOW_DAYS} from '../range.js';
import {wireDownload} from '../share.js';
import {mapNames, markSegments, showError, tryDraw} from './common.js';
import {markLegend, renderLegend} from './legend-dom.js';

/** Choices of the window control, in the order of `WINDOW_DAYS` plus the whole cycle. */
export const WINDOWS = [...Object.keys(WINDOW_DAYS), 'all'];

/**
 * Data of the two views of the average: the parties of the series and, when the scope has blocks, the blocks
 * computed in the browser (`blocks.js`) with the published block projection.
 *
 * @param {object} initial the page's initial data (`series`, `polls`, `projection` and, in the detail of a past
 *   election, `result`, the official result the chart marks, and `block_colors`, the colour of each block as the
 *   page paints it elsewhere)
 * @param {Catalog} catalog party catalogue of the run
 * @returns {object} `partidos` and `bloques` (null without blocks): `series`, `polls`, `projection`, `result`
 *   (null without one), `parties`, `colors`, `defaults` (names shown by default, null for all) and `toPolls`
 *   (polls of the view from polls by party)
 */
export function seriesViews(initial, catalog) {
  const {series, polls, projection, result = null} = initial;
  const parties = series.parties;
  const main = (catalog.bmaps.main || []).filter((name) => parties.includes(name));
  const blocks = catalog.bmaps.blocks && !Array.isArray(catalog.bmaps.blocks) ? catalog.bmaps.blocks : {};
  const names = Object.keys(blocks);
  return {
    partidos: {
      series,
      polls,
      projection,
      result,
      parties,
      colors: mapNames(parties, (name) => catalog.color(name)),
      defaults: main.length ? main : null,
      toPolls: (chosen) => chosen,
    },
    bloques: names.length ? {
      series: blockSeries(series, blocks),
      polls: blockPolls(polls, blocks),
      projection: blockProjection(projection),
      result: result ? blockResult(result, blocks) : null,
      parties: names,
      // the colour the page already uses for the block (`block_colors`, the hemicycle of a past election), else
      // the catalogue's
      colors: mapNames(names, (name) => (initial.block_colors && initial.block_colors[name]) || catalog.blockColor(name)),
      defaults: null,
      toPolls: (chosen) => blockPolls(chosen, blocks),
    } : null,
  };
}

/**
 * Names of the view shown by a legend selection, or null when it is what the view shows without a choice
 * (its defaults, or all of them).
 *
 * @param {object} current the view (`parties`, `defaults`)
 * @param {object} legendSelected `{name: boolean}` as ECharts reports it
 * @returns {string[]|null} the chosen names, in the order of the view
 */
export function chosenNames(current, legendSelected) {
  const chosen = current.parties.filter((name) => legendSelected[name] !== false);
  const base = current.defaults || current.parties;
  const same = chosen.length === base.length && chosen.every((name) => base.includes(name));
  return same ? null : chosen;
}

/**
 * Fill the HTML legend of a chart with the names of its view, lit as its selection says, each pill toggling the
 * ECharts legend of `chart`.
 *
 * @param {HTMLElement|null} legendEl the `.legend` container (nothing without it)
 * @param {object} current the view (`parties`, `colors`, `defaults`)
 * @param {string[]|null} selected names shown (null: the defaults of the view, or all)
 * @param {object} chart the ECharts instance
 * @param {object} inks `{name: ink colour}` of the island (`initial.inks`)
 */
export function fillLegend(legendEl, current, selected, chart, inks = {}) {
  if (!legendEl) {
    return;
  }
  const shown = selected || current.defaults || current.parties;
  renderLegend(legendEl, current.parties.map((name) => ({
    name, color: current.colors[name], ink: inks[name], on: shown.includes(name),
  })), (name) => chart.dispatchAction({type: 'legendToggleSelect', name}));
}

/**
 * Draw the poll average and wire its controls: view, window and layers radios (`#series-layers`: the poll points,
 * the average or both, which `render` reads from `state.layers`) and the pill legend, which selects the parties
 * or blocks shown through the (hidden) ECharts legend. Every change rewrites the URL hash. With `imageOf` and a
 * download button (`#series-download`), the button saves the chart as shown as a signed image (`share.js`).
 *
 * @param {object} views `seriesViews` of the page
 * @param {object} state `{view, window, selected, layers}` parsed from the hash
 * @param {function} render draws a chart: `(el, polls, chartState) => chart`
 * @param {object} options `anchor` (ISO date the window control counts back from), `inks` (text colour
 *   over the colour of each name, for the pills; `initial.inks`) and `imageOf` (the option of the image:
 *   `(polls, chartState, zoom) => option`, `zoom` being the `chartZoom` of the chart on screen)
 */
export function wireAverage(views, state, render, {anchor, inks = {}, imageOf = null}) {
  const mainEl = document.getElementById('series-chart');
  const legendEl = document.getElementById('series-legend');
  const windowSet = document.getElementById('series-window');
  const viewSet = document.getElementById('series-view');
  const layersSet = document.getElementById('series-layers');
  let chart = null;

  /** Write the state of the controls to the URL hash without adding a history entry. */
  function writeHash() {
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + seriesHash(state));
  }

  /** Draw the chart, fill its pill legend and, the first time, listen to the legend selection. */
  function draw() {
    const drawn = render(mainEl, views[state.view].polls, state);
    if (!chart) {
      chart = drawn;
      chart.on('legendselectchanged', (event) => {
        state.selected = chosenNames(views[state.view], event.selected);
        if (legendEl) {
          markLegend(legendEl, event.selected);
        }
        writeHash();
      });
    }
    fillLegend(legendEl, views[state.view], state.selected, chart, inks);
  }

  for (const input of document.querySelectorAll('#series-window input[name=window]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#series-view input[name=view]')) {
    input.checked = input.value === state.view;
  }
  for (const input of document.querySelectorAll('#series-layers input[name=layers]')) {
    input.checked = input.value === state.layers;
  }
  markSegments();
  tryDraw(draw, 'the poll average');
  if (imageOf) {
    wireDownload(document.getElementById('series-download'), () => imageOf(views[state.view].polls, state, chartZoom(chart)), showError);
  }

  // The controls only work with JavaScript, so the server renders them hidden.
  if (windowSet) {
    windowSet.hidden = false;
    // `click` rather than `change`: clicking the option already checked after dragging the slider must restore
    // its window too.
    windowSet.addEventListener('click', (event) => {
      if (event.target.matches('input[name=window]') && chart) {
        state.window = event.target.value;
        const current = views[state.view];
        setWindow(chart, state.window, {series: current.series, projection: current.projection, anchor, result: current.result});
        writeHash();
      }
    });
  }
  if (layersSet) {
    layersSet.hidden = false;
    layersSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=layers]')) {
        state.layers = event.target.value;
        tryDraw(draw, 'the poll average');
        writeHash();
      }
    });
  }
  if (viewSet) {
    viewSet.hidden = false;
    viewSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=view]') && views[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        tryDraw(draw, 'the poll average');
        writeHash();
      }
    });
  }
}
