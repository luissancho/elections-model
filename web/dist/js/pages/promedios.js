// Promedios tab of an event: the poll average with its window, party and block views, one house against the
// average, the hemicycle, the evolution and the filterable, sortable table of all the polls,
// drawn from the data embedded by the server. The state of the average controls lives in the URL hash
// (`blocks.js`), written only when a control changes; the pill legend of the average selects the parties.
import {Catalog} from '../catalog.js';
import {blockPolls, blockProjection, blockSeries, parseSeriesHash, seriesHash} from '../blocks.js';
import {renderEvolution} from '../charts/evolution.js';
import {hemicycleOption, renderHemicycle} from '../charts/hemicycle.js';
import {renderSeries, setWindow} from '../charts/series.js';
import {WINDOW_DAYS} from '../range.js';
import {sparklineSvg} from '../sparkline.js';
import {mapNames, markSegments, mountWhenReady, readInitial, showError, tryDraw, wireControls} from './common.js';
import {markLegend, renderLegend} from './legend-dom.js';
import {wirePollsTable} from './polls-wire.js';
import {wireDownload} from '../share.js';

const WINDOWS = [...Object.keys(WINDOW_DAYS), 'all'];
const BLOCKS_VIEW = 'bloques';

/**
 * Draw the hemicycle, wire its Partidos/Bloques switch (which also shows the matching results table) and its
 * download button (the view drawn, as a signed image) and draw the evolution.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function paintResults(initial, catalog) {
  const {meta, headline, summary, runs} = initial;
  const mode = initial.state.mode;
  const current = headline[mode];
  const names = [...new Set([
    ...catalog.parties.map((party) => party.name),
    ...current.parties.map((party) => party.name),
    ...summary.parties.map((row) => row.name),
  ])];
  const colors = mapNames(names, (name) => catalog.color(name));
  const nSeats = meta.n_seats ?? summary.n_seats;
  const majority = meta.majority ?? summary.majority;
  const blocks = (summary.blocks || []).filter((row) => row.seats !== null && row.seats !== undefined);
  // the seat differences against the last published election, written after the seats of each arc
  const deltas = initial.deltas || {parties: {}, blocks: {}, year: null};
  const views = {
    partidos: {
      seats: summary.totals ?? Object.fromEntries(current.parties.map((party) => [party.name, party.seats])),
      colors,
      deltas: deltas.parties,
      order: (list) => catalog.order(list),
      label: 'Hemiciclo con los escaños previstos por partido',
      card: 'parties-card',
    },
    bloques: {
      seats: Object.fromEntries(blocks.map((row) => [row.name, row.seats])),
      colors: mapNames(blocks.map((row) => row.name), (name) => catalog.blockColor(name)),
      deltas: deltas.blocks,
      order: (list) => catalog.orderBlocks(list),
      label: 'Hemiciclo con los escaños previstos por bloque',
      card: 'blocks-card',
    },
  };
  const el = document.getElementById('hemicycle');
  let shown = views.partidos;  // the view drawn, the one the download button exports

  /** Draw the hemicycle of `view` and show its results table, hiding the other one. */
  function draw(view) {
    const chosen = views[view];
    shown = chosen;
    el.setAttribute('aria-label', chosen.label);
    tryDraw(() => renderHemicycle(el, chosen.seats, {
      colors: chosen.colors, order: chosen.order, nSeats, majority, deltas: chosen.deltas, year: deltas.year,
    }), 'the hemicycle');
    for (const [name, item] of Object.entries(views)) {
      const card = document.getElementById(item.card);
      if (card) {
        card.hidden = name !== view;
      }
    }
  }

  draw('partidos');
  wireDownload(document.getElementById('hemicycle-download'), (box) => hemicycleOption(shown.seats, {
    colors: shown.colors, order: shown.order, nSeats, majority, deltas: shown.deltas, year: deltas.year,
    width: box.width, height: box.height,
  }), showError);
  const fieldset = document.getElementById('results-view');
  if (fieldset) {
    fieldset.hidden = false;
    fieldset.addEventListener('change', (event) => {
      if (event.target.matches('input[name=rview]') && views[event.target.value]) {
        draw(event.target.value);
        markSegments(fieldset);
      }
    });
  }

  tryDraw(() => renderEvolution(document.getElementById('evolution'), runs.runs, mode, {
    colors, parties: current.parties.map((party) => party.name),
  }), 'the evolution');
}

/**
 * Data of the two views of the average: the parties of the series and, when the scope has blocks, the blocks
 * computed in the browser (`blocks.js`) with the published block projection.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 * @returns {object} `partidos` and `bloques` (null without blocks): `series`, `polls`, `projection`, `parties`,
 *   `colors`, `defaults` (names shown by default, null for all) and `toPolls` (polls of the view
 *   from polls by party)
 */
function seriesViews(initial, catalog) {
  const {series, polls, projection} = initial;
  const parties = series.parties;
  const main = (catalog.bmaps.main || []).filter((name) => parties.includes(name));
  const blocks = catalog.bmaps.blocks && !Array.isArray(catalog.bmaps.blocks) ? catalog.bmaps.blocks : {};
  const names = Object.keys(blocks);
  return {
    partidos: {
      series,
      polls,
      projection,
      parties,
      colors: mapNames(parties, (name) => catalog.color(name)),
      defaults: main.length ? main : null,
      toPolls: (chosen) => chosen,
    },
    bloques: names.length ? {
      series: blockSeries(series, blocks),
      polls: blockPolls(polls, blocks),
      projection: blockProjection(projection),
      parties: names,
      colors: mapNames(names, (name) => catalog.blockColor(name)),
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
function chosenNames(current, legendSelected) {
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
function fillLegend(legendEl, current, selected, chart, inks = {}) {
  if (!legendEl) {
    return;
  }
  const shown = selected || current.defaults || current.parties;
  renderLegend(legendEl, current.parties.map((name) => ({
    name, color: current.colors[name], ink: inks[name], on: shown.includes(name),
  })), (name) => chart.dispatchAction({type: 'legendToggleSelect', name}));
}

/**
 * Draw the poll average and wire its controls: view and window radios and the pill legend, which selects the
 * parties or blocks shown through the (hidden) ECharts legend. Every change rewrites the URL hash.
 *
 * @param {object} initial the page's initial data
 * @param {object} views `seriesViews` of the page
 * @param {object} state `{view, window, selected}` parsed from the hash
 * @param {function} render draws a chart: `(el, polls, chartState) => chart`
 * @returns {object} the state of the average, to seed the house chart
 */
function wireAverage(initial, views, state, render) {
  const anchor = initial.meta.date_last || initial.meta.as_of;
  const mainEl = document.getElementById('series-chart');
  const legendEl = document.getElementById('series-legend');
  const windowSet = document.getElementById('series-window');
  const viewSet = document.getElementById('series-view');
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
    fillLegend(legendEl, views[state.view], state.selected, chart, initial.inks || {});
  }

  for (const input of document.querySelectorAll('#series-window input[name=window]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#series-view input[name=view]')) {
    input.checked = input.value === state.view;
  }
  markSegments();
  tryDraw(draw, 'the poll average');

  // The controls only work with JavaScript, so the server renders them hidden.
  if (windowSet) {
    windowSet.hidden = false;
    // `click` rather than `change`: clicking the option already checked after dragging the slider must restore
    // its window too.
    windowSet.addEventListener('click', (event) => {
      if (event.target.matches('input[name=window]') && chart) {
        state.window = event.target.value;
        const current = views[state.view];
        setWindow(chart, state.window, {series: current.series, projection: current.projection, anchor});
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

/**
 * Draw the house chart and wire its own controls (view, window, house selector and legend). Its state starts as
 * the average's at load and never touches the URL hash.
 *
 * @param {object} initial the page's initial data
 * @param {object} views `seriesViews` of the page
 * @param {object} start `{view, window, selected}` of the average at load
 * @param {function} render draws a chart: `(el, polls, chartState) => chart`
 */
function wireHouse(initial, views, start, render) {
  const {meta, polls} = initial;
  const anchor = meta.date_last || meta.as_of;
  const houseEl = document.getElementById('pollster-chart');
  const legendEl = document.getElementById('pollster-legend');
  const select = document.getElementById('pollster-select');
  const controls = document.getElementById('pollster-controls');
  const windowSet = document.getElementById('pollster-window');
  const viewSet = document.getElementById('pollster-view');
  const state = {...start};
  let chart = null;

  /** Polls of the chosen house, in the current view. */
  function housePolls() {
    const house = select ? select.value : '';
    const chosen = (polls.polls || []).filter((poll) => poll.pollster === house);
    return views[state.view].toPolls({parties: polls.parties, polls: chosen});
  }

  /** Draw the chart, fill its pill legend and, the first time, listen to the legend selection. */
  function draw() {
    const drawn = render(houseEl, housePolls(), state);
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

  for (const input of document.querySelectorAll('#pollster-window input[name=pwindow]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#pollster-view input[name=pview]')) {
    input.checked = input.value === state.view;
  }
  markSegments();
  tryDraw(draw, 'the house against the average');

  if (controls) {
    controls.hidden = false;
  }
  if (windowSet) {
    windowSet.hidden = false;
    windowSet.addEventListener('click', (event) => {
      if (event.target.matches('input[name=pwindow]') && chart) {
        state.window = event.target.value;
        const current = views[state.view];
        setWindow(chart, state.window, {series: current.series, projection: current.projection, anchor});
      }
    });
  }
  if (viewSet) {
    viewSet.hidden = false;
    viewSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=pview]') && views[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        tryDraw(draw, 'the house against the average');
      }
    });
  }
  if (select) {
    // `renderSeries` redraws the same ECharts instance; the window, view and legend choices are kept.
    select.addEventListener('change', () => tryDraw(draw, 'the house against the average'));
  }
}

/**
 * Wire the poll average and, when the page has one, the house chart.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function wireSeries(initial, catalog) {
  const {meta, vote} = initial;
  const views = seriesViews(initial, catalog);
  const anchor = meta.date_last || meta.as_of;
  const state = parseSeriesHash(window.location.hash, {
    parties: views.partidos.parties,
    blocks: views.bloques ? views.bloques.parties : [],
    windows: WINDOWS,
  });
  const start = {...state};

  /** Draw the chart `el` with the polls and the `{view, window, selected}` of a chart; the ECharts legend stays
   * hidden and selectable, driven by the pill legend. */
  function render(el, pollsOfChart, chartState) {
    const current = views[chartState.view];
    return renderSeries(el, {
      series: current.series,
      polls: pollsOfChart,
      projection: current.projection,
      parties: current.parties,
      selected: chartState.selected ?? current.defaults,
      colors: current.colors,
      asOf: meta.as_of,
      when: vote.when,
      window: chartState.window,
      anchor,
      legend: 'hidden',
    });
  }

  wireAverage(initial, views, state, render);
  if (document.getElementById('pollster-chart')) {
    wireHouse(initial, views, start, render);
  }
}

/** Days of the average the sparkline of a party card covers, and the sampling step. */
const SPARK_DAYS = 365;
const SPARK_STEP = 7;

/**
 * Draw the sparkline of every party card (`article.pc[data-name]`): the mean of the last year of the average,
 * one point a week.
 *
 * @param {object} initial the page's initial data (`series.mean`)
 */
function paintCards(initial) {
  const mean = (initial.series && initial.series.mean) || {};
  for (const card of document.querySelectorAll('article.pc[data-name]')) {
    const svg = card.querySelector('svg.spark');
    const values = mean[card.dataset.name];
    if (!svg || !values) {
      continue;
    }
    const recent = values.slice(-SPARK_DAYS);
    const sampled = recent.filter((_, i) => i % SPARK_STEP === 0 || i === recent.length - 1);
    sparklineSvg(svg, sampled);
  }
}

/** Draw the page from the embedded data. */
function paint() {
  const initial = readInitial();
  // the catalogue the server builds for the page (the run's parties, those of the polls table and those of the
  // last election, named and coloured by the published catalogue), with the bmaps of the run
  const catalog = new Catalog(initial.catalog || initial.meta);
  tryDraw(() => paintCards(initial), 'the party cards');
  tryDraw(() => wireSeries(initial, catalog), 'the poll average and its controls');
  paintResults(initial, catalog);
  tryDraw(() => wirePollsTable(initial.polls.polls, catalog), 'the polls table');
}

wireControls();
mountWhenReady(paint);
