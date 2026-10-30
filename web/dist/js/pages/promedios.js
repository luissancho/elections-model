// Promedios tab of an event: the poll average with its window, party and block views (wired by
// `series-wire.js`, shared with the detail of a past election), one house against the average, the hemicycle,
// the evolution and the filterable, sortable table of all the polls, drawn from the data embedded by the server.
// The state of the average controls lives in the URL hash (`blocks.js`), written only when a control changes;
// the pill legend of the average selects the parties.
import {Catalog} from '../catalog.js';
import {parseSeriesHash} from '../blocks.js';
import {hemicycleOption, renderHemicycle} from '../charts/hemicycle.js';
import {chartZoom, renderSeries, seriesImageOption, setWindow} from '../charts/series.js';
import {sparklineSvg} from '../sparkline.js';
import {mapNames, markSegments, mountWhenReady, readInitial, showError, tryDraw, wireControls} from './common.js';
import {markLegend} from './legend-dom.js';
import {wireEvolution} from './evolution-wire.js';
import {wirePollsTable} from './polls-wire.js';
import {WINDOWS, chosenNames, fillLegend, seriesViews, wireAverage} from './series-wire.js';
import {fileSlug, tableImage, wireDownload} from '../share.js';

/** Size of the hemicycle image: landscape, the hemicycle at the left and its results table at the right. */
const HEMICYCLE_IMAGE = {width: 1200, chartHeight: 360};

/**
 * Draw the hemicycle, wire its Partidos/Bloques switch (which also shows the matching results table) and its
 * download button (the view drawn beside its results table, as a signed image) and draw the evolution.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function paintResults(initial, catalog) {
  const {meta, headline, summary} = initial;
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
  }), showError, {...HEMICYCLE_IMAGE, side: () => tableImage(document.querySelector(`#${shown.card} table`))});
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

  tryDraw(() => wireEvolution(initial, catalog), 'the evolution');
}

/**
 * Draw the house chart and wire its own controls (view, window, house selector and legend) and its download
 * button (the chart as shown, its subtitle and file named after the house chosen). Its state starts as the
 * average's at load and never touches the URL hash.
 *
 * @param {object} initial the page's initial data
 * @param {object} views `seriesViews` of the page
 * @param {object} start `{view, window, selected, layers}` of the average at load (its layers are not kept: the
 *   house chart always draws its polls over the average)
 * @param {function} render draws a chart: `(el, polls, chartState) => chart`
 * @param {function} imageOf the option of its image: `(polls, chartState, zoom) => option`
 */
function wireHouse(initial, views, start, render, imageOf) {
  const {meta, polls} = initial;
  const anchor = meta.date_last || meta.as_of;
  const houseEl = document.getElementById('pollster-chart');
  const legendEl = document.getElementById('pollster-legend');
  const select = document.getElementById('pollster-select');
  const controls = document.getElementById('pollster-controls');
  const windowSet = document.getElementById('pollster-window');
  const viewSet = document.getElementById('pollster-view');
  const state = {...start, layers: 'ambos'};
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
  wireDownload(document.getElementById('pollster-download'), () => imageOf(housePolls(), state, chartZoom(chart)), showError, {
    texts: ({subtitle, file}) => {
      const house = select ? select.value : '';
      return house ? {subtitle: `${house} ${subtitle}`, file: file.replace(/\.png$/, `-${fileSlug(house)}.png`)} : {};
    },
  });

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

  /** Data of a chart (`seriesOption`) with the polls and the `{view, window, selected, layers}` of a chart; the
   * ECharts legend stays hidden and selectable, driven by the pill legend. */
  function dataOf(pollsOfChart, chartState) {
    const current = views[chartState.view];
    return {
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
      layers: chartState.layers,
    };
  }

  /** Draw the chart `el` with the polls and the state of a chart. */
  function render(el, pollsOfChart, chartState) {
    return renderSeries(el, dataOf(pollsOfChart, chartState));
  }

  /** Option of the image of a chart (`seriesImageOption`) with its polls, state and zoom on screen. */
  const imageOf = (pollsOfChart, chartState, zoom) => seriesImageOption(dataOf(pollsOfChart, chartState), zoom);

  wireAverage(views, state, render, {anchor, inks: initial.inks || {}, imageOf});
  if (document.getElementById('pollster-chart')) {
    wireHouse(initial, views, start, render, imageOf);
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
