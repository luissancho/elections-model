// Promedios tab of an event: the poll average with its window, party and block views, one house against the
// average, the hemicycle, the evolution and the "Cargar más" rows of the polls table,
// drawn from the data embedded by the server. The state of the average controls lives in the URL hash
// (`blocks.js`), written only when a control changes; the legend of the average selects the parties.
import {Catalog} from '../catalog.js';
import {blockPolls, blockProjection, blockSeries, parseSeriesHash, seriesHash} from '../blocks.js';
import {renderEvolution} from '../charts/evolution.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {renderSeries, setWindow} from '../charts/series.js';
import {captionText, nextRows} from '../polls-table.js';
import {WINDOW_DAYS} from '../range.js';
import {mapNames, mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

const WINDOWS = [...Object.keys(WINDOW_DAYS), 'all'];
const BLOCKS_VIEW = 'bloques';

/**
 * Draw the hemicycle and the evolution.
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

  const seats = summary.totals
    ?? Object.fromEntries(current.parties.map((party) => [party.name, party.seats]));
  tryDraw(() => renderHemicycle(document.getElementById('hemicycle'), seats, {
    colors, order: (list) => catalog.order(list), nSeats, majority,
  }), 'the hemicycle');

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
 * Draw the poll average and wire its controls: view and window radios and the ECharts legend, which selects the
 * parties or blocks shown. Every change rewrites the URL hash.
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
  const windowSet = document.getElementById('series-window');
  const viewSet = document.getElementById('series-view');
  let chart = null;

  /** Write the state of the controls to the URL hash without adding a history entry. */
  function writeHash() {
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + seriesHash(state));
  }

  /** Draw the chart and, the first time, listen to its legend. */
  function draw() {
    const drawn = render(mainEl, views[state.view].polls, state);
    if (!chart) {
      chart = drawn;
      chart.on('legendselectchanged', (event) => {
        state.selected = chosenNames(views[state.view], event.selected);
        writeHash();
      });
    }
  }

  for (const input of document.querySelectorAll('#series-window input[name=window]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#series-view input[name=view]')) {
    input.checked = input.value === state.view;
  }
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

  /** Draw the chart and, the first time, listen to its legend. */
  function draw() {
    const drawn = render(houseEl, housePolls(), state);
    if (!chart) {
      chart = drawn;
      chart.on('legendselectchanged', (event) => {
        state.selected = chosenNames(views[state.view], event.selected);
      });
    }
  }

  for (const input of document.querySelectorAll('#pollster-window input[name=pwindow]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#pollster-view input[name=pview]')) {
    input.checked = input.value === state.view;
  }
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

  /** Draw the chart `el` with the polls and the `{view, window, selected}` of a chart, legend on. */
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
      legend: true,
    });
  }

  wireAverage(initial, views, state, render);
  if (document.getElementById('pollster-chart')) {
    wireHouse(initial, views, start, render);
  }
}

/**
 * Wire the "Cargar más" button of the polls table: when the page embeds every row (`initial.table`, only with
 * more polls than the server rendered), each click appends the next rows and updates the caption; the button
 * goes away once the table holds them all.
 *
 * @param {object} initial the page's initial data
 */
function wirePollsTable(initial) {
  const all = initial.table ? initial.table.rows : null;
  const button = document.getElementById('polls-more');
  const body = document.querySelector('#polls-table tbody');
  const count = document.getElementById('polls-count');
  if (!all || !button || !body) {
    return;
  }
  let shown = body.rows.length;
  button.hidden = shown >= all.length;
  button.addEventListener('click', () => {
    for (const row of nextRows(all, shown)) {
      const tr = document.createElement('tr');
      for (const cell of row) {
        const td = document.createElement('td');
        td.textContent = cell;
        tr.appendChild(td);
      }
      body.appendChild(tr);
      shown += 1;
    }
    if (count) {
      count.textContent = captionText(shown, all.length);
    }
    if (shown >= all.length) {
      // The button goes away: keep the keyboard focus on the caption that says every poll is shown.
      button.hidden = true;
      if (count) {
        count.tabIndex = -1;
        count.focus();
      }
    }
  });
}

/** Draw the page from the embedded data. */
function paint() {
  const initial = readInitial();
  const catalog = new Catalog(initial.meta);
  tryDraw(() => wireSeries(initial, catalog), 'the poll average and its controls');
  paintResults(initial, catalog);
  tryDraw(() => wirePollsTable(initial), 'the polls table');
}

wireControls();
mountWhenReady(paint);
