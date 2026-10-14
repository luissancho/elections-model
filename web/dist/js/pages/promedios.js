// Promedios tab of an event: the poll average with its window, party and block views, one house against the
// average, the hemicycle, the evolution and the filterable, sortable table of all the polls,
// drawn from the data embedded by the server. The state of the average controls lives in the URL hash
// (`blocks.js`), written only when a control changes; the legend of the average selects the parties.
import {Catalog} from '../catalog.js';
import {blockPolls, blockProjection, blockSeries, parseSeriesHash, seriesHash} from '../blocks.js';
import {renderEvolution} from '../charts/evolution.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {renderSeries, setWindow} from '../charts/series.js';
import {PAGE_SIZE, filterPolls, meanOf, pageLabel, pageOf, pollsCount, sortPolls} from '../polls-table.js';
import {WINDOW_DAYS} from '../range.js';
import {mapNames, mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';
import {renderBody, renderFoot, renderHead} from './polls-dom.js';

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
 * Wire the controls of a dropdown of checkboxes (`details.dd`): "Seleccionar todos" and "Quitar todos" (which
 * leaves the first one checked), at least one box always checked, and the counter of its summary.
 *
 * @param {HTMLElement} dd the `details` element
 * @param {Function} onChange called after any change
 * @returns {function(): HTMLInputElement[]} the boxes that are checked now
 */
function wireDropdown(dd, onChange) {
  const boxes = [...dd.querySelectorAll('input[type="checkbox"]')];
  const sync = () => {
    dd.querySelector('[data-count]').textContent = String(boxes.filter((box) => box.checked).length);
    boxes.forEach((box) => box.closest('.chip').classList.toggle('is-on', box.checked));
  };
  const set = (checked) => {
    boxes.forEach((box) => { box.checked = checked; });
    if (!checked && boxes.length) {
      boxes[0].checked = true;
    }
    sync();
    onChange();
  };
  dd.querySelector('[data-all]').addEventListener('click', () => set(true));
  dd.querySelector('[data-none]').addEventListener('click', () => set(false));
  dd.addEventListener('change', (event) => {
    if (!boxes.some((box) => box.checked)) {
      event.target.checked = true;
    }
    sync();
    onChange();
  });
  sync();
  return () => boxes.filter((box) => box.checked);
}

/**
 * Wire the "Todas las encuestas" table: party and pollster dropdowns, month range, sortable headers and pages of
 * `PAGE_SIZE` rows, all computed in the browser from `initial.polls`. The server renders the 15 most recent rows
 * and hides the controls, which this function shows.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function wirePollsTable(initial, catalog) {
  const controls = document.getElementById('polls-controls');
  const table = document.getElementById('polls-table');
  const pag = document.getElementById('polls-pag');
  if (!controls || !table || !pag) {
    return;
  }
  const polls = initial.polls.polls;
  const from = document.getElementById('polls-from');
  const to = document.getElementById('polls-to');
  const state = {sort: {key: 'date', asc: false}, page: 0};
  const [prev, next] = pag.querySelectorAll('button');
  const pick = (checked, total) => (checked.length === total ? null : checked.map((box) => box.value));

  const render = () => {
    const parties = checkedParties().map((box) => box.value);
    const pollsters = pick(checkedPollsters(), pollsterTotal);
    const rows = sortPolls(filterPolls(polls, {
      pollsters,
      from: from.selectedIndex <= 0 ? null : from.value,
      to: to.selectedIndex === to.options.length - 1 ? null : to.value,
    }), state.sort.key, state.sort.asc);
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const shown = pageOf(rows, state.page);
    renderHead(table.tHead, parties, catalog, state.sort);
    renderBody(table.tBodies[0], shown, parties, catalog);
    renderFoot(table.tFoot, rows.length, meanOf(rows, parties), parties, catalog);
    document.getElementById('polls-count').textContent = pollsCount(rows.length);
    const recent = state.sort.key === 'date' && !state.sort.asc;
    prev.textContent = recent ? '‹ Más recientes' : '‹ Anteriores';
    next.textContent = recent ? 'Más antiguos ›' : 'Siguientes ›';
    prev.disabled = state.page === 0;
    next.disabled = state.page >= pages - 1;
    document.getElementById('polls-range').textContent = pageLabel(state.page, shown.length, rows.length);
    pag.hidden = pages <= 1;
  };
  const refilter = () => { state.page = 0; render(); };
  const checkedParties = wireDropdown(document.getElementById('polls-parties'), refilter);
  const checkedPollsters = wireDropdown(document.getElementById('polls-pollsters'), refilter);
  const pollsterTotal = document.querySelectorAll('#polls-pollsters input[type="checkbox"]').length;

  from.addEventListener('change', refilter);
  to.addEventListener('change', refilter);
  table.tHead.addEventListener('click', (event) => {
    const th = event.target.closest('th[data-key]');
    if (!th) {
      return;
    }
    const key = th.dataset.key;
    state.sort = key === state.sort.key ? {key, asc: !state.sort.asc} : {key, asc: key === 'pollster'};
    state.page = 0;
    render();
  });
  pag.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-page]');
    if (button && !button.disabled) {
      state.page += Number(button.dataset.page);
      render();
    }
  });
  document.addEventListener('click', (event) => {
    for (const dd of controls.querySelectorAll('details.dd[open]')) {
      if (!dd.contains(event.target)) {
        dd.open = false;
      }
    }
  });
  controls.hidden = false;
  render();
}

/** Draw the page from the embedded data. */
function paint() {
  const initial = readInitial();
  const catalog = new Catalog(initial.meta);
  tryDraw(() => wireSeries(initial, catalog), 'the poll average and its controls');
  paintResults(initial, catalog);
  tryDraw(() => wirePollsTable(initial, catalog), 'the polls table');
}

wireControls();
mountWhenReady(paint);
