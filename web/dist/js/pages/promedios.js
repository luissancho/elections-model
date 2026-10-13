// Promedios tab of an event: the poll average with its window, party and block views, one house against the
// average, the vote and seat bars, the hemicycle, the evolution and the "Cargar más" rows of the polls table,
// drawn from the data embedded by the server. The state of the average controls lives in the URL hash
// (`blocks.js`), written only when a control changes.
import {fmtInt, fmtPct} from '../format.js';
import {Catalog} from '../catalog.js';
import {blockPolls, blockProjection, blockSeries, parseSeriesHash, seriesHash} from '../blocks.js';
import {renderBars} from '../charts/bars.js';
import {renderEvolution} from '../charts/evolution.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {renderSeries, setWindow} from '../charts/series.js';
import {captionText, nextRows} from '../polls-table.js';
import {WINDOW_DAYS} from '../range.js';
import {mapNames, markPills, mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

const WINDOWS = [...Object.keys(WINDOW_DAYS), 'all'];
const BLOCKS_VIEW = 'bloques';

/**
 * Draw the vote and seat bars, the hemicycle and the evolution.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function paintResults(initial, catalog) {
  const {meta, headline, vote, summary, runs} = initial;
  const mode = initial.state.mode;
  const current = headline[mode];
  const names = [...new Set([
    ...catalog.parties.map((party) => party.name),
    ...current.parties.map((party) => party.name),
    ...vote.rows.map((row) => row.name),
    ...summary.parties.map((row) => row.name),
  ])];
  const colors = mapNames(names, (name) => catalog.color(name));
  const nSeats = meta.n_seats ?? summary.n_seats;
  const majority = meta.majority ?? summary.majority;

  const voteRows = [...vote.rows].sort((a, b) => b.pct - a.pct);
  tryDraw(() => renderBars(document.getElementById('vote-chart'), voteRows, {
    value: 'pct', lo: 'lo', hi: 'hi', colors,
    formatter: (x) => fmtPct(x), axisFormatter: (x) => fmtPct(x, 0),
  }), 'the vote bars');

  const seatRows = summary.parties
    .filter((row) => row.seats > 0 || row.seats_hi > 0)
    .sort((a, b) => b.seats - a.seats);
  tryDraw(() => renderBars(document.getElementById('seats-chart'), seatRows, {
    value: 'seats', lo: 'seats_lo', hi: 'seats_hi', colors, formatter: fmtInt, majority,
  }), 'the seat bars');

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
 * Draw the average and the house chart and wire their controls: view, window, party and block pills and the
 * house selector. The pills replace the ECharts legends, which are hidden. Every change of a control rewrites
 * the URL hash.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function wireSeries(initial, catalog) {
  const {meta, polls, vote} = initial;
  const views = seriesViews(initial, catalog);
  const anchor = meta.date_last || meta.as_of;
  const state = parseSeriesHash(window.location.hash, {
    parties: views.partidos.parties,
    blocks: views.bloques ? views.bloques.parties : [],
    windows: WINDOWS,
  });
  const mainEl = document.getElementById('series-chart');
  const houseEl = document.getElementById('pollster-chart');
  const select = document.getElementById('pollster-select');
  const houseControls = document.getElementById('pollster-controls');
  const fieldsets = {
    window: document.getElementById('series-window'),
    view: document.getElementById('series-view'),
    partidos: document.getElementById('series-parties'),
    bloques: document.getElementById('series-blocks'),
  };
  const charts = [];

  const view = () => views[state.view];
  const selected = () => state.selected ?? view().defaults;
  const boxes = (name) => (fieldsets[name] ? [...fieldsets[name].querySelectorAll('input[type=checkbox]')] : []);

  /** Write the state of the controls to the URL hash without adding a history entry. */
  function writeHash() {
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + seriesHash(state));
  }

  /** Polls of the chosen house, in the current view. */
  function housePolls() {
    const house = select ? select.value : '';
    const chosen = (polls.polls || []).filter((poll) => poll.pollster === house);
    return view().toPolls({parties: polls.parties, polls: chosen});
  }

  /** Check the pills of the current view that are shown and show only its fieldset. */
  function syncBoxes() {
    const shown = selected();
    for (const name of ['partidos', 'bloques']) {
      if (fieldsets[name]) {
        fieldsets[name].hidden = name !== state.view;
      }
    }
    for (const box of boxes(state.view)) {
      box.checked = !shown || shown.includes(box.value);
    }
    markPills(fieldsets[state.view]);
  }

  /** Show or hide the names of the current view in both charts; a merged option keeps their zoom. */
  function applySelection() {
    const shown = selected();
    const legend = {selected: mapNames(view().parties, (name) => !shown || shown.includes(name))};
    for (const chart of charts) {
      chart.setOption({legend});
    }
  }

  /** Draw the chart `el` with the current view, window and selection and the given polls; returns the chart. */
  function render(el, pollsOfChart) {
    const current = view();
    return renderSeries(el, {
      series: current.series,
      polls: pollsOfChart,
      projection: current.projection,
      parties: current.parties,
      selected: selected(),
      colors: current.colors,
      asOf: meta.as_of,
      when: vote.when,
      window: state.window,
      anchor,
      legend: false,
    });
  }

  /** Draw the average chart and, when there is one, the house chart. */
  function drawAll() {
    charts.length = 0;
    tryDraw(() => charts.push(render(mainEl, view().polls)), 'the poll average');
    if (houseEl) {
      tryDraw(() => charts.push(render(houseEl, housePolls())), 'the house against the average');
    }
  }

  for (const input of document.querySelectorAll('#series-window input[name=window]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#series-view input[name=view]')) {
    input.checked = input.value === state.view;
  }
  syncBoxes();
  drawAll();

  // The controls only work with JavaScript, so the server renders them hidden.
  if (houseControls) {
    houseControls.hidden = false;
  }
  if (fieldsets.window) {
    fieldsets.window.hidden = false;
    // `click` rather than `change`: clicking the option already checked after dragging the slider must restore
    // its window too.
    fieldsets.window.addEventListener('click', (event) => {
      if (event.target.matches('input[name=window]')) {
        state.window = event.target.value;
        for (const chart of charts) {
          setWindow(chart, state.window, {series: view().series, projection: view().projection, anchor});
        }
        writeHash();
      }
    });
  }
  if (fieldsets.view) {
    fieldsets.view.hidden = false;
    fieldsets.view.addEventListener('change', (event) => {
      if (event.target.matches('input[name=view]') && views[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        syncBoxes();
        drawAll();
        writeHash();
      }
    });
  }
  for (const name of ['partidos', 'bloques']) {
    const fieldset = fieldsets[name];
    if (!fieldset) {
      continue;
    }
    fieldset.addEventListener('change', () => {
      state.selected = boxes(name).filter((box) => box.checked).map((box) => box.value);
      markPills(fieldset);
      applySelection();
      writeHash();
    });
  }
  if (select && houseEl) {
    // `renderSeries` redraws the same ECharts instance, so `charts` keeps pointing at it.
    select.addEventListener('change', () => {
      tryDraw(() => render(houseEl, housePolls()), 'the house against the average');
    });
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
      button.hidden = true;
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
