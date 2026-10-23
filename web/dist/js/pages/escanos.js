// Seats page: the ridge of the party seat distributions, stacked blocks, majority calculator, vote fan and seat evolution, drawn
// from the data embedded by the server.
import {fmtInt} from '../format.js';
import {Catalog} from '../catalog.js';
import {THEME} from '../charts/base.js';
import {renderEvolution} from '../charts/evolution.js';
import {renderFan} from '../charts/fan.js';
import {renderHistogram} from '../charts/histogram.js';
import {renderRidge, ridgeRowHeight} from '../charts/ridge.js';
import {ridgeBlocks, ridgeData} from '../kde.js';
import {renderStacked} from '../charts/stacked.js';
import {canGroup, sortTerritory, territoryRows} from '../territory.js';
import {renderBody, renderHead} from './territory-dom.js';
import {coalitionCaption, coalitionHash, coalitionSeats, coalitionText, parseCoalition, summarize} from '../seats.js';
import {mapNames, markPills, markSegments, mountWhenReady, readInitial, tryDraw, wireControls, wireForm} from './common.js';

const EMPTY_COALITION = 'Marca partidos para calcular su mayoría.';
/** Shared seat axis of the ridges, as a share of the chamber: parties up to 60%, blocks up to 80% (Luis, 2026-10-14). */
const PARTY_AXIS_SHARE = 0.6;
const BLOCK_AXIS_SHARE = 0.8;

/** Order of the party rows of the seats bar and the ridges: by headline seats, then by mean seats. */
const bySeats = (a, b) => ((b.seats ?? 0) - (a.seats ?? 0)) || ((b.seats_mean ?? 0) - (a.seats_mean ?? 0));

/**
 * Data of the party bar of the central allocation (the headline seats), shared by «Reparto de escaños» and the
 * majority calculator: the parties with seats and a distribution, in the catalogue order, with their colours, full
 * names and seat differences against the last published election.
 *
 * @param {object} initial the page's initial data (`summary`, `dist`, `deltas`)
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} colors `{party: colour}`
 * @returns {object} `names` (by seats), `rows` (`{name, seats}` in the catalogue order), `colors`, `fullnames`,
 *   `deltas` and `year`
 */
function partyBar(initial, catalog, colors) {
  const {summary, dist} = initial;
  const partyRows = (summary.parties || []).filter((row) => row.seats !== null && row.seats !== undefined);
  const names = [...partyRows].sort(bySeats).map((row) => row.name).filter((name) => dist.parties.includes(name));
  const deltas = initial.deltas || {parties: {}, year: null};
  return {
    names,
    rows: catalog.order(names).map((name) => ({name, seats: partyRows.find((row) => row.name === name).seats})),
    colors,
    fullnames: mapNames(names, (name) => catalog.fullname(name)),
    deltas: deltas.parties,
    year: deltas.year,
  };
}

/**
 * Wire the majority calculator: restore the selection of the URL fragment, then recompute the text
 * (`coalitionText`), the bar of the central allocation (the marked parties first, in colour, and the rest after
 * them in grey) and the histogram of the marked parties, marked at their central seats, with its caption
 * (`coalitionCaption`) on load and on every change. Only a change rewrites the fragment, so a link to another
 * section (`#districts`) survives the load.
 *
 * @param {object} dist the dist part
 * @param {number} majority seats of the absolute majority
 * @param {object} bar `partyBar` of the page
 * @param {number} nSeats seats of the chamber
 */
function wireCoalition(dist, majority, bar, nSeats) {
  const fieldset = document.getElementById('coalition-parties');
  const result = document.getElementById('coalition-result');
  const chart = document.getElementById('coalition-chart');
  const barEl = document.getElementById('coalition-bar');
  if (!fieldset || !result || !chart) {
    return;
  }
  const boxes = [...fieldset.querySelectorAll('input[type=checkbox][name=coalition]')];
  const fromHash = parseCoalition(window.location.hash, dist.parties);
  if (fromHash !== null) {
    for (const box of boxes) {
      box.checked = fromHash.includes(box.value);
    }
  }

  function update() {
    markPills(fieldset);
    const names = boxes.filter((box) => box.checked).map((box) => box.value);
    const values = coalitionSeats(dist, names);
    const stats = summarize(values, majority);
    if (barEl) {
      const muted = bar.rows.map((row) => row.name).filter((name) => !names.includes(name));
      tryDraw(() => renderStacked(barEl, [
        ...bar.rows.filter((row) => names.includes(row.name)), ...bar.rows.filter((row) => muted.includes(row.name)),
      ], {
        colors: bar.colors, fullnames: bar.fullnames, majority, total: nSeats, deltas: bar.deltas, year: bar.year, muted,
      }), 'the majority bar');
    }
    result.classList.toggle('ok', Boolean(stats) && stats.pMajority >= 0.5);
    if (!stats) {
      result.textContent = EMPTY_COALITION;
      if (chart.__chart) {
        chart.__chart.clear();
      }
    } else {
      const central = bar.rows.filter((row) => names.includes(row.name)).reduce((sum, row) => sum + row.seats, 0);
      result.textContent = coalitionText(names.length, central, stats);
      renderHistogram(chart, values, {
        majority, median: central, markLabel: `Reparto central: ${fmtInt(central)}`, lo: stats.lo, hi: stats.hi,
        color: THEME.barColor, bounds: true, markLabels: false, caption: coalitionCaption(stats, central),
      });
    }
    return names;
  }

  fieldset.addEventListener('change', () => {
    const names = update();
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + coalitionHash(names));
  });
  update();
}

const CAPTION = {
  esc: 'Reparto central: D\'Hondt en cada circunscripción sobre el voto central; cada fila suma sus escaños. '
    + 'Pasa el ratón para ver el rango del 95%.',
  ccaa: 'Reparto central: D\'Hondt en cada circunscripción sobre el voto central; cada fila suma los escaños de '
    + 'las provincias de la comunidad.',
  pct: 'Voto estimado en cada territorio; en CCAA, la media ponderada por los votos de la elección anterior.',
};

/**
 * Wire "Resultados por territorio": rebuild the table of the central allocation from the embedded `districts`,
 * `scenario` and `summary` for the chosen level (CCAA or province) and mode (seats or vote), sortable from its
 * headers. The server draws the province-by-seats table; this replaces it and shows the controls.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function wireTerritory(initial, catalog) {
  const controls = document.getElementById('territory-controls');
  const table = document.getElementById('district-table');
  if (!controls || !table || !initial.districts || !initial.scenario) {
    return;
  }
  const levels = document.getElementById('territory-level');
  const {scope, mode: runMode, run, pinned} = initial.state;
  const state = {sort: {key: 'name', asc: true}};
  const checked = (name, fallback) => {
    const input = document.querySelector(`#territory-controls input[name="${name}"]:checked`);
    return input ? input.value : fallback;
  };
  const hrefOf = (row) => (state.level === 'prov' && row.id !== null
    ? `/${scope}/escanos?mode=${runMode}${pinned ? `&run=${run}` : ''}&region=${row.id}#districts`
    : null);

  const render = (focusKey = null) => {
    state.level = levels && canGroup(initial.districts.regions || []) ? checked('level', 'ccaa') : 'prov';
    const mode = checked('tmode', 'esc');
    const data = territoryRows(initial.districts, initial.scenario, initial.summary,
      {level: state.level, mode});
    if (state.sort.key !== 'name' && state.sort.key !== 'total' && !data.parties.includes(state.sort.key)) {
      state.sort = {key: 'name', asc: true};
    }
    const caption = table.querySelector('caption');
    if (caption) {
      caption.textContent = mode === 'esc' && state.level === 'ccaa' ? CAPTION.ccaa : CAPTION[mode];
    }
    const sorted = {...data, rows: sortTerritory(data.rows, state.sort.key, state.sort.asc)};
    renderHead(table.tHead, data.parties, catalog, state.sort, state.level === 'ccaa' ? 'CCAA' : 'Provincia');
    renderBody(table.tBodies[0], sorted, mode, catalog, hrefOf);
    if (focusKey) {
      const button = table.querySelector(`th[data-key="${CSS.escape(focusKey)}"] .sort`);
      if (button) {
        button.focus();
      }
    }
  };

  controls.addEventListener('change', () => render());
  table.tHead.addEventListener('click', (event) => {
    const th = event.target.closest('th[data-key]');
    if (!th) {
      return;
    }
    const key = th.dataset.key;
    state.sort = key === state.sort.key ? {key, asc: !state.sort.asc} : {key, asc: key === 'name'};
    render(key);
  });
  controls.hidden = false;
  render();
}

/**
 * Wire the seats card: the Partidos/Bloques switch draws the stacked bar and the ridges of the view and shows
 * its table. Parties are ordered by headline seats (then by mean, like `Simulator.plot_dist_kde`); blocks the
 * same way, their simulations being the sum of their parties.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} colors `{party: colour}`
 */
function wireSeats(initial, catalog, colors) {
  const {meta, summary, dist} = initial;
  const majority = meta.majority ?? summary.majority;
  const nSeats = meta.n_seats ?? summary.n_seats;
  const parties = partyBar(initial, catalog, colors);
  const blockRows = (summary.blocks || []).filter((row) => row.seats !== null && row.seats !== undefined);
  const blockParties = catalog.bmaps.blocks && !Array.isArray(catalog.bmaps.blocks) ? catalog.bmaps.blocks : {};
  const blockColors = mapNames(blockRows.map((row) => row.name), (name) => catalog.blockColor(name));
  // the seat differences against the last published election, written after the seats of each segment of the bar
  const deltas = initial.deltas || {parties: {}, blocks: {}, year: null};
  const views = {
    partidos: {
      container: 'seats-parties',
      bar: parties.rows,
      colors,
      deltas: parties.deltas,
      fullnames: parties.fullnames,
      ridges: () => ridgeData(dist, parties.names, {
        majority,
        range: [0, Math.round(nSeats * PARTY_AXIS_SHARE)],
        central: Object.fromEntries(parties.rows.map((row) => [row.name, row.seats])),
      }),
    },
    bloques: {
      container: 'seats-blocks',
      bar: catalog.orderBlocks(blockRows.map((row) => row.name))
        .map((name) => ({name, seats: blockRows.find((row) => row.name === name).seats})),
      colors: blockColors,
      deltas: deltas.blocks,
      fullnames: {},
      ridges: () => ridgeBlocks(dist, [...blockRows].sort(bySeats)
        .map((row) => ({name: row.name, parties: blockParties[row.name] || []})),
      {
        majority,
        range: [0, Math.round(nSeats * BLOCK_AXIS_SHARE)],
        central: Object.fromEntries(blockRows.map((row) => [row.name, row.seats])),
      }),
    },
  };
  const barEl = document.getElementById('seats-bar');
  const ridgeEl = document.getElementById('ridge-chart');
  const cache = {};

  /** Draw the bar and the ridges of `view` and show its table. */
  function draw(view) {
    const chosen = views[view];
    for (const [name, item] of Object.entries(views)) {
      const container = document.getElementById(item.container);
      if (container) {
        container.hidden = name !== view;
      }
    }
    if (barEl) {
      tryDraw(() => renderStacked(barEl, chosen.bar, {
        colors: chosen.colors, fullnames: chosen.fullnames, majority, total: nSeats, deltas: chosen.deltas, year: deltas.year,
      }), `the ${view} bar`);
    }
    if (ridgeEl) {
      cache[view] = cache[view] || chosen.ridges();
      if (cache[view].length) {
        tryDraw(() => renderRidge(ridgeEl, cache[view], {
          colors: chosen.colors, majority, fullnames: chosen.fullnames, rowHeight: ridgeRowHeight(window.innerWidth),
        }), `the ${view} distributions`);
      }
    }
  }

  draw('partidos');
  const fieldset = document.getElementById('seats-view');
  if (fieldset) {
    fieldset.hidden = false;
    fieldset.addEventListener('change', (event) => {
      if (event.target.matches('input[name=sview]') && views[event.target.value]) {
        draw(event.target.value);
        markSegments(fieldset);
      }
    });
  }
}

/** Draw the seats card, the fan and the evolution, and wire the majority calculator and the territory table. */
function paint() {
  const initial = readInitial();
  const {state, meta, summary, dist, fan, runs} = initial;

  const catalog = new Catalog(meta);
  const majority = meta.majority ?? summary.majority;
  const nSeats = meta.n_seats ?? summary.n_seats;
  const partyRows = summary.parties || [];
  const fanNames = catalog.order([...new Set((fan.rows || []).map((row) => row.name).filter(Boolean))]);
  const names = [...new Set([...dist.parties, ...partyRows.map((row) => row.name), ...fanNames])];
  const colors = mapNames(names, (name) => catalog.color(name));

  tryDraw(() => wireSeats(initial, catalog, colors), 'the seats card');

  tryDraw(() => wireCoalition(dist, majority, partyBar(initial, catalog, colors), nSeats), 'the majority calculator');
  tryDraw(() => wireTerritory(initial, catalog), 'the territory table');

  const main = (catalog.bmaps.main || []).filter((name) => fanNames.includes(name));
  tryDraw(() => renderFan(document.getElementById('fan-chart'), fan, {
    parties: fanNames,
    selected: main.length ? main : null,
    colors,
    markAt: state.mode === 'nowcast' ? 0 : meta.horizon_max,
    markLabel: state.mode === 'nowcast' ? 'Hoy' : 'Elección',
  }), 'the fan');

  tryDraw(() => renderEvolution(document.getElementById('seats-evolution'), runs.runs, state.mode, {
    colors, field: 'seats', formatter: fmtInt, parties: partyRows.map((row) => row.name),
  }), 'the seat evolution');
}

wireControls();
wireForm('district-form');
mountWhenReady(paint);
