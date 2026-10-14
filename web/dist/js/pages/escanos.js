// Seats page: party histograms, stacked blocks, coalition calculator, vote fan and seat evolution, drawn
// from the data embedded by the server.
import {fmtInt, fmtProb, fmtRange} from '../format.js';
import {Catalog} from '../catalog.js';
import {THEME} from '../charts/base.js';
import {renderEvolution} from '../charts/evolution.js';
import {renderFan} from '../charts/fan.js';
import {renderHistogram} from '../charts/histogram.js';
import {renderStacked} from '../charts/stacked.js';
import {canGroup, sortTerritory, territoryRows} from '../territory.js';
import {renderBody, renderHead} from './territory-dom.js';
import {coalitionHash, coalitionSeats, column, parseCoalition, summarize} from '../seats.js';
import {mapNames, markPills, mountWhenReady, readInitial, tryDraw, wireControls, wireForm} from './common.js';

const EMPTY_COALITION = 'Marca partidos para calcular su mayoría.';

/**
 * Wire the coalition calculator: restore the selection of the URL fragment, then recompute the summary
 * and the histogram of the marked parties on load and on every change. Only a change rewrites the
 * fragment, so a link to another section (`#districts`) survives the load.
 *
 * @param {object} dist the dist part
 * @param {number} majority seats of the absolute majority
 */
function wireCoalition(dist, majority) {
  const fieldset = document.getElementById('coalition-parties');
  const result = document.getElementById('coalition-result');
  const chart = document.getElementById('coalition-chart');
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
    if (!stats) {
      result.textContent = EMPTY_COALITION;
      if (chart.__chart) {
        chart.__chart.clear();
      }
    } else {
      const count = `${names.length} ${names.length === 1 ? 'partido' : 'partidos'}`;
      result.textContent = [
        count,
        `mediana ${fmtInt(stats.median)} escaños`,
        `rango 95 % ${fmtRange(stats.lo, stats.hi, 0)}`,
        `mayoría absoluta (${fmtInt(majority)}): ${fmtProb(stats.pMajority)}`,
      ].join(' · ');
      renderHistogram(chart, values, {
        majority, median: stats.median, lo: stats.lo, hi: stats.hi, color: THEME.barColor, bounds: true,
        markLabels: false,
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
  esc: 'Escenario central: la simulación más cercana a la mediana de escaños; cada fila suma los escaños de su '
    + 'circunscripción. Pasa el ratón para ver el rango del 95 %.',
  pct: 'Voto estimado en cada territorio; en CCAA, la media ponderada por los votos de la elección anterior.',
};

/**
 * Wire "Resultados por territorio": rebuild the table of the central scenario from the embedded `districts`,
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
      caption.textContent = CAPTION[mode];
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

/** Draw the histograms, the blocks bar, the fan and the evolution, and wire the coalition calculator. */
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

  for (const el of document.querySelectorAll('#histograms div[data-party]')) {
    const name = el.dataset.party;
    const values = column(dist, name);
    if (!values.length) {
      el.hidden = true;
      continue;
    }
    const stats = summarize(values, majority);
    tryDraw(() => renderHistogram(el, values, {
      color: colors[name], majority, median: stats.median, lo: stats.lo, hi: stats.hi, bounds: true,
      markLabels: false,
    }), `the histogram of ${name}`);
  }

  const blocks = new Map((summary.blocks || []).map((row) => [row.name, row]));
  const blockRows = catalog.orderBlocks([...blocks.keys()])
    .map((name) => ({name, seats: blocks.get(name).seats ?? 0}));
  tryDraw(() => renderStacked(document.getElementById('blocks-chart'), blockRows, {
    colors: mapNames([...blocks.keys()], (name) => catalog.blockColor(name)),
    majority,
    total: nSeats,
  }), 'the blocks');

  tryDraw(() => wireCoalition(dist, majority), 'the coalition calculator');
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
