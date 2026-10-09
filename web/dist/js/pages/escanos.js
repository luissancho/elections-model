// Seats page: party histograms, stacked blocks, coalition calculator, vote fan and seat evolution, drawn
// from the data embedded by the server.
import {fmtInt, fmtProb, fmtRange} from '../format.js';
import {Catalog} from '../catalog.js';
import {THEME} from '../charts/base.js';
import {renderEvolution} from '../charts/evolution.js';
import {renderFan} from '../charts/fan.js';
import {renderHistogram} from '../charts/histogram.js';
import {renderStacked} from '../charts/stacked.js';
import {coalitionHash, coalitionSeats, column, parseCoalition, summarize} from '../seats.js';
import {mapNames, mountWhenReady, readInitial, wireControls, wireForm} from './common.js';

const EMPTY_COALITION = 'Marca partidos para calcular su mayoría.';

/**
 * Wire the coalition calculator: restore the selection of the URL fragment, then recompute the summary
 * and the histogram of the marked parties on every change, keeping the fragment in sync.
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
        `intervalo 95 % ${fmtRange(stats.lo, stats.hi, 0)}`,
        `mayoría absoluta (${fmtInt(majority)}): ${fmtProb(stats.pMajority)}`,
      ].join(' · ');
      renderHistogram(chart, values, {
        majority, median: stats.median, lo: stats.lo, hi: stats.hi, color: THEME.markColor,
      });
    }
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + coalitionHash(names));
  }

  fieldset.addEventListener('change', update);
  update();
}

/** Draw the histograms, the blocks bar, the fan and the evolution, and wire the coalition calculator. */
function paint() {
  const {state, meta, summary, dist, fan, runs} = readInitial();

  const catalog = new Catalog(meta);
  const majority = meta.majority ?? summary.majority;
  const nSeats = meta.n_seats ?? summary.n_seats;
  const partyRows = summary.parties || [];
  const fanNames = [...new Set((fan.rows || []).map((row) => row.name))];
  const names = [...new Set([...dist.parties, ...partyRows.map((row) => row.name), ...fanNames])];
  const colors = mapNames(names, (name) => catalog.color(name));
  const fullnames = mapNames(names, (name) => catalog.fullname(name));

  for (const el of document.querySelectorAll('#histograms div[data-party]')) {
    const name = el.dataset.party;
    const values = column(dist, name);
    if (!values.length) {
      continue;
    }
    const stats = summarize(values, majority);
    renderHistogram(el, values, {
      color: colors[name], majority, median: stats.median, lo: stats.lo, hi: stats.hi,
    });
  }

  const blocks = new Map((summary.blocks || []).map((row) => [row.name, row]));
  const blockRows = catalog.orderBlocks([...blocks.keys()])
    .map((name) => ({name, seats: blocks.get(name).seats ?? 0}));
  renderStacked(document.getElementById('blocks-chart'), blockRows, {
    colors: mapNames([...blocks.keys()], (name) => catalog.blockColor(name)),
    majority,
    total: nSeats,
  });

  wireCoalition(dist, majority);

  const main = (catalog.bmaps.main || []).filter((name) => fanNames.includes(name));
  renderFan(document.getElementById('fan-chart'), fan, {
    parties: fanNames,
    selected: main.length ? main : null,
    colors,
    fullnames,
    markAt: state.mode === 'nowcast' ? 0 : meta.horizon_max,
    markLabel: state.mode === 'nowcast' ? 'Hoy' : 'Elección',
  });

  renderEvolution(document.getElementById('seats-evolution'), runs.runs, state.mode, {
    colors, fullnames, field: 'seats', formatter: fmtInt, parties: partyRows.map((row) => row.name),
  });
}

wireControls();
wireForm('district-form');
mountWhenReady(paint);
