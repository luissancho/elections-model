// The page of the CIS bias study (`/analisis/sesgo-cis`): the charts of its questions, drawn from the data embedded
// by the server (`initial`, `island` of `mtpy/lib/pages/lean.py`), and the «Ver las N elecciones» button of its table.
import {THEME} from '../charts/base.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderDistance} from '../charts/distance.js';
import {renderMatrix} from '../charts/matrix.js';
import {renderRanking} from '../charts/ranking.js';
import {renderScatter} from '../charts/scatter.js';
import {mountWhenReady, readInitial, tryDraw, wireMore} from './common.js';

// The colour of the CIS and the one of everything that is not national (regional bullseye and scatter points).
const RED = '#ce3b3b';
const REGIONAL_COLOR = '#c0672e';
// How many houses are named at each end of each axis in the bullseye of the last general election.
const NAMED_AT_EACH_END = 2;

/** The container `#id` when it exists and there is something to draw in it. */
function container(id, ready) {
  const el = document.getElementById(id);
  return el && ready ? el : null;
}

/** The group of a row of the two period bullseyes: the general or the regional elections. */
function periodGroup(row) {
  return row.scope === 'es' ? 'generales' : 'autonómicas';
}

/** Question 1: the bullseye of the CIS errors without and with its cycle, both on the same kind of axes. */
function drawPeriods(bull) {
  const blocks = bull.blocks || [];
  for (const [id, rows] of [['bull-without', bull.without], ['bull-with', bull.with]]) {
    const el = container(id, rows && rows.length);
    if (el) {
      tryDraw(() => renderBullseye(el, rows, {
        blocks,
        groupOf: periodGroup,
        colors: {generales: THEME.accentColor, autonómicas: REGIONAL_COLOR},
        legend: true,
        labels: true,
        labelOf: (row) => (row.scope === 'es' ? row.event_date.slice(0, 4) : ''),
        onTop: (group) => group === 'generales',
        interval: 5,
      }), 'the bullseye of the CIS errors');
    }
  }
}

/** The sign matrix of the houses by election. */
function drawMatrix(matrix) {
  const el = container('matrix-chart', matrix && matrix.houses && matrix.houses.length);
  if (el) {
    tryDraw(() => renderMatrix(el, matrix), 'the matrix of signs');
  }
}

/** The CIS error against the error of the other houses, one point per election, the national ones named. */
function drawConsensus(rows) {
  const el = container('consensus-chart', rows.length);
  if (el) {
    tryDraw(() => renderScatter(el, rows, {
      x: 'x',
      y: 'y',
      name: 'label',
      labelled: rows.filter((row) => row.national).map((row) => row.label),
      diagonal: true,
      axes: {xMin: -10, xMax: 15, yMin: -5, yMax: 15, interval: 5},
      colorOf: (row) => (row.national ? THEME.accentColor : REGIONAL_COLOR),
      xName: 'Media de las demás casas, error en la izquierda',
      xLabel: 'demás casas',
      yLabel: 'CIS',
    }), 'the CIS against the consensus');
  }
}

/** The ranking of the houses by how often and how much they overestimate the left. */
function drawRanking(rows) {
  const el = container('ranking-chart', rows.length);
  if (el) {
    tryDraw(() => renderRanking(el, rows), 'the ranking of houses');
  }
}

/** The names of the houses at the ends of both axes of the bullseye (the largest absolute error in each block). */
function extremeNames(rows, blocks) {
  const names = new Set();
  for (const block of blocks) {
    const sorted = [...rows].sort((a, b) => Math.abs(b[block]) - Math.abs(a[block]));
    sorted.slice(0, NAMED_AT_EACH_END).forEach((row) => names.add(row.name));
  }
  return names;
}

/** The last general election: every house, with the CIS and the consensus highlighted. */
function drawElection(election) {
  const houses = election.houses || [];
  const el = container('bull-2023', houses.length);
  if (!el) {
    return;
  }
  const blocks = election.blocks || [];
  const cis = houses.find((row) => row.cis);
  const others = houses.filter((row) => !row.cis);
  const named = extremeNames(others, blocks);
  const highlights = [
    cis && {row: cis, color: RED, label: 'CIS'},
    election.consensus && {row: election.consensus, color: THEME.textColor, symbol: 'diamond', label: 'Consenso'},
  ].filter(Boolean);
  tryDraw(() => renderBullseye(el, others, {
    blocks,
    color: THEME.accentColor,
    interval: 2,
    labelOf: (row) => (named.has(row.name) ? row.name : ''),
    highlights,
  }), 'the bullseye of the last general election');
}

/** The distance of every house to the result of the last general election. */
function drawDistance(rows) {
  const el = container('distance-chart', rows.length);
  if (el) {
    tryDraw(() => renderDistance(el, rows), 'the distance to the result');
  }
}

/** Draw the charts whose container is present. */
function paint() {
  const data = readInitial();
  drawPeriods(data.bull || {});
  drawMatrix(data.matrix);
  drawConsensus(data.consensus || []);
  drawRanking(data.ranking || []);
  drawElection(data.election || {});
  drawDistance(data.distance || []);
}

wireMore('elections-more', 'elections-table');
mountWhenReady(paint);
