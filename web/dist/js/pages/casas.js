// Houses page and pollster profile: rating bars, house-effects heat map, rating path and bullseye, drawn
// from the data embedded by the server. The bullseye of a profile has a scope and a years filter (no hash: the
// profile has no other state), which redraws it and its table.
import {fmtDate, fmtNum} from '../format.js';
import {fadeByDate, filterRows, yearsOf} from '../errors.js';
import {renderBars} from '../charts/bars.js';
import {THEME} from '../charts/base.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderHeatmap} from '../charts/heatmap.js';
import {renderLine} from '../charts/line.js';
import {mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

// Height of the rating bars: one row per rated house plus the axes.
const BAR_ROW_HEIGHT = 22;
const BAR_EXTRA_HEIGHT = 60;

// The bullseye of a profile: general elections in the accent, regional ones in a contrasting tone; a point fades
// with the age of its election down to this opacity.
const NATIONAL_SCOPE = 'es';
const KIND_GENERAL = 'Generales';
const KIND_REGIONAL = 'Autonómicas';
const REGIONAL_COLOR = '#c0672e';
const OLDEST_ALPHA = 0.3;

/**
 * Rebuild the body of the bullseye table with the rows shown.
 *
 * @param {HTMLTableElement|null} table the `#bullseye-table`
 * @param {object[]} rows bullseye rows (`name`, `event_date`, `days` and the error of each block)
 * @param {string[]} blocks the block columns
 */
function fillBullseyeTable(table, rows, blocks) {
  if (!table) {
    return;
  }
  const cell = (text) => {
    const td = document.createElement('td');
    td.textContent = text;
    return td;
  };
  table.tBodies[0].replaceChildren(...rows.map((row) => {
    const tr = document.createElement('tr');
    tr.append(cell(row.name || row.event), cell(fmtDate(row.event_date)), cell(fmtNum(row.days)),
      ...blocks.map((block) => cell(fmtNum(row[block], 1))));
    return tr;
  }));
}

/**
 * Bullseye of a profile with its filters: draw it and its table from the rows that pass the chosen scope and
 * years; the form only works with JavaScript, so the server renders it hidden and this shows it.
 *
 * @param {object[]} bullseye the profile's bullseye rows (`scope`, `year` and the block errors)
 * @param {string[]} blocks the two blocks of the axes
 */
function wireBullseye(bullseye, blocks) {
  const target = document.getElementById('bullseye-chart');
  if (!target || !bullseye.length || blocks.length < 2) {
    return;
  }
  const form = document.getElementById('bullseye-filter');
  const table = document.getElementById('bullseye-table');
  const count = document.getElementById('bullseye-count');
  const years = yearsOf(bullseye);
  // The fading is fixed over every election of the house, so a filter does not change the tone of a point.
  const alphas = new Map(fadeByDate(bullseye, {min: OLDEST_ALPHA}).map((alpha, i) => [bullseye[i], alpha]));
  const options = {
    blocks,
    groupOf: (row) => (row.scope === NATIONAL_SCOPE ? KIND_GENERAL : KIND_REGIONAL),
    colors: {[KIND_GENERAL]: THEME.accentColor, [KIND_REGIONAL]: REGIONAL_COLOR},
    legend: true,
    labelOf: (row) => `${row.event}-${row.scope}`,
    alphaOf: (row) => alphas.get(row) ?? 1,
  };
  const draw = (rows) => {
    tryDraw(() => renderBullseye(target, rows, options), 'the bullseye');
    fillBullseyeTable(table, rows, blocks);
    if (count) {
      count.textContent = `${rows.length} ${rows.length === 1 ? 'elección' : 'elecciones'}`;
    }
  };
  if (form) {
    form.addEventListener('submit', (event) => event.preventDefault());
    form.addEventListener('change', () => {
      const {ambito, desde, hasta} = form.elements;
      draw(filterRows(bullseye, {
        scope: ambito.value || null,
        from: Number(desde.value) === years[0] ? null : Number(desde.value),
        to: Number(hasta.value) === years[years.length - 1] ? null : Number(hasta.value),
      }));
    });
    form.hidden = false;
  }
  draw(bullseye);
}

/** Draw the charts whose container is present and has data. */
function paint() {
  const {ranking = [], grid = null, path = [], bullseye = [], blocks = []} = readInitial();

  const bars = document.getElementById('rating-chart');
  if (bars && ranking.length) {
    bars.style.height = `${ranking.length * BAR_ROW_HEIGHT + BAR_EXTRA_HEIGHT}px`;
    tryDraw(() => renderBars(bars, ranking.map((row) => ({name: row.pollster, rating: row.rating})), {
      value: 'rating', lo: 'lo', hi: 'hi', formatter: (x) => fmtNum(x, 1),
    }), 'the rating bars');
  }

  const heatmap = document.getElementById('effects-heatmap');
  if (heatmap && grid && grid.rows.length) {
    tryDraw(() => renderHeatmap(heatmap, grid), 'the house effects');
  }

  const pathEl = document.getElementById('rating-path');
  const points = path.filter((row) => row.rating !== null && row.rating !== undefined)
    .map((row) => [row.event_date, row.rating]);
  if (pathEl && points.length) {
    tryDraw(() => renderLine(pathEl, points, {formatter: (x) => fmtNum(x, 1), name: 'Rating'}), 'the rating path');
  }

  wireBullseye(bullseye, blocks);
}

wireControls();
mountWhenReady(paint);
