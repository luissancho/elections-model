// The bullseye of the last polls by election with its scope and years filters, shared by the pollster profile and
// the model in the backtest page: the chart (`#bullseye-chart`), the filters (`#bullseye-filter`) and the table
// (`#bullseye-table`), redrawn from the rows that pass them. No hash: the filters are the only state.
import {fmtDate, fmtNum, shadeAround} from '../format.js';
import {fadeByDate, filterRows} from '../errors.js';
import {THEME} from '../charts/base.js';
import {renderBullseye} from '../charts/bullseye.js';
import {tryDraw} from './common.js';

// The bullseye and the rating path of a profile: general elections in the accent, regional ones in a contrasting
// tone; a point of the bullseye fades with the age of its election down to this opacity.
const NATIONAL_SCOPE = 'es';
export const KIND_GENERAL = 'Generales';
export const KIND_REGIONAL = 'Autonómicas';
export const REGIONAL_COLOR = '#c0672e';
const OLDEST_ALPHA = 0.3;
// The double gradient of the block errors of the bullseye table, the same the server paints (`casa.html`): red the
// more the poll fell short of the block, green the more it overshot it, fullest at this many points.
const SHORT_COLOR = '#d62728';
const OVER_COLOR = '#2ca02c';
const ERROR_SPAN = 8;
// Leading columns of the bullseye table before the block columns: election, date and days in a profile; a table
// with `data-fixed` says how many it has (the model of the backtest page has the election only).
const BULLSEYE_FIXED_COLUMNS = 3;

/**
 * Rebuild the body of the bullseye table with the rows shown. The block columns come from the table header, in its
 * order (the server puts the left block first, which may differ from the order of the axes of the bullseye), and
 * their cells carry the same double gradient the server paints.
 *
 * @param {HTMLTableElement|null} table the `#bullseye-table`
 * @param {object[]} rows bullseye rows (`name`, `event_date`, `days` and the error of each block)
 */
function fillBullseyeTable(table, rows) {
  if (!table) {
    return;
  }
  const fixed = Number(table.dataset.fixed) || BULLSEYE_FIXED_COLUMNS;
  const blocks = [...table.tHead.rows[0].cells].slice(fixed).map((th) => th.textContent.trim());
  const cell = (text) => {
    const td = document.createElement('td');
    td.textContent = text;
    return td;
  };
  const shaded = (value) => {
    const td = cell(fmtNum(value, 1));
    td.className = 'cell-shade';
    td.style.backgroundColor = shadeAround(value, 0, ERROR_SPAN, SHORT_COLOR, OVER_COLOR);
    return td;
  };
  table.tBodies[0].replaceChildren(...rows.map((row) => {
    const tr = document.createElement('tr');
    tr.append(cell(row.name || row.event));
    if (fixed === BULLSEYE_FIXED_COLUMNS) {
      tr.append(cell(fmtDate(row.event_date)), cell(fmtNum(row.days)));
    }
    tr.append(...blocks.map((block) => shaded(row[block])));
    return tr;
  }));
}

/**
 * Bullseye with its filters (a pollster profile, the model in the backtest page): draw it and its table from the rows that pass the chosen scope and
 * years; the form only works with JavaScript, so the server renders it hidden and this shows it.
 *
 * @param {object[]} bullseye the bullseye rows (`scope`, `year` and the block errors)
 * @param {string[]} blocks the two blocks of the axes
 */
export function wireBullseye(bullseye, blocks) {
  const target = document.getElementById('bullseye-chart');
  if (!target || !bullseye.length || blocks.length < 2) {
    return;
  }
  const form = document.getElementById('bullseye-filter');
  const table = document.getElementById('bullseye-table');
  const count = document.getElementById('bullseye-count');
  // The fading is fixed over every election of the house, so a filter does not change the tone of a point.
  const alphas = new Map(fadeByDate(bullseye, {min: OLDEST_ALPHA}).map((alpha, i) => [bullseye[i], alpha]));
  const options = {
    blocks,
    groupOf: (row) => (row.scope === NATIONAL_SCOPE ? KIND_GENERAL : KIND_REGIONAL),
    colors: {[KIND_GENERAL]: THEME.accentColor, [KIND_REGIONAL]: REGIONAL_COLOR},
    legend: true,
    labelOf: (row) => `${row.event}-${row.scope}`,
    alphaOf: (row) => alphas.get(row) ?? 1,
    onTop: (group) => group === KIND_GENERAL,  // the general elections over the regional ones
  };
  const draw = (rows) => {
    tryDraw(() => renderBullseye(target, rows, options), 'the bullseye');
    fillBullseyeTable(table, rows);
    if (count) {
      count.textContent = `${rows.length} ${rows.length === 1 ? 'elección' : 'elecciones'}`;
    }
  };
  if (form) {
    form.addEventListener('submit', (event) => event.preventDefault());
    // The years offered run from the first election of the house to today (the server fills both selects with the
    // same list); an extreme chosen is an open bound.
    const {desde, hasta} = form.elements;
    const first = Number(desde.options[0]?.value);
    const last = Number(hasta.options[hasta.options.length - 1]?.value);
    form.addEventListener('change', () => {
      const {ambito} = form.elements;
      draw(filterRows(bullseye, {
        scope: ambito.value || null,
        from: Number(desde.value) === first ? null : Number(desde.value),
        to: Number(hasta.value) === last ? null : Number(hasta.value),
      }));
    });
    form.hidden = false;
  }
  draw(bullseye);
}
