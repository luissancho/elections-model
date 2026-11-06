// The bullseye of the last polls by election with its scope and years filters, shared by the pollster profile and
// the model in the backtest page: the chart (`#bullseye-chart`), the filters (`#bullseye-filter`) and the table
// (`#bullseye-table`), redrawn from the rows that pass them. No hash: the filters are the only state. The download
// button of the card (`#bullseye-download`, the profile) draws the bullseye as filtered, the filters and the count
// in the subtitle and the file name. A profile also has, in its own card, the house against the mean of the other
// houses in the same election in the gap between the two blocks (`wireConsensus`: `#consensus-chart`, the `gap` and
// `others` of the same rows, with the same filters in its own form `#consensus-filter`, independent of the bullseye's,
// and its own download button `#consensus-download`).
import {fmtDate, fmtNum, shadeAround} from '../format.js';
import {fadeByDate, filterRows, imageTexts, yearsText} from '../errors.js';
import {mountChart, THEME} from '../charts/base.js';
import {bullseyeOption, renderBullseye} from '../charts/bullseye.js';
import {scatterOption} from '../charts/scatter.js';
import {IMAGE, wireDownload} from '../share.js';
import {showError, tryDraw} from './common.js';

/** Height of the chart in the image of a bullseye: as wide as the chart box, so the rings stay round. */
export const BULLSEYE_IMAGE_HEIGHT = IMAGE.width - 2 * IMAGE.pad;

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
// The axes of the house against the others: symmetric, the first multiple of this step beyond the farthest point.
const CONSENSUS_STEP = 5;

/**
 * Whether a bullseye row has a point in the chart of the house against the others: its error in the gap between the
 * blocks (`gap`) and the mean of the other houses (`others`).
 *
 * @param {object} row a bullseye row
 * @returns {boolean} both numbers present
 */
const hasConsensus = (row) => Number.isFinite(row.others) && Number.isFinite(row.gap);

/**
 * Option of the house against the mean of the other houses in the gap between the two blocks (the error in the left
 * block minus the error in the right one, the signed `error_blocks`): one point per election with a mean of the
 * others (`others`) and an error of the house (`gap`), named `event-scope`, the general elections in the accent and
 * the regional ones in the contrasting tone, over a dashed diagonal between symmetric axes (`CONSENSUS_STEP`); both
 * axes named, so the downloaded image reads on its own.
 *
 * @param {object[]} rows bullseye rows (`gap`, `others`, `scope` and `event`)
 * @param {string} left the left block (`initial.left`)
 * @param {string} right the right block (`initial.right`)
 * @param {string} house the name of the house (the y axis in the tooltip)
 * @returns {object} the ECharts option
 */
export function consensusOption(rows, left, right, house) {
  const points = rows.filter(hasConsensus).map((row) => ({...row, label: `${row.event}-${row.scope}`}));
  const reach = Math.max(0, ...points.flatMap((row) => [Math.abs(row.others), Math.abs(row.gap)]));
  const measure = `error en ${left} − ${right}`;
  const limit = Math.max(CONSENSUS_STEP, Math.ceil(reach / CONSENSUS_STEP) * CONSENSUS_STEP);
  return scatterOption(points, {
    x: 'others',
    y: 'gap',
    name: 'label',
    labelled: points.map((row) => row.label),
    diagonal: true,
    axes: {xMin: -limit, xMax: limit, yMin: -limit, yMax: limit, interval: CONSENSUS_STEP},
    colorOf: (row) => (row.scope === NATIONAL_SCOPE ? THEME.accentColor : REGIONAL_COLOR),
    xName: `Media de las demás casas, ${measure}`,
    yName: `${house}, ${measure}`,
    xLabel: 'demás casas',
    yLabel: house,
  });
}

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
 * Label of the option chosen in `select`, or null when its value is empty (the open «Todos»/«Todas»).
 *
 * @param {HTMLSelectElement|undefined} select the select
 * @returns {string|null} the label
 */
export function chosenLabel(select) {
  const option = select && select.selectedOptions && select.selectedOptions[0];
  return option && option.value !== '' ? option.textContent.trim() : null;
}

/**
 * Wire a scope and years filter form (`{prefix}-filter`, the `errors_filter` macro): call `draw` with the rows that
 * pass the filters chosen, their labels (`scope`, the name of the scope chosen or null; `from` and `to`, the years
 * chosen or null for an open extreme), on every change. The form only works with JavaScript, so the server renders
 * it hidden and this shows it. Nothing without the form.
 *
 * @param {HTMLFormElement|null} form the form
 * @param {object[]} rows the rows (`scope` and `year`)
 * @param {function({rows: object[], scope: (string|null), from: (number|null), to: (number|null)}): void} draw
 */
function wireFilters(form, rows, draw) {
  if (!form) {
    return;
  }
  form.addEventListener('submit', (event) => event.preventDefault());
  // The years offered run from the first election of the house to today (the server fills both selects with the
  // same list); an extreme chosen is an open bound.
  const {ambito, desde, hasta} = form.elements;
  const first = Number(desde.options[0]?.value);
  const last = Number(hasta.options[hasta.options.length - 1]?.value);
  form.addEventListener('change', () => {
    const filters = {
      scope: ambito.value || null,
      from: Number(desde.value) === first ? null : Number(desde.value),
      to: Number(hasta.value) === last ? null : Number(hasta.value),
    };
    draw({rows: filterRows(rows, filters), scope: chosenLabel(ambito), from: filters.from, to: filters.to});
  });
  form.hidden = false;
}

/**
 * Text of the count of elections drawn («1 elección», «4 elecciones»).
 *
 * @param {number} n the elections
 * @returns {string} the text
 */
const electionsText = (n) => `${n} ${n === 1 ? 'elección' : 'elecciones'}`;

/**
 * Bullseye with its filters (a pollster profile, the model in the backtest page): draw it and its table from the rows that pass the chosen scope and
 * years; the form only works with JavaScript, so the server renders it hidden and this shows it. The download
 * button, when the card has one, exports the bullseye as filtered.
 *
 * @param {object[]} bullseye the bullseye rows (`scope`, `year` and the block errors)
 * @param {string[]} blocks the two blocks of the axes
 */
export function wireBullseye(bullseye, blocks) {
  const target = document.getElementById('bullseye-chart');
  if (!target || !bullseye.length || blocks.length < 2) {
    return;
  }
  const table = document.getElementById('bullseye-table');
  const count = document.getElementById('bullseye-count');
  let current = {rows: bullseye, scope: null, from: null, to: null};  // what is drawn, what the button exports
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
      count.textContent = electionsText(rows.length);
    }
  };
  wireFilters(document.getElementById('bullseye-filter'), bullseye, (chosen) => {
    current = chosen;
    draw(current.rows);
  });
  draw(bullseye);
  wireDownload(document.getElementById('bullseye-download'), () => bullseyeOption(current.rows, options), showError, {
    chartHeight: BULLSEYE_IMAGE_HEIGHT,
    texts: (dataset) => imageTexts(dataset, [current.scope, yearsText(current.from, current.to)], count ? count.textContent : null),
  });
}

/**
 * The house against the mean of the other houses, in its own card of the profile: draw `#consensus-chart` from the
 * bullseye rows that pass its own scope and years filters (`#consensus-filter`, the same as the bullseye's but
 * independent of them), with the count of elections drawn (`#consensus-count`), and wire its download button
 * (`#consensus-download`, the chart as filtered, the filters and the count in the subtitle and the file name).
 * Nothing without the chart, the rows or the pair of blocks.
 *
 * @param {object[]} bullseye the bullseye rows (`gap`, `others`, `scope`, `year` and `event`)
 * @param {string|null} left the left block of the pair (`initial.left`)
 * @param {string|null} right the right block of the pair (`initial.right`)
 */
export function wireConsensus(bullseye, left, right) {
  const target = document.getElementById('consensus-chart');
  if (!target || !bullseye.length || !left || !right) {
    return;
  }
  const house = target.dataset.house || '';
  const count = document.getElementById('consensus-count');
  let current = {rows: bullseye, scope: null, from: null, to: null};  // what is drawn, what the button exports
  const option = () => consensusOption(current.rows, left, right, house);
  const drawnText = () => electionsText(current.rows.filter(hasConsensus).length);
  const draw = () => {
    tryDraw(() => mountChart(target, option()), 'the house against the others');
    if (count) {
      count.textContent = drawnText();
    }
  };
  wireFilters(document.getElementById('consensus-filter'), bullseye, (chosen) => {
    current = chosen;
    draw();
  });
  draw();
  wireDownload(document.getElementById('consensus-download'), option, showError, {
    chartHeight: BULLSEYE_IMAGE_HEIGHT,
    texts: (dataset) => imageTexts(dataset, [current.scope, yearsText(current.from, current.to)], drawnText()),
  });
}
