// Matrix of signs: one row per pollster, one column per election, each cell shaded by the error of the poll on
// the left block (red when the poll gave the left more than the result, blue when less; the CIS bias study). With a
// `center` it shades a ratio around it instead (the herd study: red below 1, blue above, grey near it).
import {baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

const ROW_HEIGHT = 26;
const EXTRA_HEIGHT = 70;
const MIN_ALPHA = 0.15;
const ALPHA_SPAN = 0.75;
const LABEL_SIZE = 9.5;
// The cells near the centre, and the opacity from which the label inside a cell is written in white.
const NEUTRAL_COLOR = '#f4f5f8';
const WHITE_LABEL_ALPHA = 0.6;

/**
 * Colour of a cell: the positive or the negative one, more opaque the larger the error is (up to `cap`).
 *
 * @param {number} value error in points
 * @param {number} cap error from which the colour stops growing
 * @param {string[]} rgb `[positive, negative]` as `'r,g,b'`
 * @returns {string} `rgba(...)` colour
 */
function shade(value, cap, [positive, negative]) {
  const alpha = MIN_ALPHA + ALPHA_SPAN * Math.min(Math.abs(value), cap) / cap;
  return `rgba(${value > 0 ? positive : negative},${alpha.toFixed(2)})`;
}

/**
 * Opacity of a cell shaded around `center`: grows with the distance to it, relative to the cap of its side.
 *
 * @param {number} value the ratio
 * @param {number} center the value of no colour
 * @param {{below: number, above: number}} caps distance, on each side, of the darkest shade
 * @returns {number} the opacity
 */
function alphaAround(value, center, caps) {
  const reach = value < center ? (center - value) / caps.below : (value - center) / caps.above;
  return MIN_ALPHA + ALPHA_SPAN * Math.min(reach, 1);
}

/**
 * Whether a column is a general election (its short name starts with `GEN`): the columns in bold by default.
 *
 * @param {{short: string}} event the column
 * @returns {boolean} whether it is
 */
function isGeneral(event) {
  return String(event.short).startsWith('GEN');
}

/**
 * `#rrggbb` as `'r,g,b'`.
 *
 * @param {string} hex colour
 * @returns {string} the channels
 */
function channels(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(',');
}

/**
 * Number with one decimal and the sign always written (the minus typographic, as `fmtNum` does).
 *
 * @param {number} value number
 * @returns {string} text
 */
function signed(value) {
  return `${value > 0 ? '+' : ''}${fmtNum(value, 1)}`;
}

/**
 * ECharts option of the matrix: a heat map without scale, white borders between cells, the elections on top
 * (rotated, the general ones in bold) and the pollsters down the side.
 *
 * @param {{events: object[], houses: string[], cells: number[][]}} data `events` as `{label, short}`, `cells` as
 *   `[event index, house index, error]`
 * @param {object} opts `cap` (error, in points, of the darkest shade), `positive` and `negative` (hex colours);
 *   with `center` the cells are shaded around it instead: `positive` below with the darkest shade at
 *   `caps.below` from it, `negative` above at `caps.above`, and grey inside `neutral` (`[lo, hi]`); `label` writes
 *   the value inside each cell and `cellText(house, event, cell)` gives the text of the tooltip, `cell` being the
 *   whole value array of the cell (`[event index, house index, value, ...]`, with the extra items the data carries);
 *   `bold(event)` says which columns are named in bold (by default the general elections, `GEN …`) and `rotate`
 *   turns their names
 * @returns {object} the option
 */
export function matrixOption({events, houses, cells}, {
  cap = 8, positive = '#d62728', negative = '#1d4f86', center = null, caps = {below: 0.5, above: 1},
  neutral = [0.9, 1.2], label = false, cellText = null, bold = isGeneral, rotate = 90,
} = {}) {
  const rgb = [channels(positive), channels(negative)];
  const centred = center !== null && center !== undefined;
  const item = (cell) => {
    const value = cell[2];
    if (!centred) {
      return {value: cell, itemStyle: {color: shade(value, cap, rgb), borderColor: '#fff', borderWidth: 2}};
    }
    if (value >= neutral[0] && value <= neutral[1]) {
      return {value: cell, itemStyle: {color: NEUTRAL_COLOR, borderColor: '#fff', borderWidth: 2}};
    }
    const alpha = alphaAround(value, center, caps);
    const color = `rgba(${value < center ? rgb[0] : rgb[1]},${alpha.toFixed(2)})`;
    const out = {value: cell, itemStyle: {color, borderColor: '#fff', borderWidth: 2}};
    if (label && alpha >= WHITE_LABEL_ALPHA) {
      out.label = {color: '#fff'};
    }
    return out;
  };
  const series = {
    type: 'heatmap',
    data: cells.map(item),
    emphasis: {itemStyle: {borderColor: THEME.textColor, borderWidth: 1.5}},
  };
  if (label) {
    series.label = {
      show: true,
      fontSize: 10.5,
      color: THEME.textColor,
      formatter: (params) => fmtNum(params.value[2], 2),
    };
  }
  return {
    ...baseOption(),
    grid: {left: 8, right: 8, top: 8, bottom: 8, containLabel: true},
    tooltip: {
      ...tooltipStyle(),
      formatter(params) {
        const [x, y, value] = params.value;
        if (cellText) {
          return escapeHtml(cellText(houses[y], events[x], params.value));
        }
        return `${escapeHtml(houses[y])} · ${escapeHtml(events[x].label)}: `
          + `error en el bloque de izquierdas ${escapeHtml(signed(value))}`;
      },
    },
    xAxis: {
      type: 'category',
      data: events.map((event) => event.short),
      position: 'top',
      axisTick: {show: false},
      axisLine: {show: false},
      axisLabel: {
        rotate,
        interval: 0,
        fontSize: LABEL_SIZE,
        color: THEME.mutedColor,
        formatter: (text, index) => (bold(events[index]) ? `{g|${text}}` : text),
        rich: {g: {fontWeight: 'bold', color: THEME.textColor, fontSize: LABEL_SIZE}},
      },
    },
    yAxis: {
      type: 'category',
      data: houses,
      inverse: true,
      axisTick: {show: false},
      axisLine: {show: false},
      axisLabel: {fontSize: 11.5, color: THEME.textColor, fontWeight: 500},
    },
    series: [series],
  };
}

/**
 * Render the matrix (`matrixOption`) in `el`; the height grows with the number of pollsters.
 *
 * @param {HTMLElement} el chart container
 * @param {object} data see `matrixOption`
 * @param {object} opts see `matrixOption`
 * @returns {object} the ECharts instance
 */
export function renderMatrix(el, data, opts) {
  el.style.height = `${data.houses.length * ROW_HEIGHT + EXTRA_HEIGHT}px`;
  return mountChart(el, matrixOption(data, opts));
}
