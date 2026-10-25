// Matrix of signs: one row per pollster, one column per election, each cell shaded by the error of the poll on
// the left block (red when the poll gave the left more than the result, blue when less; the CIS bias study).
import {baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

const ROW_HEIGHT = 26;
const EXTRA_HEIGHT = 70;
const MIN_ALPHA = 0.15;
const ALPHA_SPAN = 0.75;
const LABEL_SIZE = 9.5;

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
 * @param {object} opts `cap` (error, in points, of the darkest shade), `positive` and `negative` (hex colours)
 * @returns {object} the option
 */
export function matrixOption({events, houses, cells}, {cap = 8, positive = '#d62728', negative = '#1d4f86'} = {}) {
  const rgb = [channels(positive), channels(negative)];
  return {
    ...baseOption(),
    grid: {left: 8, right: 8, top: 8, bottom: 8, containLabel: true},
    tooltip: {
      ...tooltipStyle(),
      formatter(params) {
        const [x, y, value] = params.value;
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
        rotate: 90,
        interval: 0,
        fontSize: LABEL_SIZE,
        color: THEME.mutedColor,
        formatter: (text) => (text.startsWith('GEN') ? `{g|${text}}` : text),
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
    series: [{
      type: 'heatmap',
      data: cells.map((cell) => ({
        value: cell,
        itemStyle: {color: shade(cell[2], cap, rgb), borderColor: '#fff', borderWidth: 2},
      })),
      emphasis: {itemStyle: {borderColor: THEME.textColor, borderWidth: 1.5}},
    }],
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
