// Signed errors of the polls (the hidden-vote study): a bar per election in chronological order, and a strip of
// points per party with its mean. Blue is a positive error (the polls overstate the party), red a negative one.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

const NEGATIVE_COLOR = '#d62728';
const POSITIVE_COLOR = '#1d4f86';
const MEAN_COLOR = '#e8912b';

/**
 * A number with its sign, the minus being the typographic one (U+2212); a value that rounds to zero has no sign.
 *
 * @param {number} x the value
 * @param {number} [digits=0] decimals
 * @returns {string} the text
 */
function signed(x, digits = 0) {
  const text = fmtNum(Math.abs(x), digits);
  if (!/[1-9]/.test(text)) {
    return text;
  }
  return `${x < 0 ? '−' : '+'}${text}`;
}

/** The colour of a signed error. */
function signColor(value) {
  return value < 0 ? NEGATIVE_COLOR : POSITIVE_COLOR;
}

/**
 * ECharts option of the bars of the error by election: a category axis of the short names (the general elections in
 * bold), a value axis symmetric around 0 and one bar per election coloured by sign, with the mean as a dashed line.
 *
 * @param {object[]} rows `{short, label, value, general}` per election, in the order to draw
 * @param {object} opts `mean` (the mean line, none when `null`), `meanLabel` (its name), `yMin` (the least half-range
 *   of the value axis), `step` (the tick interval), `boldOf` (which rows have a bold label), `rotate` (degrees of the
 *   labels) and `dim` (opacity of the rows that are not bold)
 * @returns {object} the option
 */
export function errorBarsOption(rows = [], {mean = null, meanLabel = 'media', yMin = 8, step = 2,
                                            boldOf = (row) => Boolean(row.general), rotate = 55, dim = 0.55} = {}) {
  const largest = Math.max(0, ...rows.map((row) => Math.abs(row.value)));
  const yMax = Math.max(yMin, Math.ceil(largest / step) * step);
  const size = rotate >= 90 ? 9 : 10;
  const bold = new Set(rows.filter(boldOf).map((row) => row.short));
  const series = {
    type: 'bar',
    barMaxWidth: 18,
    data: rows.map((row) => ({
      value: row.value,
      itemStyle: {color: signColor(row.value), opacity: boldOf(row) ? 1 : dim},
    })),
  };
  if (mean !== null && mean !== undefined) {
    series.markLine = {
      silent: true,
      symbol: 'none',
      data: [{yAxis: mean}],
      lineStyle: {color: MEAN_COLOR, width: 1.5, type: 'dashed'},
      label: {formatter: `${meanLabel} ${signed(mean, 1)}`, position: 'insideEndTop', fontSize: 10.5, color: MEAN_COLOR},
    };
  }
  return {
    ...baseOption(),
    grid: {left: 40, right: 16, top: 16, bottom: rotate >= 90 ? 64 : 52},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter(params) {
        const item = params[0];
        return `${escapeHtml(rows[item.dataIndex].label)}: ${escapeHtml(signed(item.value, 1))}`;
      },
    },
    xAxis: {
      type: 'category',
      data: rows.map((row) => row.short),
      axisTick: {show: false},
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: {
        ...axisLabelStyle(),
        fontSize: size,
        rotate,
        interval: () => true,
        formatter: (value) => (bold.has(value) ? `{bold|${value}}` : value),
        rich: {bold: {color: THEME.textColor, fontSize: size, fontWeight: 700}},
      },
    },
    yAxis: {
      type: 'value',
      min: -yMax,
      max: yMax,
      interval: step,
      axisLabel: {...axisLabelStyle(), formatter: (v) => signed(v, 0)},
      splitLine: {lineStyle: {color: THEME.gridColor}},
    },
    series: [series],
  };
}

/**
 * Render the bars (`errorBarsOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows see `errorBarsOption`
 * @param {object} opts see `errorBarsOption`
 * @returns {object} the ECharts instance
 */
export function renderErrorBars(el, rows, opts) {
  return mountChart(el, errorBarsOption(rows, opts));
}

/**
 * ECharts option of the strip of errors by party: a category axis of the parties (bottom up as `order` gives them),
 * a value axis symmetric around 0 wide enough for every point (ECharts clips the points outside the grid), a
 * translucent point per election and a bar-shaped mark at the mean of each party.
 *
 * @param {{rows: object[], means: object, order: string[]}} data `rows` as `{party, label, value}`, `means` by party
 * @param {object} opts `xMin` (the least half-range of the value axis, widened to the largest error rounded up to a
 *   tick, as in the bars) and `step` (its tick interval)
 * @returns {object} the option
 */
export function errorStripOption({rows = [], means = {}, order = []} = {}, {xMin = 8, step = 2} = {}) {
  const largest = Math.max(0, ...rows.map((row) => Math.abs(row.value)));
  const xMax = Math.max(xMin, Math.ceil(largest / step) * step);
  const meanData = order
    .filter((label) => typeof means[label] === 'number')
    .map((label) => ({value: [means[label], label]}));
  return {
    ...baseOption(),
    grid: {left: 84, right: 20, top: 12, bottom: 28},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(item) {
        const [value, party] = item.value;
        const what = item.seriesIndex === 1 ? 'media' : item.data.label;
        return `${escapeHtml(what)} · ${escapeHtml(party)}: ${escapeHtml(signed(value, 1))}`;
      },
    },
    xAxis: {
      type: 'value',
      min: -xMax,
      max: xMax,
      interval: step,
      axisLabel: {...axisLabelStyle(), formatter: (v) => signed(v, 0)},
      splitLine: {lineStyle: {color: THEME.gridColor}},
    },
    yAxis: {
      type: 'category',
      data: order,
      axisTick: {show: false},
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: {...axisLabelStyle(), fontSize: 11.5, color: THEME.textColor},
    },
    series: [
      {
        type: 'scatter',
        symbolSize: 9,
        data: rows.map((row) => ({
          value: [row.value, row.party],
          label: row.label,
          itemStyle: {color: signColor(row.value), opacity: 0.5, borderColor: '#fff', borderWidth: 1},
        })),
        markLine: {
          silent: true,
          symbol: 'none',
          data: [{xAxis: 0}],
          label: {show: false},
          lineStyle: {color: '#000', width: 1, type: 'solid'},
        },
      },
      {
        type: 'scatter',
        symbol: 'rect',
        symbolSize: [4, 22],
        z: 5,
        itemStyle: {color: MEAN_COLOR},
        data: meanData,
      },
    ],
  };
}

/**
 * Render the strip (`errorStripOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object} data see `errorStripOption`
 * @param {object} opts see `errorStripOption`
 * @returns {object} the ECharts instance
 */
export function renderErrorStrip(el, data, opts) {
  return mountChart(el, errorStripOption(data, opts));
}
