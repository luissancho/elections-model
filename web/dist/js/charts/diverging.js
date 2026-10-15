// Diverging horizontal bars: two measures per row drawn from a common zero, one to the left and one to the right
// (the error of the last poll of each house in each of the two blocks, as the notebook's deviation chart).
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {niceMax} from './bars.js';

/**
 * Absolute value of a measure, or null when it is missing.
 *
 * @param {*} value a number or anything else
 * @returns {(number|null)} `|value|`, null unless `value` is a finite number
 */
function magnitude(value) {
  return Number.isFinite(value) ? Math.abs(value) : null;
}

/**
 * ECharts option of the diverging bars: one row per entry of `rows`, in their order from the top, the `left`
 * measure growing leftwards from zero and the `right` one rightwards, both as magnitudes (the sign of the
 * measures is dropped), with the value written at the end of each bar, a legend naming the two sides and a
 * symmetric axis that reads as magnitudes.
 *
 * @param {object[]} rows `{name, <left.key>: number, <right.key>: number}` (a missing measure leaves a gap)
 * @param {object} opts `left` and `right` (`{key, name, color}`: the field, the legend name and the colour of
 *   each side), `formatter` (number → text of the bar labels and the tooltip) and `axisFormatter` (axis ticks,
 *   `formatter` by default)
 * @returns {object} the option
 */
export function divergingOption(rows, {left, right, formatter = String, axisFormatter = formatter}) {
  const data = rows.map((row) => ({name: row.name, left: magnitude(row[left.key]), right: magnitude(row[right.key])}));
  const top = Math.max(0, ...data.flatMap((row) => [row.left ?? 0, row.right ?? 0]));
  const limit = niceMax(top * 1.15);
  const label = (position) => ({
    show: true,
    position,
    formatter: (params) => (params.value === null || params.value === undefined ? '' : formatter(Math.abs(params.value))),
    color: THEME.textColor,
    fontSize: 12,
  });
  const side = (spec, sign, key, position) => ({
    type: 'bar',
    name: spec.name,
    stack: 'sides',
    barWidth: '60%',
    color: spec.color || THEME.barColor,
    data: data.map((row) => (row[key] === null ? null : sign * row[key])),
    label: label(position),
  });
  const leftSeries = side(left, -1, 'left', 'left');
  leftSeries.markLine = {
    silent: true,
    symbol: 'none',
    lineStyle: {color: THEME.intervalColor, width: 1},
    label: {show: false},
    data: [{xAxis: 0}],
  };
  return {
    ...baseOption(),
    grid: {left: 8, right: 16, top: 36, bottom: 8, containLabel: true},
    legend: {top: 0, data: [left.name, right.name]},
    xAxis: {
      type: 'value',
      min: -limit,
      max: limit,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      // The extent is not a round number: its labels would crowd the last round tick.
      axisLabel: {formatter: (x) => axisFormatter(Math.abs(x)), showMinLabel: false, showMaxLabel: false},
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: data.map((row) => row.name),
      axisTick: {show: false},
      axisLine: {onZero: false},
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter(params) {
        const row = data[params[0].dataIndex];
        if (!row) {
          return '';
        }
        const line = (spec, value) => `${escapeHtml(spec.name)}: ${escapeHtml(value === null ? '–' : formatter(value))}`;
        return [`<strong>${escapeHtml(row.name)}</strong>`, line(left, row.left), line(right, row.right)].join('<br>');
      },
    },
    series: [leftSeries, side(right, 1, 'right', 'right')],
  };
}

/**
 * Render the diverging bars (`divergingOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows rows of `divergingOption`
 * @param {object} opts options of `divergingOption`
 * @returns {object} the ECharts instance
 */
export function renderDiverging(el, rows, opts) {
  return mountChart(el, divergingOption(rows, opts));
}
