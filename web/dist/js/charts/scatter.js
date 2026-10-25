// A cloud of points, one per row, with only some of them named and an optional fitted line (the CERA study).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

// The colour of the fitted line, the red of the study's charts.
const FIT_COLOR = '#ce3b3b';

/**
 * ECharts option of the cloud: the points in the site accent, the name written beside those in `labelled` only
 * (to the left of the points in the right quarter of the range, so it never leaves the chart), and a dashed red
 * line between the two points of `fit` when given.
 *
 * @param {object[]} rows one per point
 * @param {object} opts `x`, `y` and `name` (field names of the coordinates and of the name), `labelled` (names
 *   to write beside their point), `axes` (`{xMin, xMax, yMin, yMax, interval}`: fixed limits and tick step of
 *   both axes instead of fitting the data), `diagonal` (a dashed line from `[min, min]` to `[max, max]` of
 *   those axes; needs `axes`), `colorOf` (`row => colour` of each point), `fit` (`[[x0, y0], [x1, y1]]`: the ends of the fitted line, or null),
 *   `xFormatter` and `yFormatter` (number → text of each axis and of the tooltip), `xName` (title of the x axis)
 *   and `xLabel` and `yLabel` (short names of the coordinates in the tooltip)
 * @returns {object} the option
 */
export function scatterOption(rows, {
  x, y, name, labelled = [], fit = null, xFormatter = (v) => fmtNum(v, 1), yFormatter = (v) => fmtNum(v, 1),
  xName = '', xLabel = '', yLabel = '', diagonal = false, axes = null, colorOf = null,
} = {}) {
  const named = new Set(labelled);
  const xs = rows.map((row) => row[x]).filter(Number.isFinite);
  const rightQuarter = xs.length ? Math.min(...xs) + 0.75 * (Math.max(...xs) - Math.min(...xs)) : Infinity;
  const points = {
    type: 'scatter',
    name: '',
    symbolSize: 9,
    color: THEME.accentColor,
    itemStyle: {opacity: 0.75},
    label: {position: 'right', fontSize: 10, color: THEME.mutedColor, formatter: '{b}'},
    labelLayout: {hideOverlap: true},
    data: rows.map((row) => ({
      value: [row[x], row[y]],
      name: row[name],
      ...(colorOf ? {itemStyle: {color: colorOf(row)}} : {}),
      label: {show: named.has(row[name]), position: row[x] > rightQuarter ? 'left' : 'right'},
    })),
  };
  const series = [points];
  if (fit) {
    series.push({
      type: 'line',
      name: '',
      data: fit,
      showSymbol: false,
      silent: true,
      tooltip: {show: false},
      lineStyle: {color: FIT_COLOR, width: 1.5, type: 'dashed'},
    });
  }
  if (diagonal && axes) {
    const low = Math.min(axes.xMin, axes.yMin);
    const high = Math.max(axes.xMax, axes.yMax);
    series.push({
      type: 'line',
      name: '',
      data: [[low, low], [high, high]],
      showSymbol: false,
      silent: true,
      tooltip: {show: false},
      lineStyle: {color: 'rgba(0,0,0,.35)', width: 1.5, type: 'dashed'},
    });
  }
  const fixed = (min, max) => (axes ? {min, max, interval: axes.interval} : {});
  return {
    ...baseOption(),
    grid: {left: 8, right: 16, top: 16, bottom: xName ? 32 : 8, containLabel: true},
    xAxis: {
      type: 'value',
      scale: !axes,
      ...fixed(axes && axes.xMin, axes && axes.xMax),
      name: xName,
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: {color: THEME.mutedColor, fontSize: 11},
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => xFormatter(value)},
    },
    yAxis: {
      type: 'value',
      scale: !axes,
      ...fixed(axes && axes.yMin, axes && axes.yMax),
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => yFormatter(value)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(params) {
        if (params.seriesType !== 'scatter') {
          return '';
        }
        const [px, py] = params.value;
        const part = (label, text) => escapeHtml(label ? `${label} ${text}` : text);
        return `<strong>${escapeHtml(params.name)}</strong><br>${part(xLabel, xFormatter(px))} · ${part(yLabel, yFormatter(py))}`;
      },
    },
    series,
  };
}

/**
 * Render the cloud (`scatterOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows one per point
 * @param {object} opts see `scatterOption`
 * @returns {object} the ECharts instance
 */
export function renderScatter(el, rows, opts) {
  return mountChart(el, scatterOption(rows, opts));
}
