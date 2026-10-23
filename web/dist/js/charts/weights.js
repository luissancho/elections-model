// The live example of the average: the polls that weigh on the average of one party on the day of the estimate,
// sized and shaded by their weight, the straight local line of that day with its band, the Gaussian bell of the
// kernel on a second axis and two guides at the day and at the value read.
import {axisLabelStyle, baseOption, mountChart, THEME, timeAxisLabel, tooltipStyle} from './base.js';
import {bandSeries, dayTime} from './series.js';
import {fmtNum, fmtPct} from '../format.js';
import {escapeText, meanTooltipLine, pollTooltipLines} from '../polls-tooltip.js';

const BAND_OPACITY = 0.15;
// Point size (px) and opacity of a poll grow linearly with its weight relative to the heaviest one.
const MIN_SIZE = 6;
const SIZE_RANGE = 14;
const MIN_OPACITY = 0.25;
const NAMES = {line: 'Recta del día', band: 'Banda 95 %', polls: 'Sondeos', kernel: 'Kernel', marks: 'marks'};
// Symbols of the line and the bell: invisible until hovered, so that each day has its own tooltip (as `series.js`).
const HOVER_SYMBOLS = {
  showSymbol: true,
  symbol: 'circle',
  symbolSize: 8,
  itemStyle: {opacity: 0},
  emphasis: {itemStyle: {opacity: 1}},
};

/**
 * Lines of the tooltip of one poll of the example: pollster (and sponsor), date and sample, the published and the
 * corrected figure, the total weight and its four factors, named as the columns of the list (F(kernel), F(solape),
 * F(muestra), F(rating)).
 *
 * @param {object} row a row of the example (`pollster`, `sponsor`, `date`, `sample_size`, `pct`, `pct_adj`,
 *   `weight`, `weight_kernel`, `weight_over`, `weight_sample`, `weight_rating`)
 * @param {string} name party of the example
 * @returns {string[]} HTML lines, escaped
 */
export function pollLines(row, name) {
  const who = row.sponsor ? `${row.pollster} / ${row.sponsor}` : row.pollster;
  const lines = pollTooltipLines({name: who, date: row.date, sample_size: row.sample_size}, [], {});
  lines.push(escapeText(`Publicada ${fmtPct(row.pct)} · Corregida ${fmtPct(row.pct_adj)}`));
  lines.push(escapeText(`Peso ${fmtNum(row.weight, 2)}`));
  lines.push(escapeText(`F(kernel) ${fmtNum(row.weight_kernel, 2)} · F(solape) ${fmtNum(row.weight_over, 2)}`
    + ` · F(muestra) ${fmtNum(row.weight_sample, 2)} · F(rating) ${fmtNum(row.weight_rating, 2)}`));
  return lines;
}

/**
 * Line of the tooltip of one day of the kernel bell: the factor, named as in the list.
 *
 * @param {number|null} value kernel weight of the day, in [0, 1]
 * @returns {string} the HTML line, escaped
 */
export function kernelLine(value) {
  return escapeText(`F(kernel) ${fmtNum(value, 2)}`);
}

/**
 * Range of the value axis, common to the published and the corrected figures of the polls: the union of both, of
 * the line and of its band, widened half a point and rounded outwards to the unit. Switching the points between the
 * two figures then keeps the axis still, so that the points are seen to move.
 *
 * @param {object[]} rows the polls of the example (`pct`, `pct_adj`)
 * @param {object} line the local line (`mean`, `lo`, `hi`)
 * @returns {{min: number, max: number}|{}} the axis bounds, empty without a finite value
 */
export function valueRange(rows, line) {
  const values = [
    ...rows.map((row) => row.pct), ...rows.map((row) => row.pct_adj),
    ...(line.mean || []), ...(line.lo || []), ...(line.hi || []),
  ].filter((value) => Number.isFinite(value));
  if (!values.length) {
    return {};
  }
  const pad = 0.5;
  return {min: Math.floor(Math.min(...values) - pad), max: Math.ceil(Math.max(...values) + pad)};
}

/**
 * ECharts option of the example: the local line of the day with its band, the polls sized and shaded by weight,
 * the kernel bell on the right axis and the guides at the day and the value read.
 *
 * @param {object} example the `example` of the page's initial data (`name`, `color`, `date`, `stats`, `rows`,
 *   `line` {dates, mean, lo, hi}, `kernel` {dates, values})
 * @param {object} [options]
 * @param {string} [options.field='pct'] figure of each poll the points show: `pct` (as published) or `pct_adj`
 *   (net of the house effect, the figure the line is fitted on)
 * @returns {object} the option
 */
export function weightsOption(example, {field = 'pct'} = {}) {
  const {name, color, date, stats, rows, line, kernel} = example;
  const times = line.dates.map(dayTime);
  const weights = rows.map((row) => row.weight || 0);
  const wmax = Math.max(0, ...weights) || 1;
  const points = rows.map((row, index) => {
    const share = (row.weight || 0) / wmax;
    return {
      value: [dayTime(row.date), row[field] ?? null, index],
      symbolSize: Math.round(MIN_SIZE + SIZE_RANGE * share),
      itemStyle: {color, opacity: MIN_OPACITY + (1 - MIN_OPACITY) * share},
    };
  });
  const series = [
    {
      type: 'line', name: NAMES.line, color, z: 3, ...HOVER_SYMBOLS, connectNulls: false, lineStyle: {width: 2},
      data: times.map((t, i) => [t, line.mean[i] ?? null]),
    },
    ...bandSeries(NAMES.band, times, line.lo, line.hi, {color, opacity: BAND_OPACITY, stack: 'band:weights'}),
    {type: 'scatter', name: NAMES.polls, color, z: 4, data: points},
    {
      type: 'line', name: NAMES.kernel, yAxisIndex: 1, color: THEME.mutedColor, z: 2, ...HOVER_SYMBOLS,
      lineStyle: {width: 1.5, type: 'dashed'},
      data: kernel.dates.map((d, i) => [dayTime(d), kernel.values[i] ?? null]),
    },
    {
      type: 'line', name: NAMES.marks, data: [], silent: true,
      markLine: {
        silent: true, symbol: 'none', label: {show: false}, lineStyle: {type: 'dashed', color: THEME.markColor},
        data: [{xAxis: dayTime(date)}, {yAxis: stats.mean}],
      },
    },
  ];

  /** Box of the item under the pointer: a poll, a day of the line or a day of the bell. */
  function tooltip(params) {
    if (params.seriesName === NAMES.polls) {
      const row = rows[params.data.value[2]];
      return row ? pollLines(row, name).join('<br>') : '';
    }
    if (params.seriesName === NAMES.line) {
      const i = params.dataIndex;
      return meanTooltipLine(name, line.mean[i] ?? null, line.lo[i] ?? null, line.hi[i] ?? null, line.dates[i], color);
    }
    if (params.seriesName === NAMES.kernel) {
      return kernelLine(params.data[1]);
    }
    return '';
  }

  return {
    ...baseOption(),
    legend: {data: [NAMES.polls, NAMES.line, NAMES.band, NAMES.kernel], top: 0},
    grid: {top: 40, left: 16, right: 48, bottom: 28, containLabel: false},
    tooltip: {trigger: 'item', ...tooltipStyle(), formatter: tooltip},
    xAxis: {
      type: 'time', min: times[0], max: times[times.length - 1],
      axisLabel: timeAxisLabel(),
      axisLine: {lineStyle: {color: THEME.axisColor}}, splitLine: {show: false},
    },
    yAxis: [
      {
        // the same bounds whichever figure the points show (`valueRange`); `scale` only acts without them
        type: 'value', scale: true, ...valueRange(rows, line), position: 'right',
        axisLabel: {...axisLabelStyle(), formatter: (value) => fmtPct(value, 0)},
        splitLine: {lineStyle: {color: THEME.gridColor}},
      },
      {type: 'value', min: 0, max: 1, show: false, position: 'left'},
    ],
    series,
  };
}

/**
 * Draw the example into `el`.
 *
 * @param {HTMLElement} el the chart container
 * @param {object} example the `example` of the page's initial data
 * @param {object} [options] those of `weightsOption` (`field`)
 * @returns {object} the ECharts instance
 */
export function renderWeights(el, example, options = {}) {
  const option = weightsOption(example, options);
  if (el.__chart) {
    // a redraw with the other figure merges into the mounted chart: same series, same axes, so the points glide to
    // their new values instead of being redrawn from scratch
    el.__chart.setOption(option);
    return el.__chart;
  }
  return mountChart(el, option);
}
