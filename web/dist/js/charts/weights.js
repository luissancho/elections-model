// The live example of the average: the polls that weigh on the average of one party on the day of the estimate,
// sized and shaded by their weight, the straight local line of that day with its band, the Gaussian bell of the
// kernel on a second axis and two guides at the day and at the value read.
import {axisLabelStyle, baseOption, mountChart, THEME, tooltipStyle} from './base.js';
import {bandSeries, dayTime, isoDay} from './series.js';
import {fmtDateShort, fmtNum, fmtPct} from '../format.js';
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
 * corrected figure, the total weight and its four factors.
 *
 * @param {object} row a row of the example (`pollster`, `sponsor`, `date`, `sample_size`, `pct`, `pct_adj`,
 *   `weight`, `weight_kernel`, `weight_over`, `weight_sample`, `weight_rating`)
 * @param {string} name party of the example
 * @returns {string[]} HTML lines, escaped
 */
export function pollLines(row, name) {
  const who = row.sponsor ? `${row.pollster} / ${row.sponsor}` : row.pollster;
  const lines = pollTooltipLines({name: who, date: row.date, sample_size: row.sample_size}, [], {});
  lines.push(escapeText(`Publicado ${fmtPct(row.pct)} · corregido ${fmtPct(row.pct_adj)}`));
  lines.push(escapeText(`Peso ${fmtNum(row.weight, 2)}`));
  lines.push(escapeText(`cercanía ${fmtNum(row.weight_kernel, 2)} · sobre. ${fmtNum(row.weight_over, 2)}`
    + ` · muestra ${fmtNum(row.weight_sample, 2)} · rating ${fmtNum(row.weight_rating, 2)}`));
  return lines;
}

/**
 * ECharts option of the example: the local line of the day with its band, the polls sized and shaded by weight,
 * the kernel bell on the right axis and the guides at the day and the value read.
 *
 * @param {object} example the `example` of the page's initial data (`name`, `color`, `date`, `stats`, `rows`,
 *   `line` {dates, mean, lo, hi}, `kernel` {dates, values})
 * @returns {object} the option
 */
export function weightsOption(example) {
  const {name, color, date, stats, rows, line, kernel} = example;
  const times = line.dates.map(dayTime);
  const weights = rows.map((row) => row.weight || 0);
  const wmax = Math.max(0, ...weights) || 1;
  const points = rows.map((row, index) => {
    const share = (row.weight || 0) / wmax;
    return {
      value: [dayTime(row.date), row.pct, index],
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
      return escapeText(`Peso por cercanía ${fmtNum(params.data[1], 2)}`);
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
      axisLabel: {...axisLabelStyle(), formatter: (value) => fmtDateShort(isoDay(value))},
      axisLine: {lineStyle: {color: THEME.axisColor}}, splitLine: {show: false},
    },
    yAxis: [
      {
        type: 'value', scale: true, position: 'right',
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
 * @returns {object} the ECharts instance
 */
export function renderWeights(el, example) {
  return mountChart(el, weightsOption(example));
}
