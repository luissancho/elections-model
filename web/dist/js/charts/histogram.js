// Histogram of seats over the simulations, with mark lines at the median and at the majority.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtInt, fmtPct} from '../format.js';
import {binWidth, histogram, majorityInRange} from '../seats.js';

/**
 * Index of the bin holding `value`, clamped to the bins.
 *
 * @param {number[]} starts bin starts
 * @param {number} width bin width
 * @param {number} value seats
 * @returns {number} the category index
 */
function binIndex(starts, width, value) {
  const index = Math.floor((value - starts[0]) / width);
  return Math.min(Math.max(index, 0), starts.length - 1);
}

/**
 * Render the share of simulations per seat bin as bars, with a solid mark line at the median and a dashed
 * one at the majority (only when the majority lies within the bins).
 *
 * @param {HTMLElement} el chart container
 * @param {number[]} values seats per simulation
 * @param {object} opts `color` (bars), `majority`, `median`, `lo`, `hi` (seats, tooltip subtitle),
 *   `width` (bin width, from `binWidth` when null), `formatter` (seats → text)
 * @returns {object} the ECharts instance
 */
export function renderHistogram(el, values, {
  color = OTHERS_COLOR, majority = null, median = null, lo = null, hi = null, width = null, formatter = fmtInt,
} = {}) {
  const step = width ?? binWidth(values);
  const {starts, counts} = histogram(values, step);
  const n = values.length;
  const shares = counts.map((count) => (n ? (100 * count) / n : 0));

  const marks = [];
  if (median !== null && starts.length) {
    marks.push({
      xAxis: binIndex(starts, step, median),
      lineStyle: {type: 'solid', color: THEME.markColor},
      label: {formatter: formatter(median), position: 'insideEndTop', color: THEME.markColor},
    });
  }
  if (majorityInRange(starts, step, majority)) {
    marks.push({
      xAxis: binIndex(starts, step, majority),
      lineStyle: {type: 'dashed', color: THEME.markColor},
      label: {formatter: `Mayoría: ${formatter(majority)}`, position: 'end', color: THEME.markColor},
    });
  }

  const label = (start) => (step === 1
    ? `${formatter(start)} escaños`
    : `${formatter(start)}–${formatter(start + step - 1)} escaños`);
  let subtitle = '';
  if (median !== null) {
    subtitle = `Mediana: ${formatter(median)}`;
    if (lo !== null && hi !== null) {
      subtitle += ` · intervalo 95 %: ${formatter(lo)}–${formatter(hi)}`;
    }
  }

  const option = {
    ...baseOption(),
    grid: {left: 8, right: 16, top: 24, bottom: 8, containLabel: true},
    xAxis: {
      type: 'category',
      data: starts.map(String),
      axisTick: {alignWithLabel: true},
      axisLabel: {formatter: (value) => formatter(Number(value))},
    },
    yAxis: {
      type: 'value',
      min: 0,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (value) => fmtPct(value, 0)},
    },
    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: {type: 'shadow'},
      formatter(params) {
        if (!params.length) {
          return '';
        }
        const index = params[0].dataIndex;
        const lines = [`<strong>${escapeHtml(`${label(starts[index])}: ${fmtPct(shares[index])}`)}</strong>`];
        if (subtitle) {
          lines.push(escapeHtml(subtitle));
        }
        return lines.join('<br>');
      },
    },
    series: [{
      type: 'bar',
      name: 'Simulaciones',
      barCategoryGap: '10%',
      color,
      data: shares,
      markLine: {silent: true, symbol: 'none', data: marks},
    }],
  };
  return mountChart(el, option);
}
