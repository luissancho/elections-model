// Histogram of seats over the simulations, with mark lines at the median and at the majority. The unit, the bin
// labels and the median mark are options, so the error history reuses it for the poll errors (in points).
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
 * Render the share of simulations per seat bin (or the raw count of values, with `share: false`) as bars, with a solid mark line at the median and a dashed
 * one at the majority (only when the majority lies within the bins).
 *
 * @param {HTMLElement} el chart container
 * @param {number[]} values seats per simulation
 * @param {object} opts `color` (bars), `majority`, `median`, `lo`, `hi` (seats, tooltip subtitle),
 *   `width` (bin width, from `binWidth` when null), `formatter` (seats → text), `unit` (of the default bin
 *   labels), `label` (`(start, step) => text` of a bin, replacing the default one) and `markLabel` (text of the
 *   median mark and of the tooltip subtitle, replacing `Mediana: …`), `share` (bars as the % of the values,
 *   the default; `false` draws the raw counts, with integer ticks) and `countLabel` (count → tooltip text, used
 *   when `share` is false)
 * @returns {object} the ECharts instance
 */
export function renderHistogram(el, values, {
  color = OTHERS_COLOR, majority = null, median = null, lo = null, hi = null, width = null, formatter = fmtInt,
  unit = 'escaños', label = null, markLabel = null, share = true, countLabel = fmtInt,
} = {}) {
  const step = width ?? binWidth(values);
  const {starts, counts} = histogram(values, step);
  const n = values.length;
  const shares = counts.map((count) => (n ? (100 * count) / n : 0));
  const heights = share ? shares : counts;
  const heightText = (index) => (share ? fmtPct(shares[index]) : countLabel(counts[index]));

  const marks = [];
  if (median !== null && starts.length) {
    marks.push({
      xAxis: binIndex(starts, step, median),
      lineStyle: {type: 'solid', color: THEME.markColor},
      label: {formatter: markLabel ?? formatter(median), position: 'insideEndTop', color: THEME.markColor},
    });
  }
  if (majorityInRange(starts, step, majority)) {
    marks.push({
      xAxis: binIndex(starts, step, majority),
      lineStyle: {type: 'dashed', color: THEME.markColor},
      label: {formatter: `Mayoría: ${formatter(majority)}`, position: 'end', color: THEME.markColor},
    });
  }

  const binLabel = label ? (start) => label(start, step) : (start) => (step === 1
    ? `${formatter(start)} ${unit}`
    : `${formatter(start)}–${formatter(start + step - 1)} ${unit}`);
  let subtitle = '';
  if (median !== null && markLabel !== null) {
    subtitle = markLabel;
  } else if (median !== null) {
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
      axisLabel: {formatter: share ? (value) => fmtPct(value, 0) : (value) => fmtInt(value)},
      ...(share ? {} : {minInterval: 1}),
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
        const lines = [`<strong>${escapeHtml(`${binLabel(starts[index])}: ${heightText(index)}`)}</strong>`];
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
      data: heights,
      markLine: {silent: true, symbol: 'none', data: marks},
    }],
  };
  return mountChart(el, option);
}
