// A slope chart: one line per row between two moments, with the zero marked and the negative side shaded (the
// CERA study: the block margin of the exterior vote from the previous regional election to the last one).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

// The shade of the values below zero, the red of the study's charts, faint.
const BELOW_ZERO = 'rgba(206,59,59,.08)';

/** A value with its sign: '+9,2', '-0,9', '0,0'. */
function signed(value) {
  return `${value > 0 ? '+' : ''}${fmtNum(value, 1)}`;
}

/** A tick of the value axis with its sign, without decimals when whole: '+3', '-1,5'. */
function signedTick(value) {
  return `${value > 0 ? '+' : ''}${fmtNum(value, Number.isInteger(value) ? 0 : 1)}`;
}

/** A round step for about `ticks` intervals over `span`: 1, 2 or 5 times a power of ten. */
function niceStep(span, ticks = 5) {
  const raw = span / ticks;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].find((m) => m * power >= raw) * power;
  return Number(step.toPrecision(12));
}

/**
 * ECharts option of the slope chart: one `line` series per row from its `from` value to its `to` value over two
 * categories, the end value written beside its point, a dashed line at zero and the band below zero shaded (both
 * on the first series). The value axis always shows the zero, leaves some room around the values and ends on
 * round ticks.
 *
 * @param {{name: string, from: number, to: number, color: string}[]} rows one line each
 * @param {object} opts `labels` (the two categories), `yFormatter` (number → text of the labels and the tooltip;
 *   signed with one decimal by default) and `axisFormatter` (the ticks; signed, decimals only when not whole)
 * @returns {object} the option
 */
export function slopeOption(rows, {labels = ['Comicio anterior', '2026'], yFormatter = signed, axisFormatter = signedTick} = {}) {
  const values = rows.flatMap((row) => [row.from, row.to]).filter(Number.isFinite);
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const step = niceStep(Math.max(1, hi - lo) * 1.3);
  const round = (value) => Number(value.toPrecision(12));
  const yMin = round(Math.floor((lo - 0.15 * step) / step) * step);
  const yMax = round(Math.ceil((hi + 0.15 * step) / step) * step);
  const series = rows.map((row) => ({
    type: 'line',
    name: row.name,
    color: row.color,
    symbolSize: 9,
    lineStyle: {width: 3},
    data: [row.from, {
      value: row.to,
      label: {show: true, position: 'right', color: row.color, fontWeight: 700, formatter: (p) => yFormatter(p.value)},
    }],
  }));
  if (series.length) {
    series[0].markLine = {
      silent: true,
      symbol: 'none',
      lineStyle: {type: 'dashed', color: THEME.mutedColor},
      label: {show: false},
      data: [{yAxis: 0}],
    };
    series[0].markArea = {silent: true, itemStyle: {color: BELOW_ZERO}, data: [[{yAxis: yMin}, {yAxis: 0}]]};
  }
  return {
    ...baseOption(),
    grid: {left: 8, right: 40, top: 36, bottom: 8, containLabel: true},
    legend: {top: 0, data: rows.map((row) => row.name), textStyle: {color: THEME.mutedColor, fontSize: 11.5}},
    xAxis: {type: 'category', data: labels, boundaryGap: true, axisLabel: {color: THEME.textColor, fontSize: 11.5}},
    yAxis: {
      type: 'value',
      min: yMin,
      max: yMax,
      interval: step,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => axisFormatter(value)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      formatter(params) {
        const lines = params
          .filter((item) => Number.isFinite(item.value))
          .map((item) => `${item.marker}${escapeHtml(item.seriesName)}: ${escapeHtml(yFormatter(item.value))}`);
        return [`<strong>${escapeHtml(params[0].axisValue)}</strong>`, ...lines].join('<br>');
      },
    },
    series,
  };
}

/**
 * Render the slope chart (`slopeOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {{name: string, from: number, to: number, color: string}[]} rows one line each
 * @param {object} opts see `slopeOption`
 * @returns {object} the ECharts instance
 */
export function renderSlope(el, rows, opts = {}) {
  return mountChart(el, slopeOption(rows, opts));
}
