// A radar of a few indicators, one polygon per series (the CERA study: today against «everything at the maximum»).
import {baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtPct} from '../format.js';

// Opacity of the area of a filled polygon and of an outlined one.
const FILLED = 0.25;
const OUTLINED = 0.06;

/**
 * ECharts option of the radar: the axes of `indicators` and one polygon per entry of `series`, the filled ones
 * shaded and labelled with their values (in %, but for the zeros, which would pile up on the centre), the others
 * only outlined; the tooltip gives every value with a decimal. A `\n` in a name breaks its line.
 *
 * @param {{name: string, max: number}[]} indicators the axes, clockwise from the top
 * @param {{name: string, value: number[], color: string, filled: boolean}[]} series one polygon each, with a value
 *   per indicator
 * @returns {object} the option
 */
export function radarOption(indicators, series) {
  return {
    ...baseOption(),
    legend: {bottom: 0, data: series.map((item) => item.name), textStyle: {color: THEME.mutedColor, fontSize: 11.5}},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(params) {
        const lines = indicators.map((axis, i) => `${escapeHtml(axis.name)}: ${escapeHtml(fmtPct(params.value[i], 1))}`);
        return [`<strong>${escapeHtml(params.name)}</strong>`, ...lines].join('<br>');
      },
    },
    radar: {
      indicator: indicators,
      radius: '55%',
      center: ['50%', '48%'],
      axisName: {color: THEME.textColor, fontSize: 11},
      splitArea: {areaStyle: {color: ['#fff', '#f7f7f9']}},
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLine: {lineStyle: {color: THEME.gridColor}},
    },
    series: [{
      type: 'radar',
      data: series.map((item) => ({
        name: item.name,
        value: item.value,
        lineStyle: {color: item.color, width: item.filled ? 2.5 : 2},
        itemStyle: {color: item.color},
        areaStyle: {color: item.color, opacity: item.filled ? FILLED : OUTLINED},
        label: {show: Boolean(item.filled), color: item.color, fontSize: 10, formatter: (p) => (p.value ? fmtPct(p.value, 0) : '')},
      })),
    }],
  };
}

/**
 * Render the radar (`radarOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {{name: string, max: number}[]} indicators the axes
 * @param {object[]} series the polygons
 * @returns {object} the ECharts instance
 */
export function renderRadar(el, indicators, series) {
  return mountChart(el, radarOption(indicators, series));
}
