// The monthly CERA census as a smooth line over a time axis, with the dates that matter marked (the CERA study).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtDate, fmtInt, fmtNum} from '../format.js';

// The colour of the marked dates, the red of the study's charts.
const MARK_COLOR = '#ce3b3b';

/**
 * A count of electors in millions, as the axis of the census reads it: '2,75 M'.
 *
 * @param {number} value electors
 * @returns {string} the text
 */
export function millions(value) {
  return `${fmtNum(value / 1e6, 2)} M`;
}

/**
 * ECharts option of the census: a smooth line with a light area under it, the value axis in millions (fitted to
 * the data, not from zero) and one dashed vertical line per mark, its label rotated along it.
 *
 * @param {Array} points `[iso_date, electors]` pairs, ascending by date
 * @param {{date: string, label: string}[]} marks the dates to mark and their labels
 * @param {object} opts `color` (of the line and its area, the site accent by default)
 * @returns {object} the option
 */
export function censusOption(points, marks, {color = THEME.accentColor} = {}) {
  return {
    ...baseOption(),
    grid: {left: 8, right: 16, top: 16, bottom: 8, containLabel: true},
    xAxis: {type: 'time', axisLabel: {...axisLabelStyle()}},
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => millions(value)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      formatter(params) {
        const [date, value] = params[0].value;
        return `${escapeHtml(fmtDate(date))}<br><strong>${escapeHtml(fmtInt(value))}</strong> electores`;
      },
    },
    series: [{
      type: 'line',
      name: 'Censo CERA',
      data: points,
      smooth: true,
      showSymbol: false,
      color,
      lineStyle: {width: 2.5},
      areaStyle: {color, opacity: 0.08},
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: {type: 'dashed', color: MARK_COLOR},
        label: {color: MARK_COLOR, fontSize: 10, rotate: 90, position: 'insideEndTop', formatter: '{b}'},
        data: marks.map((mark) => ({xAxis: mark.date, name: mark.label})),
      },
    }],
  };
}

/**
 * Render the census (`censusOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {Array} points `[iso_date, electors]` pairs
 * @param {{date: string, label: string}[]} marks the dates to mark
 * @param {object} opts see `censusOption`
 * @returns {object} the ECharts instance
 */
export function renderCensus(el, points, marks, opts = {}) {
  return mountChart(el, censusOption(points, marks, opts));
}
