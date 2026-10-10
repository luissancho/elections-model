// A single line with markers over a time axis (e.g. the rating of a pollster at each election).
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {fmtDate, fmtDateShort, fmtNum} from '../format.js';

/**
 * Render one line over a time axis.
 *
 * @param {HTMLElement} el chart container
 * @param {Array} points `[iso_date, value]` pairs, ascending by date
 * @param {object} opts `formatter` (value → text), `name` (series name), `color` (line colour)
 * @returns {object} the ECharts instance
 */
export function renderLine(el, points, {formatter = (x) => fmtNum(x, 1), name = '', color = THEME.markColor} = {}) {
  const option = {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 24, bottom: 8, containLabel: true},
    xAxis: {
      type: 'time',
      axisLabel: {formatter: (ms) => fmtDateShort(new Date(ms).toISOString().slice(0, 10))},
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (x) => formatter(x)},
    },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        const [date, value] = params[0].value;
        return `${escapeHtml(fmtDate(date))} · ${escapeHtml(formatter(value))}`;
      },
    },
    series: [{
      type: 'line',
      name,
      data: points,
      showSymbol: true,
      symbolSize: 7,
      color,
      lineStyle: {width: 2},
    }],
  };
  return mountChart(el, option);
}
