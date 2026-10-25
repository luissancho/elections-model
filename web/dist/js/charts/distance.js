// Horizontal bars of the distance of each pollster's last poll to the result (the CIS bias study): the
// pollster in red, the consensus of the houses in the text colour, the rest in grey.
import {baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

const CIS_COLOR = '#ce3b3b';
const OTHER_COLOR = '#c7cdd8';
const ROW_HEIGHT = 15;
const EXTRA_HEIGHT = 30;

/**
 * ECharts option of the bars, one per row in the order given (the top row first).
 *
 * @param {{name: string, dist: number, cis?: boolean, consensus?: boolean}[]} rows already sorted
 * @returns {object} the option
 */
export function distanceOption(rows) {
  const colorOf = (row) => (row.cis ? CIS_COLOR : row.consensus ? THEME.textColor : OTHER_COLOR);
  return {
    ...baseOption(),
    grid: {left: 8, right: 44, top: 4, bottom: 4, containLabel: true},
    tooltip: {
      ...tooltipStyle(),
      formatter: (params) => `<strong>${escapeHtml(rows[params.dataIndex].name)}</strong><br>`
        + `${escapeHtml(fmtNum(rows[params.dataIndex].dist, 1))} puntos del resultado`,
    },
    xAxis: {
      type: 'value',
      min: 0,
      axisLabel: {show: false},
      splitLine: {show: false},
      axisLine: {show: false},
      axisTick: {show: false},
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map((row) => row.name),
      axisTick: {show: false},
      axisLine: {show: false},
      axisLabel: {
        fontSize: 10.5,
        color: THEME.textColor,
        fontWeight: 500,
        formatter: (name, index) => (rows[index] && rows[index].consensus ? `{c|${name}}` : name),
        rich: {c: {fontWeight: 'bold', color: THEME.textColor, fontSize: 10.5}},
      },
    },
    series: [{
      type: 'bar',
      barWidth: 9,
      data: rows.map((row) => ({value: row.dist, itemStyle: {color: colorOf(row)}})),
      label: {show: true, position: 'right', fontSize: 10.5, color: THEME.textColor, formatter: (p) => fmtNum(p.value, 1)},
    }],
  };
}

/**
 * Render the bars (`distanceOption`) in `el`; the height grows with the number of rows.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows see `distanceOption`
 * @returns {object} the ECharts instance
 */
export function renderDistance(el, rows) {
  el.style.height = `${rows.length * ROW_HEIGHT + EXTRA_HEIGHT}px`;
  return mountChart(el, distanceOption(rows));
}
