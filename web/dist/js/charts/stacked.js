// One horizontal bar split into stacked segments (seats per block), with the majority marked.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtInt} from '../format.js';

const LABEL_MIN_SHARE = 0.05;

/**
 * Render `rows` as consecutive segments of a single horizontal bar from 0 to `total`, labelled inside when
 * wide enough, with a dashed mark line at `majority`.
 *
 * @param {HTMLElement} el chart container
 * @param {{name: string, seats: number}[]} rows segments, in drawing order
 * @param {object} opts `colors` ({name: colour}), `fullnames` ({name: full name}, tooltip), `majority`,
 *   `total` (seats of the chamber, axis maximum)
 * @returns {object} the ECharts instance
 */
export function renderStacked(el, rows, {colors = {}, fullnames = {}, majority = null, total} = {}) {
  const segments = rows.map((row) => ({
    type: 'bar',
    name: row.name,
    stack: 'seats',
    barWidth: '60%',
    color: colors[row.name] || OTHERS_COLOR,
    label: {
      show: total > 0 && row.seats / total >= LABEL_MIN_SHARE,
      position: 'inside',
      color: '#fff',
      formatter: `${row.name} ${fmtInt(row.seats)}`,
    },
    data: [row.seats],
  }));
  // Mark line on its own empty series, so it never depends on a segment.
  const marks = majority === null || majority === undefined ? [] : [{
    type: 'line',
    name: 'marks',
    data: [],
    silent: true,
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: {type: 'dashed', color: THEME.markColor},
      label: {formatter: `Mayoría: ${fmtInt(majority)}`, position: 'end', color: THEME.markColor},
      data: [{xAxis: majority}],
    },
  }];

  const option = {
    ...baseOption(),
    grid: {left: 8, right: 16, top: 24, bottom: 8, containLabel: true},
    xAxis: {
      type: 'value',
      min: 0,
      max: total,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (value) => fmtInt(value)},
    },
    yAxis: {type: 'category', data: ['Escaños'], show: false},
    tooltip: {
      trigger: 'item',
      formatter(params) {
        const name = fullnames[params.seriesName] || params.seriesName;
        return `${params.marker}<strong>${escapeHtml(name)}</strong><br>${escapeHtml(`${fmtInt(params.value)} escaños`)}`;
      },
    },
    series: [...segments, ...marks],
  };
  return mountChart(el, option);
}
