// One horizontal bar split into stacked segments (seats per block), with the majority marked.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtDelta, fmtInt} from '../format.js';

const LABEL_MIN_SHARE = 0.05;

/**
 * ECharts option of `rows` as consecutive segments of a single horizontal bar from 0 to `total`, labelled inside
 * when wide enough (name, seats and, when known, the difference against the last election), with a dashed mark
 * line at `majority`.
 *
 * @param {{name: string, seats: number}[]} rows segments, in drawing order
 * @param {object} opts `colors` ({name: colour}), `fullnames` ({name: full name}, tooltip), `majority`,
 *   `total` (seats of the chamber, axis maximum), `deltas` ({name: seats difference against the last election})
 *   and `year` (of that election, for the tooltip)
 * @returns {object} the option
 */
export function stackedOption(rows, {colors = {}, fullnames = {}, majority = null, total, deltas = null, year = null} = {}) {
  const deltaOf = (name) => (deltas && Number.isFinite(deltas[name]) ? deltas[name] : null);
  const segments = rows.map((row) => {
    const delta = deltaOf(row.name);
    return {
      type: 'bar',
      name: row.name,
      stack: 'seats',
      barWidth: '60%',
      color: colors[row.name] || OTHERS_COLOR,
      label: {
        show: total > 0 && row.seats / total >= LABEL_MIN_SHARE,
        position: 'inside',
        color: '#fff',
        // the name and the seats in bold, the arrow of the difference in regular weight
        formatter: `{seats|${row.name} ${fmtInt(row.seats)}}${delta === null ? '' : ` {delta|${fmtDelta(delta)}}`}`,
        rich: {seats: {fontWeight: 700, color: '#fff'}, delta: {fontWeight: 400, color: '#fff'}},
      },
      data: [row.seats],
    };
  });
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
      axisLabel: {...axisLabelStyle(), formatter: (value) => fmtInt(value)},
    },
    yAxis: {type: 'category', data: ['Escaños'], show: false},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(params) {
        const name = fullnames[params.seriesName] || params.seriesName;
        const delta = deltaOf(params.seriesName);
        const change = delta === null ? '' : ` · ${fmtDelta(delta)}${year ? ` frente a ${year}` : ''}`;
        return `${params.marker}<strong>${escapeHtml(name)}</strong><br>${escapeHtml(`${fmtInt(params.value)} escaños${change}`)}`;
      },
    },
    series: [...segments, ...marks],
  };
  return option;
}

/**
 * Render the stacked bar of `rows` in `el` (see `stackedOption`).
 *
 * @param {HTMLElement} el chart container
 * @param {{name: string, seats: number}[]} rows segments, in drawing order
 * @param {object} opts see `stackedOption`
 * @returns {object} the ECharts instance
 */
export function renderStacked(el, rows, opts = {}) {
  return mountChart(el, stackedOption(rows, opts));
}
