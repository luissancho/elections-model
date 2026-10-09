// Evolution of a published estimate (vote or seats), one line per party over the run timestamps.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtDateShort, fmtDateTime, fmtPct} from '../format.js';

const DAY_MS = 24 * 3600 * 1000;
const ZOOM_FROM = 20;

/**
 * Points `[run_at, value]` of each party in the `mode` headline of every run (runs without the party or
 * the value are skipped).
 *
 * @param {object[]} runs `runs` list of the history part
 * @param {string} mode `nowcast` or `forecast`
 * @param {string[]} parties party names, in legend order
 * @param {string} field party field drawn (`pct` or `seats`)
 * @returns {{name: string, points: Array}[]} one entry per party
 */
export function evolutionSeries(runs, mode, parties, field = 'pct') {
  return parties.map((name) => ({
    name,
    points: runs.flatMap((run) => {
      const block = run[mode] || {};
      const party = (block.parties || []).find((row) => row.name === name);
      return party && party[field] !== null && party[field] !== undefined ? [[run.run_at, party[field]]] : [];
    }),
  }));
}

/**
 * Render the evolution chart of the `mode` headline across `runs`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} runs `runs` list of the history part (ascending by run)
 * @param {string} mode `nowcast` or `forecast`
 * @param {object} opts `colors` ({name: colour}), `parties` (names), `fullnames` ({name: full name}),
 *   `field` (party field drawn), `formatter` (value → text; the axis passes 0 digits as second argument)
 * @returns {object} the ECharts instance
 */
export function renderEvolution(el, runs, mode, {
  colors = {}, parties = [], fullnames = {}, field = 'pct', formatter = fmtPct,
} = {}) {
  const zoom = runs.length > ZOOM_FROM;
  const series = evolutionSeries(runs, mode, parties, field).map((entry) => ({
    type: 'line',
    name: entry.name,
    data: entry.points,
    showSymbol: true,
    symbolSize: 6,
    color: colors[entry.name] || OTHERS_COLOR,
    lineStyle: {width: 2},
    emphasis: {focus: 'series'},
  }));
  const option = {
    ...baseOption(),
    legend: {type: 'scroll', top: 0},
    grid: {left: 8, right: 24, top: 40, bottom: zoom ? 56 : 8, containLabel: true},
    xAxis: {
      type: 'time',
      minInterval: DAY_MS,
      axisLabel: {formatter: (value) => fmtDateShort(new Date(value).toISOString())},
      splitLine: {show: false},
    },
    yAxis: {
      type: 'value',
      min: 0,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (value) => formatter(value, 0)},
    },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        if (!params.length) {
          return '';
        }
        const rows = [...params].sort((a, b) => b.value[1] - a.value[1]);
        const lines = [`<strong>${escapeHtml(fmtDateTime(params[0].value[0]))}</strong>`];
        for (const item of rows) {
          const name = fullnames[item.seriesName] || item.seriesName;
          lines.push(`${item.marker}${escapeHtml(name)}: ${escapeHtml(formatter(item.value[1]))}`);
        }
        return lines.join('<br>');
      },
    },
    dataZoom: zoom ? [{type: 'inside'}, {type: 'slider', bottom: 8}] : [],
    series,
  };
  return mountChart(el, option);
}
