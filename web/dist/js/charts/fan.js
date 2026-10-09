// Fan of the vote share by horizon: per party the mean line and the `lo–hi` band over the days ahead.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {bandSeries} from './series.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtInt, fmtPct, fmtRange} from '../format.js';

/**
 * Render the fan of `parties`: the mean line and the `lo–hi` band of each over the horizons (days), with
 * a vertical mark line at `markAt`.
 *
 * @param {HTMLElement} el chart container
 * @param {{horizons: number[], rows: object[]}} fan the fan part (rows `name, horizon, mean, sd, lo, hi`)
 * @param {object} opts `parties` (names, legend order), `selected` (names shown at first; all when null),
 *   `colors` ({name: colour}), `fullnames` ({name: full name}, tooltip), `markAt` (horizon of the mark
 *   line), `markLabel` (its text)
 * @returns {object} the ECharts instance
 */
export function renderFan(el, fan, {
  parties = [], selected = null, colors = {}, fullnames = {}, markAt = null, markLabel = '',
} = {}) {
  const color = (name) => colors[name] || OTHERS_COLOR;
  const horizons = fan.horizons || [];
  const byKey = new Map((fan.rows || []).map((row) => [`${row.name}|${row.horizon}`, row]));
  const value = (name, horizon, key) => {
    const row = byKey.get(`${name}|${horizon}`);
    return row && row[key] !== undefined ? row[key] : null;
  };

  // Mean lines first: the legend takes its icon from the first series of each name.
  const means = parties.map((name) => ({
    type: 'line',
    name,
    color: color(name),
    z: 3,
    symbolSize: 5,
    connectNulls: false,
    lineStyle: {width: 2},
    data: horizons.map((h) => [h, value(name, h, 'mean')]),
  }));
  const bands = parties.flatMap((name) => bandSeries(
    name, horizons, horizons.map((h) => value(name, h, 'lo')), horizons.map((h) => value(name, h, 'hi')),
    {color: color(name)},
  ));
  // Mark line on its own empty series, so hiding a party never hides it.
  const marks = markAt === null || markAt === undefined ? [] : [{
    type: 'line',
    name: 'marks',
    data: [],
    silent: true,
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: {type: 'dashed', color: THEME.markColor},
      data: [{xAxis: markAt, label: {formatter: markLabel, position: 'end', color: THEME.markColor}}],
    },
  }];

  function tooltip(params) {
    if (!params.length) {
      return '';
    }
    const horizon = params[0].value[0];
    // Parties of the shown series: the legend-hidden ones are not in `params`.
    const shown = new Set(params.map((item) => item.seriesName));
    const rows = parties
      .filter((name) => shown.has(name) && value(name, horizon, 'mean') !== null)
      .map((name) => ({
        name, mean: value(name, horizon, 'mean'), lo: value(name, horizon, 'lo'), hi: value(name, horizon, 'hi'),
      }))
      .sort((a, b) => b.mean - a.mean);
    const lines = [`<strong>${escapeHtml(`${fmtInt(horizon)} días`)}</strong>`];
    for (const row of rows) {
      const marker = window.echarts.format.getTooltipMarker(escapeHtml(color(row.name)));
      const label = fullnames[row.name] || row.name;
      lines.push(`${marker}${escapeHtml(`${label}: ${fmtPct(row.mean)} (${fmtRange(row.lo, row.hi)})`)}`);
    }
    return lines.join('<br>');
  }

  const option = {
    ...baseOption(),
    legend: {
      type: 'scroll',
      top: 0,
      data: parties,
      selected: Object.fromEntries(parties.map((name) => [name, !selected || selected.includes(name)])),
    },
    // Right margin wide enough for the centred mark-line label at the end of the axis.
    grid: {left: 8, right: 48, top: 40, bottom: 8, containLabel: true},
    xAxis: {
      type: 'value',
      min: Math.min(...horizons, markAt ?? Infinity),
      max: Math.max(...horizons, markAt ?? -Infinity),
      axisLabel: {formatter: '{value} d'},
      splitLine: {show: false},
    },
    yAxis: {
      type: 'value',
      min: 0,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (v) => fmtPct(v, 0)},
    },
    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: {type: 'line', snap: true},
      formatter: tooltip,
    },
    series: [...means, ...bands, ...marks],
  };
  return mountChart(el, option);
}
