// The path of a pollster's rating: one line over time, from the first election it polled to today, on a fixed
// 0-100 axis. The general and the regional elections are coloured points over the line, and a dashed tail carries
// the current rating from the last election to today, where a hollow point marks the rating that enters the average.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtDate, fmtDateShort, fmtNum} from '../format.js';

const LINE_NAME = 'Rating';
const KIND_NAMES = {general: 'Generales', regional: 'Autonómicas'};
const TODAY_NAME = 'Hoy';
const POINT_SIZE = 9;

/**
 * Text of the tooltip of a point.
 *
 * @param {object} row the point (`label`, `event_date`, `rating`, `n_polls`, `kind`)
 * @returns {string} HTML lines
 */
function pointTooltip(row) {
  const rating = `${escapeHtml(fmtDate(row.event_date))} · rating ${escapeHtml(fmtNum(row.rating, 1))}`;
  if (row.kind === 'today') {
    return [`<strong>${escapeHtml(TODAY_NAME)}</strong>`, rating, 'Entra así en el promedio'].join('<br>');
  }
  const lines = [`<strong>${escapeHtml(row.label)}</strong>`, rating];
  if (Number.isFinite(row.n_polls)) {
    lines.push(escapeHtml(`${fmtNum(row.n_polls)} ${row.n_polls === 1 ? 'sondeo' : 'sondeos'}`));
  }
  return lines.join('<br>');
}

/**
 * Label of a tick of the time axis: the year on 1 January, the day and month otherwise. ECharts places the ticks
 * at local midnight, so the date is read in local time (in UTC, 1 January would read as 31 December).
 *
 * @param {number} ms the tick, in milliseconds
 * @returns {string} the label
 */
export function tickLabel(ms) {
  const date = new Date(ms);
  if (date.getMonth() === 0 && date.getDate() === 1) {
    return String(date.getFullYear());
  }
  const pad = (n) => String(n).padStart(2, '0');
  return fmtDateShort(`${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`);
}

/**
 * ECharts option of the rating path.
 *
 * @param {object[]} points the elections polled, ascending by date: `event_date`, `label`, `kind` (`general` or
 *   `regional`), `n_polls` and `rating` (a point without rating is skipped)
 * @param {object} [opts] `today` (`{event_date, rating, ...}` of the current rating, drawn as a dashed tail from the
 *   last point; none by default) and `colors` (`{general, regional}` of the points; the accent by default)
 * @returns {object} the option: the line (`Rating`, without symbols), one scatter series per kind present
 *   (`Generales`, `Autonómicas`), the tail (`Hoy`) and a legend of the named series but the line
 */
export function ratingPathOption(points, {today = null, colors = {}} = {}) {
  const rated = points.filter((row) => Number.isFinite(row.rating));
  const series = [{
    type: 'line',
    name: LINE_NAME,
    data: rated.map((row) => [row.event_date, row.rating]),
    showSymbol: false,
    color: THEME.markColor,
    lineStyle: {width: 2},
    tooltip: {show: false},
    z: 1,
  }];
  const legend = [];
  for (const kind of Object.keys(KIND_NAMES)) {
    const rows = rated.filter((row) => row.kind === kind);
    if (rows.length) {
      series.push({
        type: 'scatter',
        name: KIND_NAMES[kind],
        color: colors[kind] || THEME.accentColor,
        symbolSize: POINT_SIZE,
        z: 10,
        data: rows.map((row) => ({value: [row.event_date, row.rating], row})),
      });
      legend.push(KIND_NAMES[kind]);
    }
  }
  if (today && Number.isFinite(today.rating)) {
    const last = rated[rated.length - 1];
    const data = last ? [{value: [last.event_date, last.rating], symbol: 'none'}] : [];
    data.push({value: [today.event_date, today.rating], symbol: 'emptyCircle', symbolSize: POINT_SIZE, row: today});
    series.push({
      type: 'line',
      name: TODAY_NAME,
      data,
      color: THEME.markColor,
      lineStyle: {width: 2, type: 'dashed'},
      itemStyle: {borderWidth: 2},
      showSymbol: true,
      z: 10,
    });
    legend.push(TODAY_NAME);
  }
  return {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 16, bottom: 44, containLabel: true},
    legend: {bottom: 0, data: legend, selectedMode: false, itemWidth: 14, itemHeight: 10},
    xAxis: {
      type: 'time',
      axisLabel: {...axisLabelStyle(), formatter: tickLabel},
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      interval: 20,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (x) => fmtNum(x)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(params) {
        const row = params.data && params.data.row;
        return row ? pointTooltip(row) : '';
      },
    },
    series,
  };
}

/**
 * Render the rating path (`ratingPathOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} points see `ratingPathOption`
 * @param {object} [opts] see `ratingPathOption`
 * @returns {object} the ECharts instance
 */
export function renderRatingPath(el, points, opts = {}) {
  return mountChart(el, ratingPathOption(points, opts));
}
