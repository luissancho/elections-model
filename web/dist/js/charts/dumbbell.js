// A dumbbell chart: one row per house and election, a line from its error with the last legal poll (grey dot) to
// its error with the last clandestine one (coloured dot), green when it got closer to the result and red when not
// (the Andorra study, question 1).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

// Closer to the result, farther from it, no change; the dot of the legal poll.
const BETTER = '#2e9e5b';
const WORSE = '#ce3b3b';
const SAME = '#9298ac';
const LEGAL_DOT = '#c3c8d4';
// The axis ends on a multiple of this step and reaches at least this far.
const AXIS_STEP = 0.5;
const AXIS_MIN_MAX = 3.5;
const MIN_HEIGHT = 320;
const ROW_HEIGHT = 22;

/** The colour of a row: how the clandestine error compares with the legal one. */
function rowColor(row) {
  if (row.error_ban < row.error_legal) {
    return BETTER;
  }
  return row.error_ban > row.error_legal ? WORSE : SAME;
}

/** The tooltip text of a row: «{label} · último legal {a} → clandestino {b} (afinó|empeoró {d})». */
function rowText(row) {
  const change = row.error_ban - row.error_legal;
  const verdict = change === 0 ? '' : ` (${change < 0 ? 'afinó' : 'empeoró'} ${fmtNum(Math.abs(change), 2)})`;
  return `${row.label} · último legal ${fmtNum(row.error_legal, 2)} → clandestino ${fmtNum(row.error_ban, 2)}${verdict}`;
}

/**
 * ECharts option of the dumbbell chart: a category axis (inverted: the first row on top) with the labels in the
 * given order and a value axis of errors from zero, whose end is the largest error rounded up to 0.5 (at least 3.5).
 * One `custom` series draws each row as a line and two circles; the colour of the row travels in the datum's
 * `itemStyle`, where `renderItem` reads it.
 *
 * @param {{label: string, error_legal: number, error_ban: number}[]} rows one per house and election
 * @param {object} opts `max` (the end of the value axis, instead of the computed one)
 * @returns {object} the option
 */
export function dumbbellOption(rows, {max} = {}) {
  const largest = Math.max(0, ...rows.flatMap((row) => [row.error_legal, row.error_ban]).filter(Number.isFinite));
  const axisMax = max ?? Math.max(AXIS_MIN_MAX, Math.ceil(largest / AXIS_STEP) * AXIS_STEP);
  return {
    ...baseOption(),
    grid: {left: 150, right: 30, top: 8, bottom: 48},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter: (params) => escapeHtml(rowText(rows[params.dataIndex])),
    },
    xAxis: {
      type: 'value',
      min: 0,
      max: axisMax,
      interval: AXIS_STEP,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => fmtNum(value, 1)},
      name: 'error medio por partido (puntos)',
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: {color: THEME.mutedColor, fontSize: 11},
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map((row) => row.label),
      axisTick: {show: false},
      axisLine: {show: false},
      axisLabel: {color: THEME.textColor, fontSize: 11.5},
    },
    series: [{
      type: 'custom',
      data: rows.map((row) => ({value: [row.error_legal, row.error_ban], itemStyle: {color: rowColor(row)}})),
      renderItem(params, api) {
        const y = api.coord([0, params.dataIndex])[1];
        const x0 = api.coord([api.value(0), 0])[0];
        const x1 = api.coord([api.value(1), 0])[0];
        const color = api.visual('color');
        return {
          type: 'group',
          children: [
            {type: 'line', shape: {x1: x0, y1: y, x2: x1, y2: y}, style: {stroke: color, lineWidth: 2.5}},
            {type: 'circle', shape: {cx: x0, cy: y, r: 5}, style: {fill: LEGAL_DOT}},
            {type: 'circle', shape: {cx: x1, cy: y, r: 6}, style: {fill: color, stroke: '#fff', lineWidth: 1.5}},
          ],
        };
      },
    }],
  };
}

/**
 * Render the dumbbell chart (`dumbbellOption`) in `el`, tall enough for one row per 22 px (320 px at least).
 *
 * @param {HTMLElement} el chart container
 * @param {{label: string, error_legal: number, error_ban: number}[]} rows one per house and election
 * @param {object} opts see `dumbbellOption`
 * @returns {object} the ECharts instance
 */
export function renderDumbbell(el, rows, opts = {}) {
  el.style.height = `${Math.max(MIN_HEIGHT, rows.length * ROW_HEIGHT)}px`;
  return mountChart(el, dumbbellOption(rows, opts));
}
