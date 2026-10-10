// Bullseye of a pollster's errors: one point per election, x = error of the first block, y = of the second.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {fmtDate, fmtNum} from '../format.js';

const RINGS = [2, 5, 10];
const RING_STEPS = 72;

/**
 * Closed polygon approximating the circle of the given radius, in axis units.
 *
 * @param {number} radius radius in points
 * @returns {number[][]} `[x, y]` vertices
 */
function ringPoints(radius) {
  return Array.from({length: RING_STEPS + 1}, (_, i) => {
    const angle = (2 * Math.PI * i) / RING_STEPS;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  });
}

/**
 * Render the bullseye: rings at ±2, ±5 and ±10 points (line series in axis units, so they follow the
 * axes), symmetric axes and one labelled point per election.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows poll-errors rows `{event, event_date, days, <block>: error}`
 * @param {object} opts `blocks` (names of the x and y blocks)
 * @returns {object} the ECharts instance
 */
export function renderBullseye(el, rows, {blocks}) {
  const [xBlock, yBlock] = blocks;
  const points = rows
    .filter((row) => Number.isFinite(row[xBlock]) && Number.isFinite(row[yBlock]))
    .map((row) => ({value: [row[xBlock], row[yBlock]], row}));
  const reach = Math.max(RINGS[RINGS.length - 1], ...points.flatMap((p) => p.value.map(Math.abs)));
  const limit = Math.ceil(reach * 1.1);

  const rings = RINGS.map((radius) => ({
    type: 'line',
    data: ringPoints(radius),
    showSymbol: false,
    silent: true,
    lineStyle: {color: THEME.gridColor, width: 1},
    endLabel: {show: true, formatter: `±${radius}`, color: THEME.markColor},
    tooltip: {show: false},
  }));
  const axis = (name) => ({
    type: 'value',
    name: `${name} (pts)`,
    nameLocation: 'middle',
    nameGap: 28,
    min: -limit,
    max: limit,
    splitLine: {show: false},
    axisLine: {onZero: true},
  });
  const option = {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 16, bottom: 8, containLabel: true},
    xAxis: axis(xBlock),
    yAxis: {...axis(yBlock), nameLocation: 'middle', nameGap: 36},
    tooltip: {
      trigger: 'item',
      formatter(params) {
        const row = params.data && params.data.row;
        if (!row) {
          return '';
        }
        return [
          `<strong>${escapeHtml(row.event)}</strong>`,
          `${escapeHtml(fmtDate(row.event_date))} · ${escapeHtml(fmtNum(row.days))} días antes`,
          `${escapeHtml(xBlock)}: ${escapeHtml(fmtNum(row[xBlock], 1))}`,
          `${escapeHtml(yBlock)}: ${escapeHtml(fmtNum(row[yBlock], 1))}`,
        ].join('<br>');
      },
    },
    series: [
      ...rings,
      {
        type: 'scatter',
        data: points,
        symbolSize: 10,
        color: THEME.markColor,
        label: {show: true, position: 'right', formatter: (params) => params.data.row.event, color: THEME.textColor},
        z: 10,
      },
    ],
  };
  return mountChart(el, option);
}
