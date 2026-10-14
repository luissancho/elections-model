// Bullseye of poll errors: one point per row, x = error of the first block, y = of the second. By default one grey
// labelled point per election (house profile, event detail); with `groupOf`, one coloured series per group (the
// error history colours the polls by election, with a legend).
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
 * Names of the groups of the points, in order of appearance.
 *
 * @param {object[]} points `{value, row}` points
 * @param {Function} groupOf `row => group`
 * @returns {string[]} distinct group names
 */
function groupNames(points, groupOf) {
  return [...new Set(points.map((point) => groupOf(point.row)))];
}

/**
 * Scatter series of the points: one grey series, or one series per group with its colour.
 *
 * @param {object[]} points `{value, row}` points
 * @param {object} opts `groupOf`, `colors`, `labels` and `color` of `renderBullseye`
 * @returns {object[]} ECharts series
 */
function pointSeries(points, {groupOf, colors, labels, color}) {
  const label = {show: labels, position: 'right', formatter: (params) => params.data.row.event, color: THEME.textColor};
  if (!groupOf) {
    // Many houses crowd the centre: a label that would overlap another is left out (the tooltip keeps it).
    return [{type: 'scatter', data: points, symbolSize: 10, color: color || THEME.markColor, label,
             labelLayout: {hideOverlap: true}, z: 10}];
  }
  return groupNames(points, groupOf).map((group) => ({
    type: 'scatter',
    name: group,
    data: points.filter((point) => groupOf(point.row) === group),
    symbolSize: 10,
    color: colors[group] ?? THEME.markColor,
    label,
    z: 10,
  }));
}

/**
 * Tooltip lines of a point of the grouped bullseye: the house, the election (its full `name` when the row has it,
 * else its short code) and its days, the block errors and, when the row has an `href`, the link to the election.
 *
 * @param {object} row poll-errors row
 * @param {string} xBlock first block
 * @param {string} yBlock second block
 * @returns {string[]} HTML lines
 */
function groupedTooltip(row, xBlock, yBlock) {
  const lines = [
    `<strong>${escapeHtml(row.pollster ?? row.event)}</strong>`,
    `${escapeHtml(row.name || row.event)} · ${escapeHtml(fmtDate(row.event_date))} · ${escapeHtml(fmtNum(row.days))} días antes`,
    `${escapeHtml(xBlock)}: ${escapeHtml(fmtNum(row[xBlock], 1))}`,
    `${escapeHtml(yBlock)}: ${escapeHtml(fmtNum(row[yBlock], 1))}`,
  ];
  if (row.href) {
    lines.push(`<a href="${escapeHtml(row.href)}">Ver la elección</a>`);
  }
  return lines;
}

/**
 * Render the bullseye: rings at ±2, ±5 and ±10 points (line series in axis units, so they follow the
 * axes), symmetric axes and one point per row.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows poll-errors rows `{event, event_date, days, <block>: error}` (`name`, the full name of
 *   the election, replaces `event` in the tooltip when present; with `groupOf`, also `pollster` and an optional
 *   `href` for the tooltip)
 * @param {object} opts `blocks` (names of the x and y blocks), `groupOf` (`row => group`: one scatter series
 *   per group, in order of appearance; `null` draws a single series, grey or of `color`), `colors` (`{group: colour}`),
 *   `color` (colour of the single series), `legend` (show the legend of the groups, scrollable), `groupLabel`
 *   (`group => text` of the legend) and `labels` (write `row.event` next to each point)
 * @returns {object} the ECharts instance
 */
export function renderBullseye(el, rows, {
  blocks, groupOf = null, colors = {}, color = null, legend = false, labels = true, groupLabel = (group) => group,
}) {
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
      // A grouped point may carry the link to its election, so the tooltip must let the pointer in.
      enterable: groupOf !== null,
      formatter(params) {
        const row = params.data && params.data.row;
        if (!row) {
          return '';
        }
        if (groupOf) {
          return groupedTooltip(row, xBlock, yBlock).join('<br>');
        }
        return [
          `<strong>${escapeHtml(row.name || row.event)}</strong>`,
          `${escapeHtml(fmtDate(row.event_date))} · ${escapeHtml(fmtNum(row.days))} días antes`,
          `${escapeHtml(xBlock)}: ${escapeHtml(fmtNum(row[xBlock], 1))}`,
          `${escapeHtml(yBlock)}: ${escapeHtml(fmtNum(row[yBlock], 1))}`,
        ].join('<br>');
      },
    },
    series: [...rings, ...pointSeries(points, {groupOf, colors, labels, color})],
  };
  if (legend && groupOf) {
    // Only the groups are named series, so the rings stay out of the legend.
    option.legend = {type: 'scroll', bottom: 0, data: groupNames(points, groupOf), formatter: groupLabel};
    option.grid = {...option.grid, bottom: 40};
  }
  return mountChart(el, option);
}
