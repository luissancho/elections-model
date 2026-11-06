// Bullseye of poll errors: one point per row, x = error of the first block, y = of the second. By default one grey
// labelled point per election (house profile, event detail); with `groupOf`, one coloured series per group (the
// error history colours the polls by election, with a legend).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtDate, fmtNum} from '../format.js';

const RINGS = [2, 5, 10];
const RING_STEPS = 72;
// Size of a point; the series of the groups `onTop` are drawn over the others.
const POINT_SIZE = 8;
const POINT_Z = 10;
const TOP_Z = 12;
// The highlighted point (the model in the detail of an election): a point like the others, over all of them
const HIGHLIGHT_Z = 14;

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
 * Scatter series of the points: one grey series, or one series per group with its colour, each followed by its
 * twin that writes the labels. The twin is an invisible, silent copy of the points (same name, so the legend
 * toggles both): hovering a label must not open the tooltip of its point, and a silent series takes no mouse
 * events. The series of the groups `onTop` get a higher `z`, so their points and labels are drawn over the other
 * groups'.
 *
 * @param {object[]} points `{value, row, itemStyle, label}` points (`itemStyle` and `label` only on a faded one)
 * @param {object} opts `groupOf`, `colors`, `labels`, `labelOf`, `color` and `onTop` of `renderBullseye`
 * @returns {object[]} ECharts series
 */
function pointSeries(points, {groupOf, colors, labels, labelOf, color, onTop}) {
  const label = {show: labels, position: 'right', formatter: (params) => labelOf(params.data.row), color: THEME.textColor};
  const pair = (data, style) => [
    {...style, type: 'scatter', symbolSize: POINT_SIZE, label: {show: false},
     data: data.map(({value, row, itemStyle}) => (itemStyle ? {value, row, itemStyle} : {value, row}))},
    // Many houses crowd the centre: a label that would overlap another is left out (the tooltip keeps it).
    {...style, type: 'scatter', symbolSize: POINT_SIZE, itemStyle: {opacity: 0}, silent: true, tooltip: {show: false},
     // The labels would inherit the invisible points' opacity: they get their own.
     label: {...label, opacity: 1}, labelLayout: {hideOverlap: true},
     data: data.map(({value, row, label: faded}) => (faded ? {value, row, label: faded} : {value, row}))},
  ];
  if (!groupOf) {
    return pair(points, {color: color || THEME.markColor, z: POINT_Z});
  }
  return groupNames(points, groupOf).flatMap((group) => pair(
    points.filter((point) => groupOf(point.row) === group),
    {name: group, color: colors[group] ?? THEME.markColor, z: onTop && onTop(group) ? TOP_Z : POINT_Z},
  ));
}

/**
 * A hex colour with an alpha, as `rgba()`.
 *
 * @param {string} hex `#rrggbb`
 * @param {number} alpha opacity, 0 to 1
 * @returns {string} the rgba colour
 */
function withAlpha(hex, alpha) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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
    `<strong>${escapeHtml(row.pollster ?? row.name ?? row.event)}</strong>`,
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
 * Silent line series of the rings (±2, ±5 and ±10 points by default) and of the zero cross, in axis units so they
 * follow the axes.
 *
 * @param {number} limit extent of both axes
 * @param {number[]} radii radii of the rings, in points
 * @returns {object[]} ECharts line series
 */
function guideSeries(limit, radii = RINGS) {
  const rings = radii.map((radius) => ({
    type: 'line',
    data: ringPoints(radius),
    showSymbol: false,
    silent: true,
    lineStyle: {color: THEME.axisColor, width: 1},
    endLabel: {show: true, formatter: `±${radius}`, color: THEME.markColor},
    tooltip: {show: false},
  }));
  const cross = [[[-limit, 0], [limit, 0]], [[0, -limit], [0, limit]]].map((data) => ({
    type: 'line',
    data,
    showSymbol: false,
    silent: true,
    lineStyle: {color: THEME.axisColor, width: 1},
    tooltip: {show: false},
  }));
  return [...rings, ...cross];
}

/**
 * ECharts option of the bullseye: the rings (±2, ±5 and ±10 points by default) and the zero cross as line series,
 * symmetric axes at the edges of the grid, each named after its block, and one point per row (with its label on
 * a silent twin series, so only the points open the tooltip).
 *
 * @param {object[]} rows poll-errors rows (see `renderBullseye`)
 * @param {object} opts options of `renderBullseye`
 * @returns {object} the option
 */
export function bullseyeOption(rows, {
  blocks, groupOf = null, colors = {}, color = null, legend = false, labels = true, groupLabel = (group) => group,
  labelOf = (row) => row.event, alphaOf = null, onTop = null, highlight = null, highlights = [], interval = null,
  rings = RINGS,
}) {
  const [xBlock, yBlock] = blocks;
  const styled = (row) => {
    if (!alphaOf) {
      return {};
    }
    const alpha = Math.min(1, Math.max(0, alphaOf(row)));
    return {itemStyle: {opacity: alpha}, label: {color: withAlpha(THEME.textColor, alpha)}};
  };
  const points = rows
    .filter((row) => Number.isFinite(row[xBlock]) && Number.isFinite(row[yBlock]))
    .map((row) => ({value: [row[xBlock], row[yBlock]], row, ...styled(row)}));
  const marked = highlight && Number.isFinite(highlight.row[xBlock]) && Number.isFinite(highlight.row[yBlock])
    ? [highlight.row[xBlock], highlight.row[yBlock]] : null;
  const drawn = highlights.filter(({row}) => Number.isFinite(row[xBlock]) && Number.isFinite(row[yBlock]));
  // The largest ring is always in view, whatever the points.
  const reach = Math.max(Math.max(...rings), ...points.flatMap((p) => p.value.map(Math.abs)),
    ...(marked || []).map(Math.abs), ...drawn.flatMap(({row}) => [row[xBlock], row[yBlock]].map(Math.abs)));
  // With a fixed tick `interval` the extent is the first multiple of it beyond the reach, so the last tick is round.
  const limit = interval ? (Math.floor(reach / interval) + 1) * interval : Math.ceil(reach * 1.1);

  const axis = (name) => ({
    type: 'value',
    name,
    nameLocation: 'middle',
    nameTextStyle: {fontWeight: 600},
    min: -limit,
    max: limit,
    splitLine: {show: false},
    axisLine: {onZero: false, show: true},
    axisTick: {show: false},
    // The extent is not a round number: its labels would crowd the last round tick.
    axisLabel: {...axisLabelStyle(), showMinLabel: false, showMaxLabel: false},
    ...(interval ? {interval} : {}),
  });
  const option = {
    ...baseOption(),
    grid: {left: 20, right: 24, top: 16, bottom: 20, containLabel: true},
    xAxis: {...axis(xBlock), nameGap: 26},
    yAxis: {...axis(yBlock), nameGap: 34},
    tooltip: {
      ...tooltipStyle(),
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
    series: [...guideSeries(limit, rings), ...pointSeries(points, {groupOf, colors, labels, labelOf, color, onTop})],
  };
  if (marked) {
    // no label: the colour and the `z` over every other point (POINT_Z, TOP_Z) are enough; the tooltip names it
    option.series.push({
      id: 'highlight', type: 'scatter', symbolSize: POINT_SIZE, z: HIGHLIGHT_Z, color: highlight.color || THEME.textColor,
      label: {show: false},
      data: [{value: marked, row: highlight.row}],
    });
  }
  drawn.forEach(({row, color: markColor, symbol = 'circle', label = null}, index) => {
    option.series.push({
      id: `highlight-${index}`,
      type: 'scatter',
      symbol,
      symbolSize: symbol === 'diamond' ? POINT_SIZE + 4 : POINT_SIZE,
      z: HIGHLIGHT_Z,
      color: markColor,
      label: label === null ? {show: false}
        : {show: true, position: 'right', formatter: label, fontWeight: 'bold', color: markColor, fontSize: 11},
      data: [{value: [row[xBlock], row[yBlock]], row}],
    });
  });
  if (legend && groupOf) {
    // Only the groups are named series, so the rings stay out of the legend.
    option.legend = {type: 'scroll', bottom: 0, data: groupNames(points, groupOf), formatter: groupLabel};
    option.grid = {...option.grid, bottom: 52};
  }
  return option;
}

/**
 * Render the bullseye (`bullseyeOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows poll-errors rows `{event, event_date, days, <block>: error}` (`name`, the full name of
 *   the election, replaces `event` in the tooltip when present; with `groupOf`, also `pollster` and an optional
 *   `href` for the tooltip)
 * @param {object} opts `blocks` (names of the x and y blocks), `groupOf` (`row => group`: one scatter series
 *   per group, in order of appearance; `null` draws a single series, grey or of `color`), `colors` (`{group: colour}`),
 *   `color` (colour of the single series), `legend` (show the legend of the groups, scrollable), `groupLabel`
 *   (`group => text` of the legend), `labels` (write a label next to each point), `labelOf` (`row => text` of
 *   that label, `row.event` by default), `alphaOf` (`row => opacity`, 0 to 1: the point and its label fade;
 *   `null` keeps them opaque) and `onTop` (`group => boolean`: the points of those groups are drawn over the
 *   other groups', e.g. the general elections over the regional ones; `null` keeps the order of the groups) and
 *   `highlight` (`{row, color}`: one more point, of the size of the others and in its `color` (the text colour by
 *   default), without a label, drawn over every other point and counted in the extent of the axes; its tooltip
 *   names `row`; nothing without coordinates), `highlights` (array of `{row, color, symbol = 'circle', label =
 *   null}`: several such points, each a series `highlight-<index>`, a diamond slightly larger, written to its right
 *   in bold when `label` is given), `interval` (step of both axes: their extent becomes the first multiple of
 *   it beyond the farthest point, with the outer tick unlabelled) and `rings` (radii of the rings drawn, in points,
 *   `[2, 5, 10]` by default; the largest is the least extent of the axes)
 * @returns {object} the ECharts instance
 */
export function renderBullseye(el, rows, opts) {
  return mountChart(el, bullseyeOption(rows, opts));
}
