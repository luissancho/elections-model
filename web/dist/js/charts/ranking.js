// Ranking of the pollsters by their lean on the left block (the CIS bias study): x is the share of elections in
// which the house overestimated the left, y the mean error on the block; the unbiased point sits at (50, 0).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

const CIS_COLOR = '#ce3b3b';
const PACK_COLOR = '#c7cdd8';
const GUIDE_COLOR = 'rgba(0,0,0,.3)';
// Margins of the box around the pack of houses: share points on x, error points on y.
const BOX_X = 4;
const BOX_Y = 0.5;

/**
 * Silent dashed line between two points.
 *
 * @param {number[][]} data the two ends
 * @returns {object} line series
 */
function guide(data) {
  return {
    type: 'line',
    data,
    showSymbol: false,
    silent: true,
    tooltip: {show: false},
    lineStyle: {color: GUIDE_COLOR, type: 'dashed', width: 1},
  };
}

/**
 * Text of the tooltip of a pollster.
 *
 * @param {Array} value `[x, y, name, k, n]`
 * @returns {string} the HTML
 */
function tooltipText([x, y, name, k, n]) {
  const lean = `${y > 0 ? '+' : ''}${fmtNum(y, 1)}`;
  return `<strong>${escapeHtml(name)}</strong><br>Sobrestimó a la Izquierda en ${escapeHtml(fmtNum(k))} de `
    + `${escapeHtml(fmtNum(n))} elecciones (${escapeHtml(fmtNum(x))}%) · ${escapeHtml(lean)} puntos de media`;
}

/**
 * ECharts option of the ranking: the guide lines at x 50 and y 0, the «Sin sesgo» point, the grey pack of houses
 * inside a dashed box and the highlighted pollster in red with its figures written beside it.
 *
 * @param {{x: number, y: number, name: string, k: number, n: number, cis?: boolean}[]} rows one per pollster
 * @param {object} opts `pollster` (name of the highlighted house; the rows flagged `cis` are highlighted too),
 *   `xMax`, `yMin`, `yMax` and `yInterval` (limits and step of the axes)
 * @returns {object} the option
 */
export function rankingOption(rows, {pollster = 'CIS', xMax = 100, yMin = -2, yMax = 10, yInterval = 2} = {}) {
  const isPollster = (row) => row.cis || row.name === pollster;
  const value = (row) => [row.x, row.y, row.name, row.k, row.n];
  const pack = rows.filter((row) => !isPollster(row));
  const xs = pack.map((row) => row.x);
  const ys = pack.map((row) => row.y);
  const area = pack.length
    ? [[{xAxis: Math.min(...xs) - BOX_X, yAxis: Math.min(...ys) - BOX_Y},
      {xAxis: Math.max(...xs) + BOX_X, yAxis: Math.max(...ys) + BOX_Y}]]
    : [];
  const centre = {
    type: 'scatter',
    data: [[xMax / 2, 0]],
    symbol: 'circle',
    symbolSize: 8,
    silent: true,
    itemStyle: {color: '#fff', borderColor: THEME.textColor, borderWidth: 1.5},
    label: {
      show: true, formatter: 'Sin sesgo', position: 'bottom', offset: [-34, 4], fontSize: 11, fontWeight: 600,
      color: THEME.textColor,
    },
    tooltip: {show: false},
  };
  const others = {
    type: 'scatter',
    symbolSize: 11,
    itemStyle: {color: PACK_COLOR, borderColor: THEME.mutedColor, borderWidth: 1},
    label: {show: true, position: 'right', fontSize: 10.5, color: THEME.mutedColor, formatter: (params) => params.value[2]},
    labelLayout: {hideOverlap: true},
    markArea: {
      silent: true,
      itemStyle: {color: 'rgba(29,79,134,.06)', borderColor: 'rgba(29,79,134,.35)', borderWidth: 1, borderType: 'dashed'},
      label: {show: false},
      data: area,
    },
    data: pack.map((row) => ({value: value(row)})),
  };
  const marked = {
    type: 'scatter',
    symbolSize: 18,
    itemStyle: {color: CIS_COLOR},
    label: {
      show: true,
      position: 'left',
      fontSize: 12,
      fontWeight: 800,
      color: CIS_COLOR,
      formatter: ({value: [, y, name, k, n]}) => `${name} · ${k} de ${n} · ${fmtNum(y, 1)} puntos`,
    },
    data: rows.filter(isPollster).map((row) => ({value: value(row)})),
  };
  return {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 16, bottom: 30, containLabel: true},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter: (params) => (params.seriesType === 'scatter' && params.value.length > 2 ? tooltipText(params.value) : ''),
    },
    xAxis: {
      type: 'value',
      min: 0,
      max: xMax,
      name: 'Elecciones en las que sobreestimó a la izquierda',
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: {color: THEME.mutedColor, fontSize: 11},
      axisLabel: {...axisLabelStyle(), formatter: '{value}%'},
      splitLine: {lineStyle: {color: THEME.gridColor}},
    },
    yAxis: {
      type: 'value',
      min: yMin,
      max: yMax,
      interval: yInterval,
      name: 'Sesgo de bloques medio, puntos (+ izquierda, − derecha)',
      nameTextStyle: {color: THEME.mutedColor, fontSize: 11, align: 'left'},
      axisLabel: {...axisLabelStyle(), formatter: (v) => fmtNum(v)},
      splitLine: {lineStyle: {color: THEME.gridColor}},
    },
    series: [guide([[xMax / 2, yMin], [xMax / 2, yMax]]), guide([[0, 0], [xMax, 0]]), centre, others, marked],
  };
}

/**
 * Render the ranking (`rankingOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows see `rankingOption`
 * @param {object} opts see `rankingOption`
 * @returns {object} the ECharts instance
 */
export function renderRanking(el, rows, opts) {
  return mountChart(el, rankingOption(rows, opts));
}
