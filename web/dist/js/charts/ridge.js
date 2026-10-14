// Ridge chart of the seat distributions: one row per party, the density filled with the party colour, a thick
// mark at the median, thin ones at the ends of the interval and a dashed line at the majority, all over the
// same seat axis, as `Simulator.plot_dist_kde` draws it in the notebooks.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {fmtInt, fmtProb} from '../format.js';

const TOP = 12;
const BOTTOM = 34;
const LEFT = 118;
const RIGHT = 20;
const GAP = 6;

/**
 * Height the container needs for `n` rows.
 *
 * @param {number} n rows
 * @param {number} rowHeight height of each row, in px
 * @returns {number} the height, in px
 */
export function ridgeHeight(n, rowHeight) {
  return TOP + n * rowHeight + BOTTOM;
}

/**
 * Render the ridges of `rows` (from `kde.js` `ridgeData`) in `el`, whose height is set to fit them.
 *
 * @param {HTMLElement} el chart container
 * @param {{grid: number[], rows: object[]}} data the shared grid and one row per party
 * @param {object} opts `colors` ({name: colour}), `xMax` (last seat of the axis), `majority` (seats of the
 *   absolute majority; the dashed line is drawn when it lies within the axis), `rowHeight` (px per row, 56 by
 *   default), `headline` ({name: seats} shown in the row label instead of the median, so the figure matches the
 *   table) and `fullnames` ({name: full name}, tooltip)
 * @returns {object} the ECharts instance
 */
export function renderRidge(el, {grid, rows}, {colors = {}, xMax, majority = null, rowHeight = 56, headline = {},
  fullnames = {}} = {}) {
  const last = rows.length - 1;
  el.style.height = `${ridgeHeight(rows.length, rowHeight)}px`;
  const showMajority = majority !== null && majority !== undefined && majority > 0 && majority <= xMax;
  // Narrow containers (phones) give the labels less room
  const narrow = el.clientWidth > 0 && el.clientWidth < 600;
  const left = narrow ? 84 : LEFT;

  const grids = rows.map((row, index) => ({
    top: TOP + index * rowHeight,
    height: rowHeight - GAP,
    left,
    right: RIGHT,
  }));
  const xAxes = rows.map((row, index) => ({
    type: 'value',
    gridIndex: index,
    min: 0,
    max: xMax,
    axisLine: {show: index === last, lineStyle: {color: THEME.gridColor}},
    axisTick: {show: index === last},
    axisLabel: {show: index === last, formatter: (value) => fmtInt(value)},
    splitLine: {show: false},
  }));
  const yAxes = rows.map((row, index) => ({
    type: 'value',
    gridIndex: index,
    min: 0,
    max: 1.08,
    show: false,
  }));
  const series = rows.map((row, index) => {
    const color = colors[row.name] || THEME.barColor;
    const marks = [
      {xAxis: row.median, lineStyle: {color, width: 2, type: 'solid'}},
      {xAxis: row.lo, lineStyle: {color, width: 1, type: 'solid', opacity: 0.7}},
      {xAxis: row.hi, lineStyle: {color, width: 1, type: 'solid', opacity: 0.7}},
    ];
    if (showMajority) {
      marks.push({
        xAxis: majority,
        lineStyle: {color: THEME.markColor, width: 1, type: 'dashed'},
        label: index === 0 ? {show: true, position: 'start', formatter: `Mayoría ${fmtInt(majority)}`,
          color: THEME.textColor, fontSize: 11} : {show: false},
      });
    }
    return {
      name: row.name,
      type: 'line',
      xAxisIndex: index,
      yAxisIndex: index,
      showSymbol: false,
      smooth: false,
      lineStyle: {color, width: 1.5},
      areaStyle: {color, opacity: 0.35},
      data: grid.map((x, i) => [x, row.y[i]]),
      markLine: {silent: true, symbol: 'none', animation: false, data: marks, label: {show: false}},
      z: 2,
    };
  });
  const labels = rows.map((row, index) => ({
    type: 'text',
    left: 8,
    top: TOP + index * rowHeight + (rowHeight - GAP) / 2 - 15,
    style: {
      text: `${row.name}\n${fmtInt(headline[row.name] ?? row.median)} (${fmtInt(row.lo)}–${fmtInt(row.hi)})`,
      fill: THEME.textColor,
      font: `600 ${narrow ? 11 : 13}px ${THEME.fontFamily}`,
      lineHeight: 16,
    },
  }));

  const option = {
    ...baseOption(),
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    graphic: labels,
    axisPointer: {link: [{xAxisIndex: 'all'}], label: {show: false}},
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'line', lineStyle: {color: THEME.markColor}},
      formatter(params) {
        const first = Array.isArray(params) ? params[0] : params;
        if (!first) {
          return '';
        }
        const row = rows[first.seriesIndex];
        const seats = Math.round(first.axisValue);
        const share = row.pmf[seats] || 0;
        const name = fullnames[row.name] || row.name;
        return `<strong>${escapeHtml(name)}</strong><br>${escapeHtml(`${fmtInt(seats)} escaños en el ${fmtProb(share)} de las simulaciones`)}`;
      },
    },
    series,
  };
  return mountChart(el, option);
}
