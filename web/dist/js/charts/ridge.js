// Ridge chart of the seat distributions: one row per party or block, each over its own seat range, the density
// filled with the row's colour, a thick mark at the median, thin ones at the ends of the interval and a dashed
// line at the majority when it falls within the row, as `Simulator.plot_dist_kde` draws it in the notebooks but
// with the detail of every row.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtInt, fmtProb, fmtRange} from '../format.js';

const TOP = 10;
const BOTTOM = 10;
const LEFT = 150;
const RIGHT = 20;
const AXIS = 18;
const GAP = 8;

/**
 * Height the container needs for `n` rows.
 *
 * @param {number} n rows
 * @param {number} rowHeight height of each row, in px (curve plus its axis)
 * @returns {number} the height, in px
 */
export function ridgeHeight(n, rowHeight) {
  return TOP + n * rowHeight + BOTTOM;
}

/**
 * Render `rows` (from `kde.js`) in `el`, whose height is set to fit them.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows one row per party or block (`ridgeRows`)
 * @param {object} opts `colors` ({name: colour}), `majority` (seats of the absolute majority; the dashed line is
 *   drawn in the rows whose range holds it, labelled on the first), `rowHeight` (px per row, 144 by default),
 *   `fullnames` ({name: full name}, tooltip) and `guideEvery` (seats between the faint dashed vertical guides;
 *   10 by default, 20 on narrow containers; the labelled axis shows a number at each). The row label reads the
 *   name, `Mediana: …` and `Rango 95 %: …`, with the name and the figures in bold.
 *   When every row shares the same range, only the last row shows the axis labels; otherwise each row shows its own.
 * @returns {object} the ECharts instance
 */
export function renderRidge(el, rows, {colors = {}, majority = null, rowHeight = 144, fullnames = {}, guideEvery = null} = {}) {
  el.style.height = `${ridgeHeight(rows.length, rowHeight)}px`;
  // Narrow containers (phones) give the labels less room
  const narrow = el.clientWidth > 0 && el.clientWidth < 600;
  const left = narrow ? 108 : LEFT;
  const every = guideEvery || (narrow ? 20 : 10);
  const curveHeight = rowHeight - AXIS - GAP;
  const last = rows.length - 1;
  const shared = rows.length > 0 && rows.every((row) => row.x0 === rows[0].x0 && row.x1 === rows[0].x1);

  const grids = rows.map((row, index) => ({
    top: TOP + index * rowHeight,
    height: curveHeight,
    left,
    right: RIGHT,
  }));
  const xAxes = rows.map((row, index) => ({
    type: 'value',
    gridIndex: index,
    min: row.x0,
    max: row.x1,
    // A tick (and, on the labelled row, a number) at every guide
    interval: every,
    axisLine: {show: true, lineStyle: {color: THEME.axisColor}},
    axisTick: {show: !shared || index === last, lineStyle: {color: THEME.axisColor}},
    axisLabel: {...axisLabelStyle(), show: !shared || index === last, fontSize: 10, color: THEME.markColor, formatter: (value) => fmtInt(value)},
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
    const marks = [];
    // Faint dashed guides every `every` seats, behind the marks of the row
    for (let x = every; x < row.x1; x += every) {
      if (x > row.x0) {
        marks.push({xAxis: x, lineStyle: {color: THEME.gridColor, width: 1, type: 'dashed', opacity: 0.8}});
      }
    }
    marks.push(
      {xAxis: row.median, lineStyle: {color, width: 2, type: 'solid'}},
      {xAxis: row.lo, lineStyle: {color, width: 1, type: 'solid', opacity: 0.7}},
      {xAxis: row.hi, lineStyle: {color, width: 1, type: 'solid', opacity: 0.7}},
    );
    const withMajority = majority !== null && majority !== undefined && majority >= row.x0 && majority <= row.x1;
    if (withMajority) {
      marks.push({
        xAxis: majority,
        lineStyle: {color: THEME.markColor, width: 1, type: 'dashed'},
        label: {show: index === 0, position: 'insideEndTop', formatter: `Mayoría ${fmtInt(majority)}`,
          color: THEME.textColor, fontSize: 11},
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
      data: row.grid.map((x, i) => [x, row.y[i]]),
      markLine: {silent: true, symbol: 'none', animation: false, data: marks, label: {show: false}},
      z: 2,
    };
  });
  const labels = rows.map((row, index) => ({
    type: 'text',
    left: 8,
    top: TOP + index * rowHeight + curveHeight / 2 - 24,
    style: {
      // Rich text: the name, the median and the range in bold, the captions in regular weight
      text: `{b|${row.name}}\nMediana: {b|${fmtInt(row.median)}}\nRango 95 %: {b|${fmtRange(row.lo, row.hi, 0)}}`,
      fill: THEME.textColor,
      fontSize: narrow ? 10.5 : 12,
      fontFamily: THEME.fontFamily,
      lineHeight: 16,
      rich: {b: {fontWeight: 600, fontSize: narrow ? 10.5 : 12, fontFamily: THEME.fontFamily, fill: THEME.textColor}},
    },
  }));

  const option = {
    ...baseOption(),
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    graphic: labels,
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      axisPointer: {type: 'line', lineStyle: {color: THEME.markColor}, label: {show: false}},
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
