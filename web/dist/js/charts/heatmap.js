// Heat map of the house effects: one row per pollster, one column per party or block.
import {baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

const NEGATIVE_COLOR = '#c0392b';
const POSITIVE_COLOR = '#2c6fbb';
const ZERO_COLOR = '#ffffff';
const ROW_HEIGHT = 24;
const EXTRA_HEIGHT = 80;
const MIN_LABEL_WIDTH = 40;

/**
 * Render the house-effects heat map. The scale is diverging and symmetric around 0 (red negative, white
 * zero, blue positive); the container height grows with the number of rows. The values are written in the
 * cells only when each column gets more than `MIN_LABEL_WIDTH` pixels (not on phones); the tooltip always has them.
 *
 * @param {HTMLElement} el chart container
 * @param {{columns: string[], rows: object[]}} grid `rows` as `{pollster, cells: [{cycle, hist, trend}]}`
 * @param {object} opts `formatter` (value → text, default 1 decimal)
 * @returns {object} the ECharts instance
 */
export function renderHeatmap(el, {columns, rows}, {formatter = (x) => fmtNum(x, 1)} = {}) {
  el.style.height = `${rows.length * ROW_HEIGHT + EXTRA_HEIGHT}px`;
  const showLabels = el.clientWidth / Math.max(columns.length, 1) > MIN_LABEL_WIDTH;
  const data = [];
  let top = 0;
  rows.forEach((row, y) => {
    row.cells.forEach((cell, x) => {
      if (cell && Number.isFinite(cell.cycle)) {
        data.push([x, y, cell.cycle]);
        top = Math.max(top, Math.abs(cell.cycle));
      }
    });
  });
  top = top || 1;

  const option = {
    ...baseOption(),
    grid: {left: 8, right: 16, top: 32, bottom: 48, containLabel: true},
    xAxis: {type: 'category', data: columns, position: 'top', splitArea: {show: true}, axisTick: {show: false}},
    yAxis: {type: 'category', data: rows.map((row) => row.pollster), inverse: true, axisTick: {show: false}},
    visualMap: {
      min: -top,
      max: top,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: {color: [NEGATIVE_COLOR, ZERO_COLOR, POSITIVE_COLOR]},
      formatter: (x) => formatter(x),
    },
    tooltip: {
      ...tooltipStyle(),
      formatter(params) {
        const cell = rows[params.value[1]].cells[params.value[0]];
        return `<strong>${escapeHtml(rows[params.value[1]].pollster)} · ${escapeHtml(columns[params.value[0]])}</strong><br>`
          + `Ciclo ${escapeHtml(formatter(cell.cycle))} (historia ${escapeHtml(formatter(cell.hist))}, `
          + `tendencia ${escapeHtml(formatter(cell.trend))}/año)`;
      },
    },
    series: [{
      type: 'heatmap',
      data,
      label: {show: showLabels, formatter: (params) => formatter(params.value[2]), color: THEME.textColor},
      emphasis: {itemStyle: {borderColor: THEME.markColor, borderWidth: 1}},
    }],
  };
  return mountChart(el, option);
}
