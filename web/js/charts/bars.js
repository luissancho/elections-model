// Horizontal bars with an interval line per row (vote shares or seats).
import {mountChart, THEME} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';

const INTERVAL_COLOR = '#333';

/** Escape text for the HTML tooltips of ECharts. */
function escapeHtml(text) {
  return window.echarts.format.encodeHTML(String(text));
}

/**
 * Round `x` up to a readable axis limit: a multiple of a fifth of its power of ten.
 *
 * @param {number} x largest value the axis must show
 * @returns {number} the axis maximum
 */
export function niceMax(x) {
  if (!Number.isFinite(x) || x <= 0) {
    return 1;
  }
  const step = 10 ** Math.floor(Math.log10(x)) / 5;
  return Number((Math.ceil(x / step) * step).toPrecision(12));
}

/**
 * Normalise the rows to `{name, value, lo, hi}` with the given field names (missing → null).
 *
 * @param {object[]} rows input rows with a `name`
 * @param {{value: string, lo: string, hi: string}} fields field names of the value and interval
 * @returns {object[]} the normalised rows, in the same order
 */
export function barRows(rows, {value, lo, hi}) {
  const pick = (row, key) => (row[key] === undefined || row[key] === null ? null : row[key]);
  return rows.map((row) => ({
    name: row.name,
    value: pick(row, value),
    lo: pick(row, lo),
    hi: pick(row, hi),
  }));
}

/**
 * Render horizontal bars, one per row and in the order of `rows`, with the interval `[lo, hi]` drawn as
 * a capped line and the value label after the end of the bar (past the interval cap so they never
 * overlap).
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows rows with a `name` and the value/interval fields
 * @param {object} opts `value`, `lo`, `hi` (field names), `colors` ({name: colour}),
 *   `fullnames` ({name: full name}, tooltip), `formatter` (number → text), `majority` (vertical mark)
 * @returns {object} the ECharts instance
 */
export function renderBars(el, rows, {
  value = 'pct', lo = 'lo', hi = 'hi', colors = {}, fullnames = {}, formatter = String, majority = null,
} = {}) {
  const data = barRows(rows, {value, lo, hi});
  const top = Math.max(0, ...data.map((row) => Math.max(row.value ?? 0, row.hi ?? 0)), majority ?? 0);

  const bar = {
    type: 'bar',
    barWidth: '60%',
    data: data.map((row) => ({value: row.value, itemStyle: {color: colors[row.name] || OTHERS_COLOR}})),
  };
  if (majority !== null && majority !== undefined) {
    bar.markLine = {
      silent: true,
      symbol: 'none',
      lineStyle: {type: 'dashed', color: '#555'},
      label: {formatter: `Mayoría: ${formatter(majority)}`, position: 'end'},
      data: [{xAxis: majority}],
    };
  }

  const interval = {
    type: 'custom',
    z: 10,
    encode: {x: [1, 2], y: 0},
    data: data.map((row, i) => [i, row.lo, row.hi, row.value]),
    renderItem(params, api) {
      const index = api.value(0);
      const low = api.value(1);
      const high = api.value(2);
      const val = api.value(3);
      const children = [];
      let labelAt = Number.isFinite(val) ? val : 0;
      const style = {stroke: INTERVAL_COLOR, lineWidth: 1.5};
      if (Number.isFinite(low) && Number.isFinite(high)) {
        const start = api.coord([low, index]);
        const end = api.coord([high, index]);
        const cap = api.size([0, 1])[1] * 0.2;
        children.push(
          {type: 'line', shape: {x1: start[0], y1: start[1], x2: end[0], y2: end[1]}, style},
          {type: 'line', shape: {x1: start[0], y1: start[1] - cap, x2: start[0], y2: start[1] + cap}, style},
          {type: 'line', shape: {x1: end[0], y1: end[1] - cap, x2: end[0], y2: end[1] + cap}, style},
        );
        labelAt = Math.max(labelAt, high);
      }
      if (Number.isFinite(val)) {
        const point = api.coord([labelAt, index]);
        children.push({
          type: 'text',
          x: point[0] + 6,
          y: point[1],
          style: {
            text: formatter(val),
            align: 'left',
            verticalAlign: 'middle',
            fill: THEME.textColor,
            font: `12px ${THEME.fontFamily}`,
          },
        });
      }
      return {type: 'group', children};
    },
  };

  const option = {
    animationDuration: THEME.animationDuration,
    textStyle: {fontFamily: THEME.fontFamily, color: THEME.textColor},
    grid: {left: 8, right: 16, top: 24, bottom: 8, containLabel: true},
    xAxis: {
      type: 'value',
      min: 0,
      max: niceMax(top * 1.15),
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (x) => formatter(x)},
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: data.map((row) => row.name),
      axisTick: {show: false},
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter(params) {
        const row = data[params[0].dataIndex];
        if (!row) {
          return '';
        }
        const lines = [`<strong>${escapeHtml(fullnames[row.name] || row.name)}</strong>`, escapeHtml(formatter(row.value))];
        if (row.lo !== null && row.hi !== null) {
          lines.push(`Intervalo: ${escapeHtml(formatter(row.lo))} – ${escapeHtml(formatter(row.hi))}`);
        }
        return lines.join('<br>');
      },
    },
    series: [bar, interval],
  };
  return mountChart(el, option);
}
