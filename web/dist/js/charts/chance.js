// What chance would give, against what was observed (question 3 of the herd study): the histogram of the change of
// the separation between the rest of the cycle and the last three weeks over the shuffles of the null-hypothesis
// test, one bar per bin of the `chance` part, and a vertical line at the observed change.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtCount, fmtNum} from '../format.js';

const MINUS = '−';
const AXIS_NAME = 'cambio de la separación entre el resto del ciclo y las tres últimas semanas';
// Bins of margin on each side of the bins and the observed value, and the bins the ends of the axis round out to
const MARGIN_BINS = 2;
const ROUND_BINS = 5;
const LABEL_BACKGROUND = 'rgba(255,255,255,.85)';
// Below this width of the container (a phone) the name of the axis goes in two lines: ECharts cuts a long axis name
// instead of wrapping it
const NARROW = 480;

/**
 * A percentage as the page writes it: «−26 %», «+10 %», «0 %» (the typographic minus, a space before the sign).
 *
 * @param {number} value the percentage, already in the 0-100 scale
 * @param {number} digits decimals
 * @returns {string} the label
 */
export function pct(value, digits = 0) {
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) {
    return `${fmtNum(0, digits)} %`;
  }
  return `${rounded < 0 ? MINUS : '+'}${fmtNum(Math.abs(rounded), digits)} %`;
}

/**
 * The bins in % (`[lo, hi, count]`), rounded so that the borders of contiguous bins meet exactly.
 *
 * @param {object[]} bins `{lo, hi, count}` as fractions
 * @returns {number[][]} one row per bin
 */
function binsInPercent(bins) {
  const toPct = (x) => Math.round(x * 100 * 1e6) / 1e6;
  return bins.map((bin) => [toPct(bin.lo), toPct(bin.hi), bin.count]);
}

/**
 * ECharts option of the chart: a value axis in % that covers the bins and the observed change with a margin of two
 * bins, a custom series with one bar per bin spanning exactly its width, and a `markLine` at the observed change.
 *
 * @param {object} chance the `chance` block of the part: `change` (a fraction) and `null.bins` (`{lo, hi, count}`
 *   as fractions) and `null.bin_width`
 * @param {object} opts `color` (of the bars), `observed` (of the line and its label) and `narrow` (the name of the
 *   axis in two lines)
 * @returns {object} the option
 */
export function chanceOption(chance = {}, {color = THEME.mutedColor, observed = THEME.accentColor,
                                           narrow = false} = {}) {
  const block = chance || {};
  const nul = block.null || {};
  const rows = binsInPercent(nul.bins || []);
  const width = (nul.bin_width || 0.01) * 100;
  const change = typeof block.change === 'number' ? block.change * 100 : null;
  const edges = rows.flatMap((row) => [row[0], row[1]]);
  if (change !== null) {
    edges.push(change);
  }
  const low = edges.length ? Math.min(...edges) : -width;
  const high = edges.length ? Math.max(...edges) : width;
  // the margin, then out to five bins so that the ends of the axis fall on round figures
  const step = ROUND_BINS * width;
  const min = Math.floor((low - MARGIN_BINS * width) / step) * step;
  const max = Math.ceil((high + MARGIN_BINS * width) / step) * step;
  // the borders of the bins in the tooltip: whole points, or one decimal for bins narrower than a point
  const digits = width < 1 ? 1 : 0;
  const bars = {
    type: 'custom',
    name: 'barajas',
    encode: {x: [0, 1], y: 2, tooltip: 2},
    data: rows.map((row) => ({value: row})),
    itemStyle: {color},
    renderItem(params, api) {
      const start = api.coord([api.value(0), api.value(2)]);
      const end = api.coord([api.value(1), 0]);
      const style = api.style();
      return {
        type: 'rect',
        shape: {x: start[0], y: start[1], width: end[0] - start[0], height: end[1] - start[1]},
        // a hairline between bars, white, so that neighbouring bins read apart
        style: {...style, fill: color, stroke: '#fff', lineWidth: 0.5},
      };
    },
  };
  if (change !== null) {
    bars.markLine = {
      silent: true,
      symbol: 'none',
      data: [{xAxis: change}],
      lineStyle: {color: observed, width: 2, type: 'solid'},
      label: {
        // a plain string (no `{…}` templates): it survives the serialization of the option
        formatter: `observado ${pct(change)}`,
        // above the top of the line, level: it lies far out on the left, where the bars leave room
        position: 'end',
        fontSize: 11,
        fontWeight: 700,
        color: observed,
        backgroundColor: LABEL_BACKGROUND,
        padding: [1, 3],
      },
    };
  }
  return {
    ...baseOption(),
    grid: {left: 58, right: 16, top: 28, bottom: 58},
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(params) {
        const [lo, hi, count] = params.value;
        const label = (x) => `${Number(x.toFixed(digits)) < 0 ? MINUS : ''}${fmtNum(Math.abs(x), digits)}`;
        return escapeHtml(`de ${label(lo)} a ${label(hi)} %: ${fmtCount(count)} barajas`);
      },
    },
    xAxis: {
      type: 'value',
      min,
      max,
      name: narrow ? AXIS_NAME.replace(' y las', '\ny las') : AXIS_NAME,
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: {fontSize: 11, lineHeight: 13, color: THEME.mutedColor},
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisTick: {show: false},
      axisLabel: {...axisLabelStyle(), hideOverlap: true, formatter: (x) => pct(x)},
      splitLine: {show: false},
    },
    yAxis: {
      type: 'value',
      name: 'barajas',
      nameLocation: 'middle',
      nameGap: 44,
      nameTextStyle: {fontSize: 11, color: THEME.mutedColor},
      // at the left edge, not over the zero of the value axis below
      axisLine: {show: false, onZero: false},
      axisLabel: {...axisLabelStyle(), formatter: (x) => fmtCount(x)},
      splitLine: {lineStyle: {color: THEME.gridColor}},
    },
    series: [bars],
  };
}

/**
 * Render the chart (`chanceOption`) in `el`, with the name of the axis in two lines in a narrow container.
 *
 * @param {HTMLElement} el chart container
 * @param {object} chance see `chanceOption`
 * @param {object} opts see `chanceOption`
 * @returns {object} the ECharts instance
 */
export function renderChance(el, chance, opts = {}) {
  return mountChart(el, chanceOption(chance, {narrow: el.clientWidth < NARROW, ...opts}));
}
