// Dispersion of the polls by weeks before the election (the herd study): one thin grey line per past cycle, their
// dashed black mean over the line of chance (1) and its shaded band below, and the cycle in progress thick in blue.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

// The greys of the past cycles (the approved mockup), used in turn.
const GREYS = ['#9aa3b2', '#8d96a6', '#7f8898', '#aab2bf', '#737c8c', '#b5bcc8', '#67707f'];
const REFERENCE_COLOR = '#ce3b3b';
const BAND_COLOR = 'rgba(206,59,59,.06)';
const REFERENCE_LABEL = 'lo que permite el azar de las muestras';
// Room on the right for the end label of the cycle in progress when it reaches the last bucket.
const END_LABEL_ROOM = 80;

/**
 * The values of some rows in the order of the buckets (`null` where a bucket has no row or no ratio).
 *
 * @param {object[]} buckets `{lo, hi}` per bucket
 * @param {object[]} rows `{lo, hi, ratio}` rows
 * @returns {Array<number|null>} one value per bucket
 */
function byBucket(buckets, rows) {
  const values = new Map(rows.map((row) => [`${row.lo}-${row.hi}`, row.ratio]));
  return buckets.map((bucket) => {
    const value = values.get(`${bucket.lo}-${bucket.hi}`);
    return value === undefined ? null : value;
  });
}

/**
 * The rows of each cycle, in the order the cycles first appear.
 *
 * @param {object[]} rows the rows of the part, `{event_date, label, current, ...}`
 * @returns {{label: string, current: boolean, rows: object[]}[]} the cycles
 */
function cyclesOf(rows) {
  const cycles = new Map();
  for (const row of rows) {
    if (!cycles.has(row.event_date)) {
      cycles.set(row.event_date, {label: row.label, current: Boolean(row.current), rows: []});
    }
    cycles.get(row.event_date).rows.push(row);
  }
  return [...cycles.values()];
}

/**
 * ECharts option of the chart: a category axis of the buckets of weeks, a value axis from 0 in steps of 0.5 and the
 * series of the past cycles, of their mean (carrying the line of chance and its band) and of the cycle in progress.
 *
 * @param {{buckets: object[], rows: object[], mean: object[]}} data the `weeks` part: `buckets` as `{lo, hi, label}`,
 *   `rows` as `{event_date, label, current, lo, hi, k, ratio}`, `mean` as `{lo, hi, k, ratio}`
 * @param {object} opts `current` (colour of the cycle in progress), `mean` (colour of the mean), `reference` (the
 *   ratio of chance) and `yMax` (the least top of the value axis)
 * @returns {object} the option
 */
export function weeksOption({buckets = [], rows = [], mean = []} = {},
                            {current = '#1d4f86', mean: meanColor = THEME.textColor, reference = 1, yMax = 3} = {}) {
  const cycles = cyclesOf(rows);
  const past = cycles.filter((cycle) => !cycle.current);
  const live = cycles.find((cycle) => cycle.current);
  const ratios = [...rows, ...mean].map((row) => row.ratio).filter((x) => typeof x === 'number');
  const top = Math.max(yMax, Math.ceil(Math.max(0, ...ratios) * 2) / 2);
  const series = past.map((cycle, i) => ({
    name: cycle.label,
    type: 'line',
    data: byBucket(buckets, cycle.rows),
    connectNulls: true,
    symbol: 'circle',
    symbolSize: 4,
    lineStyle: {width: 1.4, color: GREYS[i % GREYS.length]},
    itemStyle: {color: GREYS[i % GREYS.length]},
    emphasis: {focus: 'series'},
  }));
  series.push({
    name: `Media de los ${past.length}`,
    type: 'line',
    data: byBucket(buckets, mean),
    symbol: 'none',
    lineStyle: {width: 2.5, color: meanColor, type: 'dashed'},
    itemStyle: {color: meanColor},
    markLine: {
      silent: true,
      symbol: 'none',
      data: [{yAxis: reference}],
      lineStyle: {color: REFERENCE_COLOR, width: 1.2, type: 'solid'},
      // at the start, inside the band: the end is where the cycle in progress writes its label
      label: {formatter: REFERENCE_LABEL, position: 'insideStartBottom', fontSize: 10.5, color: REFERENCE_COLOR},
    },
    markArea: {silent: true, itemStyle: {color: BAND_COLOR}, data: [[{yAxis: 0}, {yAxis: reference}]]},
  });
  let right = 16;
  if (live) {
    const data = byBucket(buckets, live.rows);
    if (data.length && data[data.length - 1] !== null) {
      right = END_LABEL_ROOM;
    }
    series.push({
      name: live.label,
      type: 'line',
      data,
      symbol: 'circle',
      symbolSize: 7,
      lineStyle: {width: 3.5, color: current},
      itemStyle: {color: current, borderColor: '#fff', borderWidth: 1.5},
      z: 5,
      endLabel: {
        show: true,
        fontSize: 11,
        fontWeight: 700,
        color: current,
        formatter: (params) => `${live.label} · ${fmtNum(params.value, 2)}`,
      },
    });
  }
  return {
    ...baseOption(),
    grid: {left: 44, right, top: 36, bottom: 44},
    legend: {
      type: 'scroll',
      top: 0,
      left: 0,
      itemWidth: 16,
      itemHeight: 3,
      textStyle: {fontSize: 11, color: THEME.mutedColor},
      // the cycle in progress and the mean first: in a narrow column the legend pages, and they must stay in sight
      data: [...series.slice(past.length).reverse(), ...series.slice(0, past.length)].map((item) => item.name),
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      formatter(params) {
        const lines = params
          .filter((item) => item.value !== null && item.value !== undefined)
          .map((item) => `${item.marker}${escapeHtml(item.seriesName)}: ${escapeHtml(fmtNum(item.value, 2))}`);
        return [escapeHtml(params[0].axisValue), ...lines].join('<br>');
      },
    },
    xAxis: {
      type: 'category',
      data: buckets.map((bucket) => bucket.label),
      name: 'semanas antes de la elección',
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: {fontSize: 11, color: THEME.mutedColor},
      axisTick: {show: false},
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: {...axisLabelStyle(), fontSize: 11},
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: top,
      interval: 0.5,
      axisLabel: {...axisLabelStyle(), fontSize: 11, formatter: (x) => fmtNum(x, 2)},
      splitLine: {lineStyle: {color: THEME.gridColor}},
    },
    series,
  };
}

/**
 * Render the chart (`weeksOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {object} data see `weeksOption`
 * @param {object} opts see `weeksOption`
 * @returns {object} the ECharts instance
 */
export function renderWeeks(el, data, opts) {
  return mountChart(el, weeksOption(data, opts));
}
