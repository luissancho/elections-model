// Separation between houses by weeks before the election (the herd study): one coloured line per past general
// election, their dashed black mean over the thin line of chance (1), the election in progress thick in blue, and two
// light zones over the stretches the counts below compare (3 to 8 weeks before and the last three).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtNum} from '../format.js';

// The colours of the past elections, used in turn in the order of the rows.
export const PALETTE = ['#e07b39', '#2a9d8f', '#8e44ad', '#c0392b', '#7f8c3a', '#d4a017', '#5d6d7e'];
const REFERENCE_LABEL = 'lo que cabría esperar sólo por el azar';
// The two stretches of the counts below the chart: `[from, to, label, colour]` by bucket label. A `markArea` on a
// category axis runs from the centre of one bucket to the centre of another (and has no width over a single bucket),
// so the zones are shaded whole by the `splitArea` of the axis, one colour per bucket, and the `markArea`, clear,
// only carries their labels over them.
const STRETCHES = [
  ['5-7', '3-4', '3-8 semanas', 'rgba(90,95,114,.08)'],
  ['0-2', '0-2', 'tres últimas', 'rgba(29,79,134,.08)'],
];
const LABEL_BACKGROUND = 'rgba(255,255,255,.85)';

/**
 * Some field of some rows in the order of the buckets (`null` where a bucket has no row or no value).
 *
 * @param {object[]} buckets `{lo, hi}` per bucket
 * @param {object[]} rows `{lo, hi, ...}` rows
 * @param {string} field the field to read (`ratio`, `pairs`, `k`)
 * @returns {Array<number|null>} one value per bucket
 */
function byBucket(buckets, rows, field = 'ratio') {
  const values = new Map(rows.map((row) => [`${row.lo}-${row.hi}`, row[field]]));
  return buckets.map((bucket) => {
    const value = values.get(`${bucket.lo}-${bucket.hi}`);
    return value === undefined ? null : value;
  });
}

/**
 * The rows of each election, in the order the elections first appear.
 *
 * @param {object[]} rows the rows of the part, `{event_date, label, current, ...}`
 * @returns {{label: string, current: boolean, rows: object[]}[]} the elections
 */
function electionsOf(rows) {
  const elections = new Map();
  for (const row of rows) {
    if (!elections.has(row.event_date)) {
      elections.set(row.event_date, {label: row.label, current: Boolean(row.current), rows: []});
    }
    elections.get(row.event_date).rows.push(row);
  }
  return [...elections.values()];
}

/**
 * What the tooltip adds to the value of an election in each bucket: « (12 parejas)».
 *
 * @param {object[]} buckets `{lo, hi}` per bucket
 * @param {object[]} rows the rows of the election
 * @returns {string[]} one note per bucket (`''` without pairs)
 */
function pairsNotes(buckets, rows) {
  return byBucket(buckets, rows, 'pairs').map((pairs) => (pairs === null ? '' : ` (${pairs} parejas)`));
}

/**
 * The labels of the stretches whose buckets the chart has (the `markArea` data).
 *
 * @param {string[]} labels the labels of the buckets
 * @returns {Array<object[]>} the `markArea` data
 */
function stretchAreas(labels) {
  return STRETCHES
    .filter(([from, to]) => labels.includes(from) && labels.includes(to))
    .map(([from, to, name]) => [{name, xAxis: from}, {xAxis: to}]);
}

/**
 * The colour of each bucket in the `splitArea` of the axis: the one of its stretch, transparent outside them.
 *
 * @param {string[]} labels the labels of the buckets
 * @returns {string[]} one colour per bucket
 */
function stretchColors(labels) {
  const colors = labels.map(() => 'rgba(0,0,0,0)');
  for (const [from, to, , color] of STRETCHES) {
    const start = labels.indexOf(from);
    const end = labels.indexOf(to);
    if (start >= 0 && end >= start) {
      colors.fill(color, start, end + 1);
    }
  }
  return colors;
}

/**
 * ECharts option of the chart: a category axis of the buckets of weeks, a value axis from 0 in steps of 0.5 and the
 * series of the past elections, of their mean (carrying the line of chance and the zones of the stretches) and of
 * the election in progress.
 *
 * @param {{buckets: object[], rows: object[], mean: object[]}} data the `weeks` part: `buckets` as `{lo, hi, label}`
 *   (farthest first), `rows` as `{event_date, label, current, lo, hi, pairs, comparisons, ratio}`, `mean` as
 *   `{lo, hi, k, ratio}`
 * @param {object} opts `current` (colour of the election in progress), `mean` (colour of the mean), `reference`
 *   (the ratio of chance), `yMax` (the least top of the value axis) and `palette` (the colours of the past
 *   elections, cyclic)
 * @returns {object} the option
 */
export function weeksOption({buckets = [], rows = [], mean = []} = {},
                            {current = THEME.accentColor, mean: meanColor = THEME.textColor, reference = 1, yMax = 3,
                             palette = PALETTE} = {}) {
  const elections = electionsOf(rows);
  const past = elections.filter((election) => !election.current);
  const live = elections.find((election) => election.current);
  const labels = buckets.map((bucket) => bucket.label);
  const ratios = [...rows, ...mean].map((row) => row.ratio).filter((x) => typeof x === 'number');
  const top = Math.max(yMax, Math.ceil(Math.max(0, ...ratios) * 2) / 2);
  // what the tooltip adds to each value, by series and bucket: the pairs of an election, the elections of the mean
  const notes = [];
  const series = past.map((election, i) => {
    const color = palette[i % palette.length];
    notes.push(pairsNotes(buckets, election.rows));
    return {
      name: election.label,
      type: 'line',
      data: byBucket(buckets, election.rows),
      connectNulls: true,
      symbol: 'circle',
      symbolSize: 5,
      lineStyle: {width: 1.6, color},
      itemStyle: {color},
      emphasis: {focus: 'series'},
    };
  });
  notes.push(byBucket(buckets, mean, 'k').map((k) => {
    if (k === null || k >= past.length) {
      return '';
    }
    return ` (${k} ${k === 1 ? 'general' : 'generales'})`;
  }));
  series.push({
    name: `Media de las ${past.length}`,
    type: 'line',
    data: byBucket(buckets, mean),
    symbol: 'none',
    lineStyle: {width: 2.5, color: meanColor, type: 'dashed'},
    itemStyle: {color: meanColor},
    z: 4,
    markLine: {
      silent: true,
      symbol: 'none',
      data: [{yAxis: reference}],
      lineStyle: {color: THEME.mutedColor, width: 1, type: 'solid'},
      // at the start: the end is where the election in progress writes its label
      label: {
        formatter: REFERENCE_LABEL,
        position: 'insideStartBottom',
        fontSize: 10.5,
        color: THEME.mutedColor,
        backgroundColor: LABEL_BACKGROUND,
        padding: [1, 3],
      },
    },
    markArea: {
      silent: true,
      itemStyle: {color: 'rgba(0,0,0,0)'},
      // in two lines: over a phone the buckets are narrow
      label: {position: 'insideTop', fontSize: 10.5, lineHeight: 13, color: THEME.mutedColor,
              formatter: (params) => params.name.replace(' ', '\n')},
      data: stretchAreas(labels),
    },
  });
  if (live) {
    const data = byBucket(buckets, live.rows);
    notes.push(pairsNotes(buckets, live.rows));
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
        // above its last point and ending there: the buckets on its left have room, the right edge may not
        align: 'right',
        verticalAlign: 'bottom',
        distance: 0,
        offset: [0, -6],
        backgroundColor: LABEL_BACKGROUND,
        padding: [1, 3],
        borderRadius: 3,
        formatter: (params) => `${live.label} · ${fmtNum(params.value, 2)}`,
      },
    });
  }
  return {
    ...baseOption(),
    grid: {left: 44, right: 16, top: 44, bottom: 44},
    legend: {
      type: 'scroll',
      top: 0,
      left: 0,
      itemWidth: 16,
      itemHeight: 3,
      textStyle: {fontSize: 11, color: THEME.mutedColor},
      // the election in progress and the mean first: in a narrow column the legend pages, and they must stay in sight;
      // the one in progress as a plain bar (its points, ringed in white, would cut its icon like the dashed mean's)
      data: [...series.slice(past.length).reverse(), ...series.slice(0, past.length)]
        .map((item) => (live && item.name === live.label ? {name: item.name, icon: 'roundRect'} : item.name)),
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      formatter(params) {
        const lines = params
          .filter((item) => item.value !== null && item.value !== undefined)
          .map((item) => {
            const note = (notes[item.seriesIndex] || [])[item.dataIndex] || '';
            return `${item.marker}${escapeHtml(item.seriesName)}: ${escapeHtml(fmtNum(item.value, 2))}${note}`;
          });
        return [`${escapeHtml(params[0].axisValue)} semanas`, ...lines].join('<br>');
      },
    },
    xAxis: {
      type: 'category',
      data: labels,
      name: 'semanas antes de la elección',
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: {fontSize: 11, color: THEME.mutedColor},
      axisTick: {show: false},
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: {...axisLabelStyle(), fontSize: 11},
      splitArea: {show: true, interval: 0, areaStyle: {color: stretchColors(labels)}},
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
