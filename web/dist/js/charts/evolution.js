// Evolution of the publications: what the model published each day (the vote or the seats of each party, or of
// each block), one point per day joined by a line, the retrospective runs hollow. A sibling of the poll average
// (`series.js`): same height on the page, a hidden legend driven by the pill legend (`pages/evolution-wire.js`)
// and a vertical axis fitted to the names shown, so marking a single party zooms on its changes.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {axisPctLabel, dayTime, isoDay} from './series.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtDate, fmtDateShort, fmtDateTime, fmtPct} from '../format.js';

const DAY_MS = 24 * 3600 * 1000;

/**
 * Day of a run: `as_of` (the day the model read) or, for headlines without it, the day of `run_at`.
 *
 * @param {object} run a run of the history part
 * @returns {string} ISO date
 */
export function runDay(run) {
  return run.as_of || String(run.run_at).slice(0, 10);
}

/** Whether `run` replaces `kept` on the same day: a live run beats a retrospective one, then the later `run_at`. */
function beats(run, kept) {
  const live = !run.backfill;
  if (live !== !kept.backfill) {
    return live;
  }
  return String(run.run_at) > String(kept.run_at);
}

/**
 * One run per day, ascending by day: the live run over the retrospective one (`backfill`) and, among several of
 * the same kind, the last `run_at`. The original objects, not copies.
 *
 * @param {object[]} runs `runs` list of the history part
 * @returns {object[]} the runs kept
 */
export function dailyRuns(runs) {
  const byDay = new Map();
  for (const run of runs || []) {
    const day = runDay(run);
    const kept = byDay.get(day);
    if (!kept || beats(run, kept)) {
      byDay.set(day, run);
    }
  }
  return [...byDay.keys()].sort().map((day) => byDay.get(day));
}

/**
 * Points `[time, value, index]` of each party in the `nowcast` headline of the runs: `time` the day of the run at
 * local midnight, `index` the position of the run in `runs` (the tooltip reads the run from it). A run without
 * the party, or with a null value, gives no point.
 *
 * @param {object[]} runs daily runs (`dailyRuns`)
 * @param {string[]} parties party or block names, in legend order
 * @param {string} [field='pct'] party field drawn (`pct` or `seats`)
 * @returns {{name: string, points: Array<[number, number, number]>}[]} one entry per name
 */
export function evolutionPoints(runs, parties, field = 'pct') {
  return parties.map((name) => ({
    name,
    points: runs.flatMap((run, index) => {
      const party = ((run.nowcast || {}).parties || []).find((row) => row.name === name);
      const value = party ? party[field] : null;
      return value === null || value === undefined ? [] : [[dayTime(runDay(run)), value, index]];
    }),
  }));
}

/**
 * Bounds of the vertical axis: whole units around the visible data (not multiples of five, as the average: the
 * changes between publications are tiny), at least one unit high.
 *
 * @param {{min: number, max: number}} extent the extent of the visible data
 * @returns {Array<number|null>} `[low, high]`, nulls when the extent is not finite (no data on screen)
 */
export function evolutionBounds({min, max}) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [null, null];
  }
  const low = Math.floor(min);
  return [low, Math.max(Math.ceil(max), low + 1)];
}

/**
 * ECharts option of the evolution: a line per name over the daily runs, the retrospective points hollow, an
 * axis tooltip per day and the slider of the zoom under the chart.
 *
 * @param {object} data `runs` (daily runs, `dailyRuns`), `parties` (names, legend order), `selected` (names shown
 *   at first; all when null), `colors` ({name: colour}), `fullnames` ({name: full name}, for the tooltip),
 *   `field` (party field drawn: `pct` or `seats`), `formatter` (value → text), `legend` (`'hidden'`: no legend
 *   drawn but a selectable one kept, for the HTML pill legend that drives it with `legendToggleSelect`; `true`:
 *   the legend drawn)
 * @returns {object} the ECharts option
 */
export function evolutionOption({
  runs = [], parties = [], selected = null, colors = {}, fullnames = {}, field = 'pct', formatter = fmtPct,
  legend = 'hidden',
} = {}) {
  const colorOf = (name) => colors[name] || OTHERS_COLOR;
  const series = evolutionPoints(runs, parties, field).map((entry) => ({
    type: 'line',
    name: entry.name,
    data: entry.points.map((point) => (runs[point[2]].backfill ? {
      value: point,
      symbolSize: 8,
      itemStyle: {color: '#fff', borderColor: colorOf(entry.name), borderWidth: 2},
    } : point)),
    showSymbol: true,
    symbolSize: 6,
    color: colorOf(entry.name),
    lineStyle: {width: 2},
    emphasis: {focus: 'series'},
  }));
  const hidden = legend === 'hidden';

  /** Box of a day: the run (its time, or its day with the retrospective tag) and every name shown, highest first. */
  function tooltip(params) {
    const items = (Array.isArray(params) ? params : [params]).filter((item) => Array.isArray(item.value));
    if (!items.length) {
      return '';
    }
    const run = runs[items[0].value[2]];
    const head = run.backfill ? `${fmtDate(runDay(run))} · estimación retrospectiva` : fmtDateTime(run.run_at);
    const lines = [`<strong>${escapeHtml(head)}</strong>`];
    for (const item of [...items].sort((a, b) => b.value[1] - a.value[1])) {
      const name = fullnames[item.seriesName] || item.seriesName;
      lines.push(`${item.marker}${escapeHtml(name)}: ${escapeHtml(formatter(item.value[1]))}`);
    }
    return lines.join('<br>');
  }

  return {
    ...baseOption(),
    legend: {
      show: !hidden,
      selectedMode: true,
      type: 'scroll',
      top: 0,
      data: parties,
      selected: Object.fromEntries(parties.map((name) => [name, !selected || selected.includes(name)])),
    },
    grid: {left: 8, right: 24, top: hidden ? 24 : 40, bottom: 56, containLabel: true},
    xAxis: {
      type: 'time',
      minInterval: DAY_MS,
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => fmtDateShort(isoDay(value))},
      splitLine: {show: false},
    },
    // Whole units around the data on screen (`evolutionBounds`); whole ticks for seats.
    yAxis: {
      type: 'value',
      min: (extent) => evolutionBounds(extent)[0],
      max: (extent) => evolutionBounds(extent)[1],
      ...(field === 'seats' ? {minInterval: 1} : {}),
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: field === 'pct' ? axisPctLabel : (value) => formatter(value)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      confine: true,
      formatter: tooltip,
    },
    dataZoom: [{
      type: 'slider',
      bottom: 8,
      filterMode: 'filter',
      labelFormatter: (value) => fmtDate(isoDay(value)),
    }],
    series,
  };
}

/**
 * Option of the image of the evolution (`share.js`): the chart as shown, with a static legend of the names shown
 * only (the others are left out) and, instead of the slider, an invisible zoom at `zoom` (`chartZoom` of the
 * chart on screen), so the image keeps the days and the names of the screen.
 *
 * @param {object} data the arguments of `evolutionOption`
 * @param {object|null} [zoom=null] `{startValue, endValue}` of the chart on screen (null: every day)
 * @returns {object} the ECharts option
 */
export function evolutionImageOption(data, zoom = null) {
  const option = evolutionOption({...data, legend: true});
  const parties = data.parties || [];
  const shown = data.selected ? parties.filter((name) => data.selected.includes(name)) : parties;
  return {
    ...option,
    legend: {show: true, type: 'plain', selectedMode: false, top: 0, data: shown},
    grid: {...option.grid, top: 40, bottom: 8},
    dataZoom: [{type: 'inside', filterMode: 'filter', disabled: true, ...(zoom || {})}],
    series: option.series.filter((entry) => shown.includes(entry.name)),
  };
}

/**
 * Render the evolution in `el` (see `evolutionOption` for the data).
 *
 * @param {HTMLElement} el chart container
 * @param {object} data the arguments of `evolutionOption`
 * @returns {object} the ECharts instance
 */
export function renderEvolution(el, data) {
  return mountChart(el, evolutionOption(data));
}
