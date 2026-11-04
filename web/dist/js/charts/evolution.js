// Evolution of the publications: what the model published each day (the vote or the seats of each party, or of
// each block), one point per day joined by a line, hollow where the figure shown changed from the day before (Luis,
// 2026-11-04: live and retrospective runs alike). A sibling of the poll average
// (`series.js`): same height on the page, a hidden legend driven by the pill legend (`pages/evolution-wire.js`)
// and a vertical axis fitted to the names shown with the rule of the average (ticks every `AXIS_STEP` points on
// multiples of it; every `SEATS_STEP` seats, Luis 2026-10-30), so marking a single party zooms on its changes.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {AXIS_STEP, axisBounds, axisPctLabel, dayTime, isoDay} from './series.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtBadge, fmtDate, fmtDateShort, fmtPct} from '../format.js';

const DAY_MS = 24 * 3600 * 1000;
/** Tick step of the seats axis: the whole chamber keeps about as many lines as the vote axis (Luis, 2026-10-30). */
export const SEATS_STEP = 20;
/** Decimals of the figure shown of each field: the vote to one decimal, whole seats. */
export const FIELD_DIGITS = {pct: 1, seats: 0};

/** Formats that round as the site's figures do (Intl rounds the shortest decimal form of the number). */
const roundFormats = new Map();

/**
 * The figure shown of a value: rounded to the decimals of its field (`FIELD_DIGITS`) the way `fmtPct`/`fmtInt` write
 * it (Intl, not `Math.round`: block sums carry float noise such as 7.1499999999999995, written 7,1). The donuts, the
 * arrows and the table of changes compare these, so a drift below the rounding is no change.
 *
 * @param {(number|null|undefined)} value the value
 * @param {string} field party field (`pct` or `seats`)
 * @returns {(number|null)} the rounded value, null when missing
 */
export function shownValue(value, field) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  const digits = FIELD_DIGITS[field] ?? 0;
  if (!roundFormats.has(digits)) {
    roundFormats.set(digits, new Intl.NumberFormat('en-US', {
      minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: false,
    }));
  }
  return Number(roundFormats.get(digits).format(value));
}

/**
 * Day of a run: `as_of` (the day the model read) or, for headlines without it, the day of `run_at`.
 *
 * @param {object} run a run of the history part
 * @returns {string} ISO date
 */
export function runDay(run) {
  return run.as_of || String(run.run_at).slice(0, 10);
}

/** Whether `run` replaces `kept` on the same day: the later `run_at`, live or retrospective (Luis, 2026-11-04). */
function beats(run, kept) {
  return String(run.run_at) > String(kept.run_at);
}

/**
 * One run per day, ascending by day: the last `run_at` of the day, whether live or retrospective (`backfill`).
 * The original objects, not copies.
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
 * ECharts option of the evolution: a line per name over the daily runs, hollow where the figure shown
 * (`shownValue`) changed from the previous point of the name, an axis tooltip per day with the arrow of each change
 * and the slider of the zoom under the chart.
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
  const digits = FIELD_DIGITS[field] ?? 0;
  // change of the figure shown from the previous point of the name, by `name|run index` (the tooltip reads it)
  const changes = new Map();
  const series = evolutionPoints(runs, parties, field).map((entry) => ({
    type: 'line',
    name: entry.name,
    data: entry.points.map((point, i) => {
      if (i === 0) {
        return point;
      }
      const delta = shownValue(point[1], field) - shownValue(entry.points[i - 1][1], field);
      if (Math.abs(delta) < 0.5 * 10 ** -digits) {
        return point;
      }
      changes.set(`${entry.name}|${point[2]}`, delta);
      return {value: point, symbolSize: 8, itemStyle: {color: '#fff', borderColor: colorOf(entry.name), borderWidth: 2}};
    }),
    // a filled circle: the default `emptyCircle` strokes with `color`, so a white hollow point would vanish
    symbol: 'circle',
    showSymbol: true,
    symbolSize: 6,
    color: colorOf(entry.name),
    lineStyle: {width: 2},
    emphasis: {focus: 'series'},
  }));
  const hidden = legend === 'hidden';
  const step = field === 'seats' ? SEATS_STEP : AXIS_STEP;

  /** Box of a day: its date and every name shown, highest first, with the arrow of its change when it changed. */
  function tooltip(params) {
    const items = (Array.isArray(params) ? params : [params]).filter((item) => Array.isArray(item.value));
    if (!items.length) {
      return '';
    }
    const run = runs[items[0].value[2]];
    const lines = [`<strong>${escapeHtml(fmtDate(runDay(run)))}</strong>`];
    for (const item of [...items].sort((a, b) => b.value[1] - a.value[1])) {
      const name = fullnames[item.seriesName] || item.seriesName;
      const delta = changes.get(`${item.seriesName}|${item.value[2]}`);
      const badge = delta === undefined ? null : fmtBadge(delta, digits);
      const arrow = badge ? ` <span class="dlt ${badge.dir}">${escapeHtml(badge.text)}</span>` : '';
      lines.push(`${item.marker}${escapeHtml(name)}: ${escapeHtml(formatter(item.value[1]))}${arrow}`);
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
    // Ticks every `step` on multiples of it, around the data on screen (`axisBounds`, as the average).
    yAxis: {
      type: 'value',
      min: (extent) => axisBounds(extent, step)[0],
      max: (extent) => axisBounds(extent, step)[1],
      interval: step,
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
