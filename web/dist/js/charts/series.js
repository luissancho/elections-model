// Poll average over time: fitted series with its band, published polls and the projection to the event.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtDate, fmtDateShort, fmtNum, fmtPct, fmtRange} from '../format.js';

const DAY_MS = 24 * 3600 * 1000;
const WINDOW_DAYS = 180;
const BAND_OPACITY = 0.15;
const PROJECTION_OPACITY = 0.1;
const MARK_COLOR = '#555';

/**
 * Timestamp of local midnight of an ISO date (`YYYY-MM-DD` parsed as UTC would shift a day west of UTC).
 *
 * @param {string} iso date `YYYY-MM-DD`
 * @returns {number} milliseconds since the epoch
 */
export function dayTime(iso) {
  return new Date(`${iso}T00:00:00`).getTime();
}

/**
 * ISO date (`YYYY-MM-DD`, local calendar) of the day nearest to a timestamp.
 *
 * @param {number} time milliseconds since the epoch
 * @returns {string} the ISO date
 */
export function isoDay(time) {
  // Half a day forward, so a timestamp a few hours off midnight (DST, unsnapped pointer) rounds right.
  const date = new Date(time + DAY_MS / 2);
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/**
 * Initial zoom window: from `days` before `anchor` (clamped to the first date) to the last date.
 *
 * @param {string[]} dates ascending ISO dates of the axis
 * @param {number} days window length before the anchor
 * @param {string|null} anchor ISO date the window counts back from (the last date when null)
 * @returns {{startValue: string, endValue: string}|null} ISO bounds, or null without dates
 */
export function lastWindow(dates, days, anchor = null) {
  if (!dates.length) {
    return null;
  }
  const first = dates[0];
  const last = dates[dates.length - 1];
  const from = new Date(`${anchor || last}T00:00:00Z`);
  from.setUTCDate(from.getUTCDate() - days);
  const start = from.toISOString().slice(0, 10);
  return {startValue: start < first ? first : start, endValue: last};
}

/**
 * Band `lo–hi` of one party as two stacked `line` series: a transparent lower edge at `lo` and an upper
 * series of height `hi - lo` whose area is filled with the party colour. Both carry the party `name`, so
 * the legend toggles them with the party.
 *
 * @param {string} name party name (series and legend name)
 * @param {number[]} times x values (timestamps)
 * @param {Array<number|null>} lo lower bound per x value
 * @param {Array<number|null>} hi upper bound per x value
 * @param {object} opts `color`, `opacity` (area), `stack` (stack id), `z`
 * @returns {object[]} the two series
 */
export function bandSeries(name, times, lo, hi, {color = OTHERS_COLOR, opacity = BAND_OPACITY, stack = `band:${name}`, z = 1} = {}) {
  const valid = (i) => lo[i] !== null && lo[i] !== undefined && hi[i] !== null && hi[i] !== undefined;
  const common = {
    type: 'line',
    name,
    stack,
    color,
    z,
    silent: true,
    symbol: 'none',
    connectNulls: false,
    lineStyle: {opacity: 0},
    emphasis: {disabled: true},
  };
  return [
    {...common, data: times.map((t, i) => [t, valid(i) ? lo[i] : null])},
    {
      ...common,
      areaStyle: {color, opacity},
      data: times.map((t, i) => [t, valid(i) ? hi[i] - lo[i] : null]),
    },
  ];
}

/**
 * Poll points of one party: `[date, value, pollster]` for the polls with a value for it.
 *
 * @param {object[]} polls `polls` list of the polls part (`polls.polls`)
 * @param {string} name party name (column of the poll records)
 * @returns {Array} the points, with ISO dates
 */
export function pollPoints(polls, name) {
  return polls
    .filter((poll) => poll[name] !== null && poll[name] !== undefined)
    .map((poll) => [poll.date, poll[name], poll.pollster]);
}

/**
 * Polls grouped by ISO date, in input order.
 *
 * @param {object[]} polls `polls` list of the polls part (`polls.polls`)
 * @returns {Map<string, object[]>} date → poll records
 */
export function pollsByDate(polls) {
  const byDate = new Map();
  for (const poll of polls) {
    if (!byDate.has(poll.date)) {
      byDate.set(poll.date, []);
    }
    byDate.get(poll.date).push(poll);
  }
  return byDate;
}

/** Vertical mark line at `time` with a label at the top. */
function markAt(time, label) {
  return {xAxis: time, label: {formatter: label, position: 'end', color: MARK_COLOR}};
}

/**
 * Render the poll average of `parties`: per party the `lo–hi` band, the `mean` line, the published polls
 * and, in forecast, the projected band from `asOf` to `when`, with mark lines at both dates.
 *
 * @param {HTMLElement} el chart container
 * @param {object} data `series`, `polls`, `projection` (the `data` members of the envelopes), `parties`
 *   (names, legend order), `selected` (names shown at first; all when null), `colors` ({name: colour}),
 *   `fullnames` ({name: full name}, tooltip), `asOf`, `when` (ISO dates), `anchor` (ISO date the initial
 *   180-day window counts back from: the last poll)
 * @returns {object} the ECharts instance
 */
export function renderSeries(el, {
  series, polls, projection, parties, selected = null, colors = {}, fullnames = {}, asOf, when, anchor = null,
}) {
  const color = (name) => colors[name] || OTHERS_COLOR;
  const seriesTimes = series.dates.map(dayTime);
  const seriesIndex = new Map(series.dates.map((date, i) => [date, i]));
  const groups = (projection && projection.groups && projection.groups.parties) || null;
  // In nowcast the projection is the single day `asOf`: only the mark line is drawn.
  const withProjection = Boolean(groups && projection.dates.length > 1);
  const projectionTimes = withProjection ? projection.dates.map(dayTime) : [];
  const projectionIndex = new Map(withProjection ? projection.dates.map((date, i) => [date, i]) : []);
  const pollRecords = polls.polls || [];
  const byDate = pollsByDate(pollRecords);

  // Mean lines first: the legend takes its icon from the first series of each name.
  const means = parties.map((name) => ({
    type: 'line',
    name,
    color: color(name),
    z: 3,
    showSymbol: false,
    connectNulls: false,
    lineStyle: {width: 2},
    data: seriesTimes.map((t, i) => [t, (series.mean[name] || [])[i] ?? null]),
  }));
  const bands = parties.flatMap((name) => bandSeries(
    name, seriesTimes, series.lo[name] || [], series.hi[name] || [], {color: color(name)},
  ));
  const points = parties.map((name) => ({
    type: 'scatter',
    name,
    color: color(name),
    z: 4,
    symbolSize: 6,
    itemStyle: {opacity: 0.6},
    data: pollPoints(pollRecords, name).map(([date, value, pollster]) => [dayTime(date), value, pollster]),
  }));
  const projected = !withProjection ? [] : parties.flatMap((name) => [
    {
      type: 'line',
      name,
      color: color(name),
      z: 3,
      showSymbol: false,
      connectNulls: false,
      lineStyle: {width: 2, type: 'dashed'},
      data: projectionTimes.map((t, i) => [t, (groups.mean[name] || [])[i] ?? null]),
    },
    ...bandSeries(name, projectionTimes, groups.lo[name] || [], groups.hi[name] || [], {
      color: color(name), opacity: PROJECTION_OPACITY, stack: `projection:${name}`,
    }),
  ]);
  const marks = [markAt(dayTime(asOf), 'Estimación')];
  if (withProjection && when) {
    marks.push(markAt(dayTime(when), 'Elección'));
  }
  // Mark lines on their own empty series, so hiding a party never hides them.
  const markSeries = {
    type: 'line',
    name: 'marks',
    data: [],
    silent: true,
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: {type: 'dashed', color: MARK_COLOR},
      data: marks,
    },
  };

  const axisDates = withProjection ? [...series.dates, ...projection.dates.slice(1)] : series.dates;
  const zoom = lastWindow(axisDates, WINDOW_DAYS, anchor);

  function tooltip(params) {
    if (!params.length) {
      return '';
    }
    const date = isoDay(params[0].axisValue);
    // Parties of the shown series: the legend-hidden ones are not in `params`.
    const shown = new Set(params.map((item) => item.seriesName));
    const visible = parties.filter((name) => shown.has(name));
    const lines = [`<strong>${escapeHtml(fmtDate(date))}</strong>`];
    let source = null;
    let index = null;
    if (seriesIndex.has(date)) {
      source = series;
      index = seriesIndex.get(date);
    } else if (projectionIndex.has(date)) {
      source = groups;
      index = projectionIndex.get(date);
      lines.push('Proyección');
    }
    if (source) {
      const rows = visible
        .map((name) => ({
          name,
          mean: (source.mean[name] || [])[index] ?? null,
          lo: (source.lo[name] || [])[index] ?? null,
          hi: (source.hi[name] || [])[index] ?? null,
        }))
        .filter((row) => row.mean !== null)
        .sort((a, b) => b.mean - a.mean);
      for (const row of rows) {
        const marker = window.echarts.format.getTooltipMarker(escapeHtml(color(row.name)));
        const label = fullnames[row.name] || row.name;
        lines.push(`${marker}${escapeHtml(`${label}: ${fmtPct(row.mean)} (${fmtRange(row.lo, row.hi)})`)}`);
      }
    }
    for (const poll of byDate.get(date) || []) {
      const values = visible
        .filter((name) => poll[name] !== null && poll[name] !== undefined)
        .map((name) => `${name} ${fmtNum(poll[name], 1)}`);
      if (values.length) {
        lines.push(escapeHtml(`${poll.pollster} · ${fmtDateShort(poll.date)} · ${values.join(' · ')}`));
      }
    }
    return lines.join('<br>');
  }

  const option = {
    ...baseOption(),
    legend: {
      type: 'scroll',
      top: 0,
      data: parties,
      selected: Object.fromEntries(parties.map((name) => [name, !selected || selected.includes(name)])),
    },
    // Right margin wide enough for the centred mark-line labels at the end of the axis.
    grid: {left: 8, right: 48, top: 40, bottom: 56, containLabel: true},
    xAxis: {
      type: 'time',
      minInterval: DAY_MS,
      axisLabel: {formatter: (value) => fmtDateShort(isoDay(value))},
      splitLine: {show: false},
    },
    yAxis: {
      type: 'value',
      min: 0,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (value) => fmtPct(value, 0)},
    },
    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: {type: 'line'},
      formatter: tooltip,
    },
    dataZoom: zoom ? [{
      type: 'slider',
      bottom: 8,
      filterMode: 'none',
      startValue: dayTime(zoom.startValue),
      endValue: dayTime(zoom.endValue),
      labelFormatter: (value) => fmtDate(isoDay(value)),
    }] : [],
    series: [...means, ...bands, ...points, ...projected, markSeries],
  };
  return mountChart(el, option);
}
