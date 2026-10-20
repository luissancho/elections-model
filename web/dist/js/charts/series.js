// Poll average over time: fitted series with its band, published polls, the projection to the event and, in the
// detail of a past election, the official result marked at the election date. The vertical axis follows what is
// on screen (the window and the parties the legend shows) instead of starting at zero, so the curves keep their
// shape (Luis, 2026-10-19).
import {axisLabelStyle, baseOption, mountChart, THEME, tooltipStyle} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtDate, fmtDateShort, fmtPct} from '../format.js';
import {meanTooltipLine, pollTooltipLines, resultTooltipLine} from '../polls-tooltip.js';
import {DEFAULT_WINDOW, seriesWindow} from '../range.js';

export {meanTooltipLine, pollTooltipLines, resultTooltipLine};

const DAY_MS = 24 * 3600 * 1000;
const BAND_OPACITY = 0.15;
const PROJECTION_OPACITY = 0.1;
// Symbols of the average lines: invisible until hovered, so that each day of a line has its own tooltip.
const HOVER_SYMBOLS = {
  showSymbol: true,
  symbol: 'circle',
  symbolSize: 8,
  itemStyle: {opacity: 0},
  emphasis: {itemStyle: {opacity: 1}},
};

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
 * Poll points of one party: `[date, value, index]` for the polls with a value for it, `index` being the position
 * of the poll in `polls` (the tooltip reads the whole poll from it).
 *
 * @param {object[]} polls `polls` list of the polls part (`polls.polls`)
 * @param {string} name party name (column of the poll records)
 * @returns {Array} the points, with ISO dates
 */
export function pollPoints(polls, name) {
  return polls
    .map((poll, index) => [poll.date, poll[name], index])
    .filter(([, value]) => value !== null && value !== undefined);
}

/**
 * Label of a tick of the vertical axis: whole percentages without decimals and, when the axis is so tight that
 * the ticks fall between them, one decimal (`32%`, `32,2%`).
 *
 * @param {number} value the tick value
 * @returns {string} the label
 */
export function axisPctLabel(value) {
  return fmtPct(value, Math.abs(value - Math.round(value)) < 1e-6 ? 0 : 1);
}

/** Vertical mark line at `time` with a label at the top. */
function markAt(time, label) {
  return {xAxis: time, label: {formatter: label, position: 'end', color: THEME.markColor}};
}

/**
 * ISO dates of the time axis: the fitted series plus, in forecast, the projected days after it and, with an
 * official result later than the last of them, its date (the detail of a past election: the series stops where
 * the model was frozen and the result sits at the election).
 *
 * @param {object} series `data` of the series part
 * @param {object|null} projection `data` of the projection part
 * @param {{date: string}|null} [result=null] the official result marked (`seriesOption`)
 * @returns {string[]} ascending ISO dates
 */
export function axisDatesOf(series, projection, result = null) {
  const groups = (projection && projection.groups && projection.groups.parties) || null;
  const withProjection = Boolean(groups && projection.dates.length > 1);
  const dates = withProjection ? [...series.dates, ...projection.dates.slice(1)] : series.dates;
  const last = dates[dates.length - 1];
  return result && result.date && (!last || result.date > last) ? [...dates, result.date] : dates;
}

/**
 * Zoom a rendered series chart to a choice of the window control (`3m`, `6m`, `1a`, `all`).
 *
 * @param {object} chart the ECharts instance returned by `renderSeries`
 * @param {string} choice value of the control
 * @param {object} data `series`, `projection` (parts), `anchor` (ISO date the window counts back from) and
 *   `result` (the official result marked, whose date closes the axis)
 */
export function setWindow(chart, choice, {series, projection, anchor = null, result = null}) {
  const zoom = seriesWindow(choice, axisDatesOf(series, projection, result), anchor);
  if (zoom) {
    chart.dispatchAction({
      type: 'dataZoom', dataZoomIndex: 0, startValue: dayTime(zoom.startValue), endValue: dayTime(zoom.endValue),
    });
  }
}

/**
 * Render the poll average of `parties`: per party the `lo–hi` band, the `mean` line, the published polls
 * and, in forecast, the projected band from `asOf` to `when`, with mark lines at both dates. The initial
 * zoom is the `window` choice (a key of `WINDOW_DAYS`: the last days before `anchor`; `all`: the whole axis).
 * The window filters the data (`filterMode: 'filter'`) and the vertical axis is scaled to the visible data of
 * the series the legend shows, without a fixed zero, so a tight race fills the height.
 * The Promedios page also draws the blocks with it, passing the block series, polls and projection as parties.
 * The tooltip follows the item under the pointer: a poll shows its box (`pollTooltipLines`), a day of the
 * average or of the projection its value and band (`meanTooltipLine`), the official result its share
 * (`resultTooltipLine`); the bands have none.
 *
 * @param {object} data `series`, `polls`, `projection` (the `data` members of the envelopes), `parties`
 *   (names, legend order), `selected` (names shown at first; all when null), `colors` ({name: colour}),
 *   `asOf`, `when` (ISO dates), `window` (choice of the window control), `anchor` (ISO date the window counts
 *   back from: the last poll), `legend` (`true`: selectable legend; `'readonly'`: a legend that only shows the
 *   key and follows `selected`; `'hidden'`: no legend drawn but a selectable one kept, for the HTML pill legend
 *   that drives it with `legendToggleSelect`; `false`: hidden, but its `selected` still decides the names shown,
 *   for pages with their own selector), `result` (`{date, values: {name: share}}`, the official result of a
 *   past election: a diamond per party with a value, at `date`, which also gets the «Elección» mark line and
 *   closes the axis; null for none) and `asOfLabel` (label of the mark line at `asOf`, «Estimación» by default)
 * @returns {object} the ECharts option
 */
export function seriesOption({
  series, polls, projection, parties, selected = null, colors = {}, asOf, when,
  window: windowChoice = DEFAULT_WINDOW, anchor = null, legend = true, result = null, asOfLabel = 'Estimación',
}) {
  const color = (name) => colors[name] || OTHERS_COLOR;
  const seriesTimes = series.dates.map(dayTime);
  const groups = (projection && projection.groups && projection.groups.parties) || null;
  // In nowcast the projection is the single day `asOf`: only the mark line is drawn.
  const withProjection = Boolean(groups && projection.dates.length > 1);
  const projectionTimes = withProjection ? projection.dates.map(dayTime) : [];
  const pollRecords = polls.polls || [];
  const colorOf = Object.fromEntries(parties.map((name) => [name, color(name)]));
  // Role of each series for the tooltip: `mean`, `projection` or `poll` (bands and marks have none).
  const roles = new Map();
  const withRole = (role, entry) => {
    roles.set(entry, role);
    return entry;
  };

  // Mean lines first: the legend takes its icon from the first series of each name.
  const means = parties.map((name) => withRole('mean', {
    type: 'line',
    name,
    color: color(name),
    z: 3,
    ...HOVER_SYMBOLS,
    connectNulls: false,
    lineStyle: {width: 2},
    data: seriesTimes.map((t, i) => [t, (series.mean[name] || [])[i] ?? null]),
  }));
  const bands = parties.flatMap((name) => bandSeries(
    name, seriesTimes, series.lo[name] || [], series.hi[name] || [], {color: color(name)},
  ));
  const points = parties.map((name) => withRole('poll', {
    type: 'scatter',
    name,
    color: color(name),
    z: 4,
    symbolSize: 6,
    itemStyle: {opacity: 0.6},
    data: pollPoints(pollRecords, name).map(([date, value, index]) => [dayTime(date), value, index]),
  }));
  const projected = !withProjection ? [] : parties.flatMap((name) => [
    withRole('projection', {
      type: 'line',
      name,
      color: color(name),
      z: 3,
      ...HOVER_SYMBOLS,
      connectNulls: false,
      lineStyle: {width: 2, type: 'dashed'},
      data: projectionTimes.map((t, i) => [t, (groups.mean[name] || [])[i] ?? null]),
    }),
    ...bandSeries(name, projectionTimes, groups.lo[name] || [], groups.hi[name] || [], {
      color: color(name), opacity: PROJECTION_OPACITY, stack: `projection:${name}`,
    }),
  ]);
  // The official result of a past election: a diamond per party with a value, drawn over everything else.
  const results = !result ? [] : parties
    .filter((name) => Number.isFinite(result.values[name]))
    .map((name) => withRole('result', {
      type: 'scatter',
      name,
      color: color(name),
      z: 5,
      symbol: 'diamond',
      symbolSize: 12,
      itemStyle: {borderColor: '#fff', borderWidth: 1.5},
      data: [[dayTime(result.date), result.values[name]]],
    }));
  const marks = [markAt(dayTime(asOf), asOfLabel)];
  if (result) {
    marks.push(markAt(dayTime(result.date), 'Elección'));
  } else if (withProjection && when) {
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
      lineStyle: {type: 'dashed', color: THEME.markColor},
      data: marks,
    },
  };

  const axisDates = axisDatesOf(series, projection, result);
  const zoom = seriesWindow(windowChoice, axisDates, anchor);

  const allSeries = [...means, ...bands, ...points, ...projected, ...results, markSeries];

  /** Box of the item under the pointer: a poll, a day of the average or of the projection, or the result. */
  function tooltip(params) {
    const role = roles.get(allSeries[params.seriesIndex]);
    const name = params.seriesName;
    const i = params.dataIndex;
    if (role === 'poll') {
      const poll = pollRecords[params.data[2]];
      return poll ? pollTooltipLines(poll, parties, colorOf).join('<br>') : '';
    }
    if (role === 'result') {
      return resultTooltipLine(name, result.values[name], result.date, color(name));
    }
    if (role === 'mean' || role === 'projection') {
      const source = role === 'mean' ? series : groups;
      const dates = role === 'mean' ? series.dates : projection.dates;
      const at = (values) => (values || [])[i] ?? null;
      return meanTooltipLine(name, at(source.mean[name]), at(source.lo[name]), at(source.hi[name]), dates[i],
        color(name), role === 'projection');
    }
    return '';
  }

  const drawn = Boolean(legend) && legend !== 'hidden';
  return {
    ...baseOption(),
    legend: {
      show: drawn,
      ...(legend === 'readonly' ? {selectedMode: false} : {}),
      ...(legend === 'hidden' ? {selectedMode: true} : {}),
      type: 'scroll',
      top: 0,
      data: parties,
      selected: Object.fromEntries(parties.map((name) => [name, !selected || selected.includes(name)])),
    },
    // Right margin wide enough for the centred mark-line labels at the end of the axis.
    // Without a drawn legend the top margin only keeps room for the mark-line labels.
    grid: {left: 8, right: 48, top: drawn ? 40 : 24, bottom: 56, containLabel: true},
    xAxis: {
      type: 'time',
      minInterval: DAY_MS,
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: {...axisLabelStyle(), formatter: (value) => fmtDateShort(isoDay(value))},
      splitLine: {show: false},
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: axisPctLabel},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      confine: true,
      formatter: tooltip,
    },
    dataZoom: zoom ? [{
      type: 'slider',
      bottom: 8,
      filterMode: 'filter',
      startValue: dayTime(zoom.startValue),
      endValue: dayTime(zoom.endValue),
      labelFormatter: (value) => fmtDate(isoDay(value)),
    }] : [],
    series: allSeries,
  };
}

/**
 * Render the poll average in `el` (see `seriesOption` for the data).
 *
 * @param {HTMLElement} el chart container
 * @param {object} data the arguments of `seriesOption`
 * @returns {object} the ECharts instance
 */
export function renderSeries(el, data) {
  return mountChart(el, seriesOption(data));
}
