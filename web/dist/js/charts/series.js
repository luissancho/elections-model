// Poll average over time: fitted series, published polls, the dashed projection to the event with the 95 % range of
// each party at the election drawn as a bar (the band of the average and the fan of the projection are no longer
// drawn: seen together they read as one band with a step, Luis 2026-10-26) and, in the detail of a past election,
// the official result marked at the election date. The vertical axis follows what is on screen (the window and the
// parties the legend shows) instead of starting at zero, so the curves keep their shape (Luis, 2026-10-19), with a
// tick every 5 points and the same span whichever layers are drawn (Luis, 2026-10-28).
import {axisLabelStyle, baseOption, mountChart, THEME, timeAxisLabel, tooltipStyle} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtDate, fmtPct} from '../format.js';
import {meanTooltipLine, pollTooltipLines, resultTooltipLine} from '../polls-tooltip.js';
import {DEFAULT_WINDOW, seriesWindow} from '../range.js';

export {meanTooltipLine, pollTooltipLines, resultTooltipLine};

const DAY_MS = 24 * 3600 * 1000;
const AXIS_STEP = 5;  // the ticks of the vertical axis, in points of vote share (Luis, 2026-10-28)
const BAND_OPACITY = 0.15;  // `bandSeries`, still drawn by the live example of the method (`weights.js`)
const RANGE_OPACITY = 0.45;  // the range bar at the election; full opacity when hovered
const RANGE_BLUR_OPACITY = 0.15;  // the other bars while one is hovered
// Blur states of the lines and points: the hovered bar blurs every other series of the chart, and these keep them as
// they are, so only the other bars fade.
const KEEP_LINE_ON_BLUR = {blur: {lineStyle: {opacity: 1}}};
const KEEP_POINTS_ON_BLUR = (opacity) => ({blur: {itemStyle: {opacity}}});
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
 * Bounds of the vertical axis for the extent `{min, max}` of the data on screen: the multiples of `AXIS_STEP`
 * just outside it (a value on a multiple stays on it), at least one step apart, so the ticks fall on 20, 25,
 * 30… whatever the window, the parties shown or the layers drawn. ECharts calls `min` and `max` of the axis
 * separately, each with the whole extent, and both read their end here. Without data (every party hidden) the
 * extent is infinite and the bounds are null, which leaves the axis to ECharts.
 *
 * @param {{min: number, max: number}} extent the extent of the visible data
 * @returns {Array<number|null>} `[low, high]`
 */
export function axisBounds({min, max}) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [null, null];
  }
  const low = Math.floor(min / AXIS_STEP) * AXIS_STEP;
  return [low, Math.max(Math.ceil(max / AXIS_STEP) * AXIS_STEP, low + AXIS_STEP)];
}

/**
 * Label of a tick of the vertical axis: whole percentages without decimals (the ticks are multiples of
 * `AXIS_STEP`) and, should a tick ever fall between them, one decimal (`32%`, `32,2%`).
 *
 * @param {number} value the tick value
 * @returns {string} the label
 */
export function axisPctLabel(value) {
  return fmtPct(value, Math.abs(value - Math.round(value)) < 1e-6 ? 0 : 1);
}

/**
 * Vertical mark line at `time` with a label at the top. `align` places the text on one side of the line
 * (`'right'`: the text ends at the line, so it reads to its left; `'left'`: it starts there): «Estimación» goes
 * left and «Elección» right, so the two labels never overlap on a short horizon (Luis, 2026-10-26).
 */
function markAt(time, label, align = 'center') {
  return {xAxis: time, label: {formatter: label, position: 'end', align, color: THEME.markColor}};
}

/**
 * The 95 % range of a party at the election as a vertical bar from `lo` to `hi` at `time`, with a white point at
 * `mean`: a `custom` series named after the party (the legend toggles it with its line). Hovering its point brings
 * the bar to the front at full opacity while the other bars blur, so overlapping ranges can still be read one by
 * one; the bar itself is silent, so a pointer crossing the bars activates nothing (Luis, 2026-10-26). ECharts
 * reads `focus`/`blurScope` of a custom series from the root element `renderItem` returns, not from the series:
 * they go on the group. The blur reaches every series of the chart, so the lines, points and marks declare a blur
 * state that keeps them as they are (`KEEP_ON_BLUR`): only the other bars fade.
 *
 * @param {string} name party or block
 * @param {number} time timestamp of the election day
 * @param {number} lo lower bound of the range
 * @param {number} hi upper bound of the range
 * @param {number} mean central estimate
 * @param {string} color colour of the party
 * @returns {object} the ECharts series
 */
export function rangeBarSeries(name, time, lo, hi, mean, color) {
  return {
    type: 'custom',
    name,
    color,
    z: 6,
    encode: {x: 0, y: [1, 2, 3]},
    renderItem: (params, api) => {
      const x = api.coord([api.value(0), 0])[0];
      const y = (k) => api.coord([api.value(0), api.value(k)])[1];
      return {
        type: 'group',
        focus: 'series',
        blurScope: 'coordinateSystem',
        children: [
          {
            type: 'line',
            silent: true,
            shape: {x1: x, y1: y(1), x2: x, y2: y(2)},
            style: {stroke: color, lineWidth: 4, lineCap: 'round', opacity: RANGE_OPACITY},
            emphasis: {style: {opacity: 1}},
            blur: {style: {opacity: RANGE_BLUR_OPACITY}},
          },
          {
            type: 'circle',
            shape: {cx: x, cy: y(3), r: 4},
            style: {fill: '#fff', stroke: color, lineWidth: 2},
            blur: {style: {opacity: RANGE_BLUR_OPACITY}},
          },
        ],
      };
    },
    data: [[time, lo, hi, mean]],
  };
}

/**
 * Invisible series of one party with the values of the layer not drawn, so the vertical axis spans the same
 * values whichever layers are on screen (Luis, 2026-10-28): a silent line without symbols at opacity 0, named
 * after the party, so the legend hides it with the party and the window filters it like the rest. It goes after
 * the drawn series, which keep the legend icon and the tooltip.
 *
 * @param {string} name party or block
 * @param {Array<[number, number]>} points `[time, value]` of the hidden layer
 * @returns {object} the ECharts series
 */
export function extentSeries(name, points) {
  return {
    type: 'line',
    name,
    z: 0,
    silent: true,
    symbol: 'none',
    legendHoverLink: false,
    lineStyle: {opacity: 0},
    emphasis: {disabled: true},
    data: points,
  };
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
 * Render the poll average of `parties`: per party the `mean` line, the published polls and, in forecast, the
 * dashed projection from `asOf` to `when` with the bar of its 95 % range at the election (`rangeBarSeries`), with
 * mark lines at both dates. Neither the band of the average nor the fan of the projection are drawn. The initial
 * zoom is the `window` choice (a key of `WINDOW_DAYS`: the last days before `anchor`; `all`: the whole axis).
 * The window filters the data (`filterMode: 'filter'`) and the vertical axis spans the visible data of the
 * series the legend shows, without a fixed zero, so a tight race fills the height; its ticks fall on multiples
 * of 5 (`axisBounds`) and the layers not drawn still count for it (`extentSeries`), so choosing a layer never
 * changes the scale.
 * The Promedios page also draws the blocks with it, passing the block series, polls and projection as parties.
 * The tooltip follows the item under the pointer: a poll shows its box (`pollTooltipLines`), a day of the
 * average its value, a day of the projection or the range bar its value and range (`meanTooltipLine`), the
 * official result its share (`resultTooltipLine`).
 *
 * @param {object} data `series`, `polls`, `projection` (the `data` members of the envelopes), `parties`
 *   (names, legend order), `selected` (names shown at first; all when null), `colors` ({name: colour}),
 *   `asOf`, `when` (ISO dates), `window` (choice of the window control), `anchor` (ISO date the window counts
 *   back from: the last poll), `legend` (`true`: selectable legend; `'readonly'`: a legend that only shows the
 *   key and follows `selected`; `'hidden'`: no legend drawn but a selectable one kept, for the HTML pill legend
 *   that drives it with `legendToggleSelect`; `false`: hidden, but its `selected` still decides the names shown,
 *   for pages with their own selector), `result` (`{date, values: {name: share}}`, the official result of a
 *   past election: a diamond per party with a value, at `date`, which also gets the «Elección» mark line and
 *   closes the axis; null for none), `asOfLabel` (label of the mark line at `asOf`, «Estimación» by default) and
 *   `layers` (what is drawn: `ambos`, the default, the poll points and the average; `sondeos`, the points alone,
 *   with the result and the mark lines; `promedio`, the lines, projection and range bars alone, without the points)
 * @returns {object} the ECharts option
 */
export function seriesOption({
  series, polls, projection, parties, selected = null, colors = {}, asOf, when,
  window: windowChoice = DEFAULT_WINDOW, anchor = null, legend = true, result = null, asOfLabel = 'Estimación',
  layers = 'ambos',
}) {
  const color = (name) => colors[name] || OTHERS_COLOR;
  const seriesTimes = series.dates.map(dayTime);
  const groups = (projection && projection.groups && projection.groups.parties) || null;
  // In nowcast the projection is the single day `asOf`: only the mark line is drawn.
  const withProjection = Boolean(groups && projection.dates.length > 1);
  const projectionTimes = withProjection ? projection.dates.map(dayTime) : [];
  const pollRecords = polls.polls || [];
  const colorOf = Object.fromEntries(parties.map((name) => [name, color(name)]));
  // Role of each series for the tooltip: `mean`, `projection`, `range`, `poll` or `result` (marks have none).
  const roles = new Map();
  const withRole = (role, entry) => {
    roles.set(entry, role);
    return entry;
  };

  // The layers control of the pages: `sondeos` leaves the poll points alone, `promedio` the average alone.
  const withAverage = layers !== 'sondeos';
  const withPolls = layers !== 'promedio';
  // Mean lines first: the legend takes its icon from the first series of each name.
  const means = !withAverage ? [] : parties.map((name) => withRole('mean', {
    type: 'line',
    name,
    color: color(name),
    z: 3,
    ...HOVER_SYMBOLS,
    ...KEEP_LINE_ON_BLUR,
    connectNulls: false,
    lineStyle: {width: 2},
    data: seriesTimes.map((t, i) => [t, (series.mean[name] || [])[i] ?? null]),
  }));
  const points = !withPolls ? [] : parties.map((name) => withRole('poll', {
    type: 'scatter',
    name,
    color: color(name),
    z: 4,
    symbolSize: 6,
    itemStyle: {opacity: 0.6},
    ...KEEP_POINTS_ON_BLUR(0.6),
    data: pollPoints(pollRecords, name).map(([date, value, index]) => [dayTime(date), value, index]),
  }));
  const projected = !withProjection || !withAverage ? [] : parties.map((name) => withRole('projection', {
    type: 'line',
    name,
    color: color(name),
    z: 3,
    ...HOVER_SYMBOLS,
    ...KEEP_LINE_ON_BLUR,
    connectNulls: false,
    lineStyle: {width: 2, type: 'dashed'},
    data: projectionTimes.map((t, i) => [t, (groups.mean[name] || [])[i] ?? null]),
  }));
  // The 95 % range at the election: the last day of the projection, for the parties that have one.
  const lastDay = withProjection ? projection.dates[projection.dates.length - 1] : null;
  const lastOf = (values) => (values || [])[projection.dates.length - 1] ?? null;
  const ranges = !withProjection || !withAverage ? [] : parties.flatMap((name) => {
    const [lo, hi, mean] = [lastOf(groups.lo[name]), lastOf(groups.hi[name]), lastOf(groups.mean[name])];
    return [lo, hi, mean].every(Number.isFinite)
      ? [withRole('range', rangeBarSeries(name, dayTime(lastDay), lo, hi, mean, color(name)))] : [];
  });
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
      itemStyle: {borderColor: '#fff', borderWidth: 1.5, opacity: 0.8},  // the scatter default, made explicit
      ...KEEP_POINTS_ON_BLUR(0.8),
      data: [[dayTime(result.date), result.values[name]]],
    }));
  // The layer not drawn, as invisible points per party, so the axis spans the same values in every layer.
  const finite = (points) => points.filter(([, value]) => Number.isFinite(value));
  const hiddenPoints = (name) => {
    if (!withAverage) {
      // the mean, the projection and the range at the election (`ranges` is empty in this layer)
      const electionTime = withProjection ? dayTime(lastDay) : null;
      return finite([
        ...seriesTimes.map((t, i) => [t, (series.mean[name] || [])[i] ?? null]),
        ...projectionTimes.map((t, i) => [t, (groups.mean[name] || [])[i] ?? null]),
        ...(withProjection ? [[electionTime, lastOf(groups.lo[name])], [electionTime, lastOf(groups.hi[name])]] : []),
      ]);
    }
    if (!withPolls) {
      return pollPoints(pollRecords, name).map(([date, value]) => [dayTime(date), value]);
    }
    return [];
  };
  const extents = parties.flatMap((name) => {
    const points = hiddenPoints(name);
    return points.length ? [extentSeries(name, points)] : [];
  });
  const marks = [markAt(dayTime(asOf), asOfLabel, 'right')];
  if (result) {
    marks.push(markAt(dayTime(result.date), 'Elección', 'left'));
  } else if (withProjection && when) {
    marks.push(markAt(dayTime(when), 'Elección', 'left'));
  }
  // Mark lines on their own empty series, so hiding a party never hides them.
  const markSeries = {
    type: 'line',
    name: 'marks',
    data: [],
    silent: true,
    ...KEEP_LINE_ON_BLUR,
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: {type: 'dashed', color: THEME.markColor},
      blur: {lineStyle: {opacity: 1}, label: {opacity: 1}},
      data: marks,
    },
  };

  const axisDates = axisDatesOf(series, projection, result);
  const zoom = seriesWindow(windowChoice, axisDates, anchor);

  const allSeries = [...means, ...points, ...projected, ...ranges, ...results, ...extents, markSeries];

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
    if (role === 'range') {
      return meanTooltipLine(name, lastOf(groups.mean[name]), lastOf(groups.lo[name]), lastOf(groups.hi[name]),
        lastDay, color(name), 'Elección');
    }
    if (role === 'mean') {
      // the band of the average is not drawn any more, so its interval is not shown either
      return meanTooltipLine(name, (series.mean[name] || [])[i] ?? null, null, null, series.dates[i], color(name));
    }
    if (role === 'projection') {
      const at = (values) => (values || [])[i] ?? null;
      return meanTooltipLine(name, at(groups.mean[name]), at(groups.lo[name]), at(groups.hi[name]),
        projection.dates[i], color(name), true);
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
    // Right margin wide enough for the «Elección» label, which reads to the right of the last axis date.
    // Without a drawn legend the top margin only keeps room for the mark-line labels.
    grid: {left: 8, right: 72, top: drawn ? 40 : 24, bottom: 56, containLabel: true},
    xAxis: {
      type: 'time',
      minInterval: DAY_MS,
      axisLine: {lineStyle: {color: THEME.axisColor}},
      axisLabel: timeAxisLabel(),
      splitLine: {show: false},
    },
    // Ticks every `AXIS_STEP` points on multiples of it, around the data on screen (`axisBounds`).
    yAxis: {
      type: 'value',
      min: (extent) => axisBounds(extent)[0],
      max: (extent) => axisBounds(extent)[1],
      interval: AXIS_STEP,
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
 * The current zoom of a rendered series chart: the ends of the time window on screen, after the window control
 * or a drag of the slider (ECharts keeps them up to date); null without a dataZoom.
 *
 * @param {object} chart the ECharts instance returned by `renderSeries`
 * @returns {{startValue: number, endValue: number}|null} timestamps of the ends
 */
export function chartZoom(chart) {
  const zoom = ((chart && chart.getOption().dataZoom) || [])[0];
  return zoom ? {startValue: zoom.startValue, endValue: zoom.endValue} : null;
}

/**
 * Option of the downloadable image of the poll average (`share.js`): the chart of `seriesOption` with a drawn,
 * static legend of the names shown only and, instead of the slider, an invisible zoom at `zoom` (`chartZoom` of
 * the chart on screen), so the image keeps the window, the parties and the vertical axis of the screen.
 *
 * @param {object} data the arguments of `seriesOption`
 * @param {object|null} [zoom=null] the zoom of the chart on screen (null: the `window` of `data`)
 * @returns {object} the ECharts option
 */
export function seriesImageOption(data, zoom = null) {
  const option = seriesOption({...data, legend: true});
  const shown = data.selected ? data.parties.filter((name) => data.selected.includes(name)) : data.parties;
  const [slider] = option.dataZoom;
  return {
    ...option,
    legend: {...option.legend, type: 'plain', selectedMode: false, data: shown},
    grid: {...option.grid, bottom: 8},
    dataZoom: slider ? [{type: 'inside', filterMode: 'filter', disabled: true, ...(zoom || {
      startValue: slider.startValue, endValue: slider.endValue,
    })}] : [],
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
