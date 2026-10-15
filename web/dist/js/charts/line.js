// A single line with markers over a time axis (e.g. the rating of a pollster at each election).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtDate, fmtDateShort, fmtNum} from '../format.js';

/**
 * Render one line over a time axis.
 *
 * @param {HTMLElement} el chart container
 * @param {Array} points `[iso_date, value]` pairs, ascending by date
 * @param {object} opts `formatter` (value → text), `name` (series name), `color` (line colour)
 * @returns {object} the ECharts instance
 */
export function renderLine(el, points, {formatter = (x) => fmtNum(x, 1), name = '', color = THEME.markColor} = {}) {
  const option = {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 24, bottom: 8, containLabel: true},
    xAxis: {
      type: 'time',
      axisLabel: {...axisLabelStyle(), formatter: (ms) => fmtDateShort(new Date(ms).toISOString().slice(0, 10))},
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (x) => formatter(x)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      formatter(params) {
        const [date, value] = params[0].value;
        return `${escapeHtml(fmtDate(date))} · ${escapeHtml(formatter(value))}`;
      },
    },
    series: [{
      type: 'line',
      name,
      data: points,
      showSymbol: true,
      symbolSize: 7,
      color,
      lineStyle: {width: 2},
    }],
  };
  return mountChart(el, option);
}

/**
 * Render several lines over a category axis made of the sorted union of their `x` values (e.g. the horizons).
 *
 * @param {HTMLElement} el chart container
 * @param {Array} series `{name, points: [[x, y], ...], color?, dashed?}` per line; without `color` ECharts picks one from its palette
 * @param {object} opts `formatter` (value → text), `yMin` and `yMax` (bounds of the value axis; `null` lets ECharts choose) and `references` (values
 *   drawn as labelled dashed horizontal lines)
 * @returns {object} the ECharts instance
 */
export function renderLines(el, series, {formatter = (x) => fmtNum(x, 1), yMin = null, yMax = null, references = []} = {}) {
  const categories = [...new Set(series.flatMap((line) => line.points.map(([x]) => x)))].sort((a, b) => a - b);
  const lines = series.map((line) => {
    const values = new Map(line.points);
    const item = {
      type: 'line',
      name: line.name,
      data: categories.map((x) => (values.has(x) ? values.get(x) : null)),
      showSymbol: true,
      symbolSize: 7,
      lineStyle: {width: 2, type: line.dashed ? 'dashed' : 'solid'},
    };
    if (line.color) {
      item.color = line.color;
    }
    return item;
  });
  if (references.length) {
    // The references hang from a helper series with no data, outside the legend, so hiding a line never hides them.
    lines.push({
      type: 'line',
      name: '',
      data: [],
      silent: true,
      legendHoverLink: false,
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: {type: 'dashed', color: THEME.markColor},
        label: {formatter: (params) => formatter(params.value), color: THEME.textColor},
        data: references.map((value) => ({yAxis: value})),
      },
    });
  }
  const option = {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 40, bottom: 8, containLabel: true},
    legend: {top: 0, data: series.map((line) => line.name), textStyle: {color: THEME.textColor}},
    xAxis: {type: 'category', data: categories, axisLabel: {...axisLabelStyle(), formatter: (x) => `${x} d`}},
    yAxis: {
      type: 'value',
      scale: true,
      min: yMin,
      max: yMax,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {...axisLabelStyle(), formatter: (x) => formatter(x)},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      formatter(params) {
        const rows = params
          .filter((item) => item.seriesName && item.value !== null && item.value !== undefined)
          .map((item) => `${escapeHtml(item.seriesName)}: ${escapeHtml(formatter(item.value))}`);
        return [`${escapeHtml(params[0].axisValue)} d`, ...rows].join('<br>');
      },
    },
    series: lines,
  };
  return mountChart(el, option);
}
