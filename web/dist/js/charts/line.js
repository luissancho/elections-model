// A single line with markers over a time axis (e.g. the rating of a pollster at each election).
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
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
      axisLabel: {formatter: (ms) => fmtDateShort(new Date(ms).toISOString().slice(0, 10))},
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (x) => formatter(x)},
    },
    tooltip: {
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
 * @param {Array} series `{name, points: [[x, y], ...], color?, dashed?}` per line
 * @param {object} opts `formatter` (value → text), `yMax` (top of the value axis) and `references` (values
 *   drawn as labelled dashed horizontal lines)
 * @returns {object} the ECharts instance
 */
export function renderLines(el, series, {formatter = (x) => fmtNum(x, 1), yMax = undefined, references = []} = {}) {
  const categories = [...new Set(series.flatMap((line) => line.points.map(([x]) => x)))].sort((a, b) => a - b);
  const lines = series.map((line, index) => {
    const values = new Map(line.points);
    const item = {
      type: 'line',
      name: line.name,
      data: categories.map((x) => (values.has(x) ? values.get(x) : null)),
      showSymbol: true,
      symbolSize: 7,
      color: line.color || THEME.markColor,
      lineStyle: {width: 2, type: line.dashed ? 'dashed' : 'solid'},
    };
    if (index === 0 && references.length) {
      item.markLine = {
        silent: true,
        symbol: 'none',
        lineStyle: {type: 'dashed', color: THEME.markColor},
        label: {formatter: (params) => formatter(params.value), color: THEME.textColor},
        data: references.map((value) => ({yAxis: value})),
      };
    }
    return item;
  });
  const option = {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 40, bottom: 8, containLabel: true},
    legend: {top: 0, textStyle: {color: THEME.textColor}},
    xAxis: {type: 'category', data: categories, axisLabel: {formatter: (x) => `${x} d`}},
    yAxis: {
      type: 'value',
      scale: true,
      max: yMax,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (x) => formatter(x)},
    },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        const rows = params
          .filter((item) => item.value !== null && item.value !== undefined)
          .map((item) => `${escapeHtml(item.seriesName)}: ${escapeHtml(formatter(item.value))}`);
        return [`${escapeHtml(params[0].axisValue)} d`, ...rows].join('<br>');
      },
    },
    series: lines,
  };
  return mountChart(el, option);
}
