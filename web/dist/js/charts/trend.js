// Mean error of the polls of each election over time: one scatter point per election (sized by its polls, one
// series per scope), a trend line and, with a house chosen, the line of the other houses.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {fmtDate, fmtNum} from '../format.js';

const MIN_SIZE = 6;
const SIZE_RANGE = 12;
const TREND_NAME = 'Tendencia (últimas cinco elecciones)';
const REST_NAME = 'Resto de encuestadoras';
const EMPTY_TITLE = 'Sin sondeos con estos filtros';

/**
 * Render the evolution of the error by election.
 *
 * @param {HTMLElement} el chart container
 * @param {object} data `series` (`[{name, points: [{date, value, n, label, href}]}]`, one scatter per scope),
 *   `trend` (`[date, value][]`, the moving average) and `rest` (`[date, value][]` or `null`, the other houses)
 * @param {object} opts `formatter` (value → text), `colors` (`{scope name: colour}`), `unit` (tooltip unit) and
 *   `accent` (colour of the trend line)
 * @returns {object} the ECharts instance
 */
export function renderTrend(el, {series, trend, rest}, {formatter = (x) => fmtNum(x, 1), colors = {}, unit = 'puntos', accent = THEME.markColor} = {}) {
  const nMax = Math.max(1, ...series.flatMap((scope) => scope.points.map((point) => point.n)));
  const scatters = series.map((scope) => ({
    type: 'scatter',
    name: scope.name,
    color: colors[scope.name] ?? THEME.markColor,
    z: 10,
    data: scope.points.map((point) => ({
      value: [point.date, point.value],
      symbolSize: MIN_SIZE + (SIZE_RANGE * point.n) / nMax,
      point,
    })),
  }));
  const line = (name, data, style) => ({
    type: 'line', name, data, showSymbol: false, silent: true, z: 5, ...style,
  });
  const lines = [];
  if (trend.length) {
    lines.push(line(TREND_NAME, trend, {color: accent, lineStyle: {width: 2, color: accent}}));
  }
  if (rest && rest.length) {
    lines.push(line(REST_NAME, rest, {color: THEME.markColor, lineStyle: {width: 2, type: 'dashed', color: THEME.markColor}}));
  }
  const option = {
    ...baseOption(),
    grid: {left: 8, right: 24, top: 48, bottom: 8, containLabel: true},
    legend: {top: 0, type: 'scroll', textStyle: {color: THEME.textColor}},
    xAxis: {
      type: 'time',
      axisLabel: {formatter: (ms) => String(new Date(ms).getUTCFullYear())},
    },
    yAxis: {
      type: 'value',
      min: 0,
      splitLine: {lineStyle: {color: THEME.gridColor}},
      axisLabel: {formatter: (x) => formatter(x)},
    },
    tooltip: {
      trigger: 'item',
      formatter(params) {
        const point = params.data && params.data.point;
        if (point) {
          return [
            `<strong>${escapeHtml(point.label)}</strong>`,
            `media ${escapeHtml(formatter(point.value))} ${escapeHtml(unit)} · ${escapeHtml(point.n)} ${point.n === 1 ? 'sondeo' : 'sondeos'}`,
          ].join('<br>');
        }
        const [date, value] = params.value;
        return `${escapeHtml(params.seriesName)}<br>${escapeHtml(fmtDate(date))} · ${escapeHtml(formatter(value))} ${escapeHtml(unit)}`;
      },
    },
    series: [...scatters, ...lines],
  };
  if (!scatters.some((scope) => scope.data.length)) {
    option.title = {text: EMPTY_TITLE, left: 'center', top: 'middle', textStyle: {color: THEME.markColor, fontWeight: 'normal', fontSize: 14}};
    option.legend = {show: false};
  }
  const chart = mountChart(el, option);
  if (!el.__trendClick) {
    // The click handler lives once per chart and reads the datum, so redraws do not stack listeners.
    el.__trendClick = true;
    chart.on('click', (params) => {
      const point = params.data && params.data.point;
      if (point && point.href) {
        window.location.assign(point.href);
      }
    });
  }
  return chart;
}
