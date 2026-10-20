// Bars grouped (or stacked) by category, one series per measure, with the values written on the bars (the CERA
// study: participation by origin, the change of the left vote, the additions per period, the gap to the last seat).
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';

// The stack of the stacked bars, shared by the series and by the helper series of the totals.
const STACK = 'total';

/**
 * Where to write the value of a bar: inside a stacked one; otherwise past its end, on the side it grows to.
 *
 * @param {(number|null)} value the value of the bar
 * @param {boolean} horizontal whether the bars grow sideways
 * @param {boolean} stacked whether the bars are stacked
 * @returns {string} an ECharts label position
 */
function labelPosition(value, horizontal, stacked) {
  if (stacked) {
    return 'inside';
  }
  const negative = Number.isFinite(value) && value < 0;
  if (horizontal) {
    return negative ? 'left' : 'right';
  }
  return negative ? 'bottom' : 'top';
}

/**
 * ECharts option of the grouped bars: one `bar` series per entry of `series` over the `categories`, side by side
 * (or in one stack per category with `stacked`), the value of each bar written on it, in the series' colour or,
 * for a data item `{value, color}`, in its own. With `totals` (stacked only), a text per category above its stack,
 * written by an empty helper series outside the legend.
 *
 * @param {string[]} categories the categories, in order (from the left, or from the top when horizontal)
 * @param {{name: string, data: Array, color?: string, labels?: string[], labelColor?: string, formatter?: Function}[]}
 *   series one per measure: `data` holds a number (or null) or `{value, color}` per category, `labels` the text
 *   written on each bar instead of its value, `labelColor` the colour of that text (white inside a stack, the text
 *   colour outside, by default) and `formatter` the series' own number → text
 * @param {object} opts `formatter` (number → text of the labels and the tooltip), `axisFormatter` (value axis,
 *   `formatter` by default), `horizontal` (bars growing sideways), `stacked`, `max` (of the value axis), `totals`
 *   (text above each stack) and `legend` (`'top'` or `'bottom'`)
 * @returns {object} the option
 */
export function groupedOption(categories, series, {
  formatter = String, axisFormatter = formatter, horizontal = false, stacked = false, max = null, totals = null,
  legend = 'top',
} = {}) {
  const bars = series.map((entry) => {
    const format = entry.formatter || formatter;
    const bar = {
      type: 'bar',
      name: entry.name,
      barMaxWidth: stacked ? 110 : 40,
      label: {
        show: true,
        color: entry.labelColor || (stacked ? '#fff' : THEME.textColor),
        fontSize: 11,
        fontWeight: 700,
        formatter: (params) => (Number.isFinite(params.value) ? format(params.value) : ''),
      },
      data: entry.data.map((raw, i) => {
        const value = raw !== null && typeof raw === 'object' ? raw.value : raw;
        const item = {value: Number.isFinite(value) ? value : null, label: {position: labelPosition(value, horizontal, stacked)}};
        if (raw !== null && typeof raw === 'object' && raw.color) {
          item.itemStyle = {color: raw.color};
        }
        if (entry.labels) {
          item.label.formatter = entry.labels[i] ?? '';
        }
        return item;
      }),
    };
    if (entry.color) {
      bar.color = entry.color;
    }
    if (stacked) {
      bar.stack = STACK;
    }
    return bar;
  });
  if (stacked && totals) {
    bars.push({
      type: 'bar',
      name: '',
      stack: STACK,
      silent: true,
      tooltip: {show: false},
      itemStyle: {color: 'transparent'},
      label: {show: true, color: THEME.textColor, fontSize: 12, fontWeight: 700},
      data: totals.map((text) => ({value: 0, label: {position: horizontal ? 'right' : 'top', formatter: text ?? ''}})),
    });
  }

  const valueAxis = {
    type: 'value',
    splitLine: {lineStyle: {color: THEME.gridColor}},
    axisLabel: {...axisLabelStyle(), hideOverlap: true, formatter: (value) => axisFormatter(value)},
  };
  if (max !== null && max !== undefined) {
    valueAxis.max = max;
  } else if (!stacked) {
    valueAxis.boundaryGap = ['10%', '10%'];  // room for the labels past the ends of the longest bars
  }
  const categoryAxis = {
    type: 'category',
    data: categories,
    axisTick: {show: false},
    // at the edge of the grid, not on the zero: the names never fall on the negative bars
    axisLine: {onZero: false},
    axisLabel: {color: THEME.textColor, fontSize: 11, interval: 0},  // every category named, on a phone too
  };
  if (horizontal) {
    categoryAxis.inverse = true;
  }
  return {
    ...baseOption(),
    grid: {left: 8, right: 24, top: legend === 'top' ? 36 : 12, bottom: legend === 'bottom' ? 36 : 8, containLabel: true},
    // one line on a phone too: the names that do not fit page with arrows
    legend: {type: 'scroll', [legend === 'bottom' ? 'bottom' : 'top']: 0, data: series.map((entry) => entry.name),
      textStyle: {color: THEME.mutedColor, fontSize: 11.5}},
    xAxis: horizontal ? valueAxis : categoryAxis,
    yAxis: horizontal ? categoryAxis : valueAxis,
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter(params) {
        const lines = params
          .filter((item) => item.seriesName && Number.isFinite(item.value))
          .map((item) => {
            const format = series[item.seriesIndex].formatter || formatter;
            return `${item.marker}${escapeHtml(item.seriesName)}: ${escapeHtml(format(item.value))}`;
          });
        return [`<strong>${escapeHtml(params[0].axisValue)}</strong>`, ...lines].join('<br>');
      },
    },
    series: bars,
  };
}

/**
 * Render the grouped bars (`groupedOption`) in `el`.
 *
 * @param {HTMLElement} el chart container
 * @param {string[]} categories the categories
 * @param {object[]} series one per measure
 * @param {object} opts see `groupedOption`
 * @returns {object} the ECharts instance
 */
export function renderGrouped(el, categories, series, opts = {}) {
  return mountChart(el, groupedOption(categories, series, opts));
}
