// Vertical bars of the vote of one poll (its card): one column per party, from the largest, with the share over the
// column and, under it, its change against the previous delivery of the house, coloured as the `.dlt` badges (green
// up, red down, grey when it does not change), as the party cards of the event page write theirs.
import {axisLabelStyle, baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {fmtBadge} from '../format.js';
import {niceMax} from './bars.js';

/** Width (px) under which the columns go compact: thirteen parties do not fit a phone with the full labels. */
export const COMPACT_WIDTH = 560;

/** Colours of a change by its direction, those of the `.dlt` badges of `site.css`. */
export const CHANGE_COLORS = {up: '#2ca02c', down: '#d62728', flat: THEME.mutedColor};

/**
 * Text and colour of the change of a column: the arrow and magnitude at one decimal (`fmtBadge`), or null without a
 * change (no previous delivery, or no figure in it).
 *
 * @param {number|null|undefined} change the change in points
 * @returns {{text: string, color: string}|null} the change to write under the share
 */
export function changeText(change) {
  if (change === null || change === undefined || !Number.isFinite(change)) {
    return null;
  }
  const {dir, text} = fmtBadge(change, 1);
  return {text, color: CHANGE_COLORS[dir]};
}

/**
 * ECharts option of the columns of a poll.
 *
 * @param {{name: string, value: number, change: (number|null)}[]} rows the parties in the order to draw (and «Otros»)
 * @param {object} [opts] `colors` ({name: colour}), `fullnames` ({name: full name}, tooltip), `formatter` (share →
 *   text), `axisFormatter` (ticks, default `formatter`), `shortFormatter` (the share over a column in the compact
 *   layout, default `formatter`) and `compact` (smaller labels, the short share and the names of the parties
 *   tilted, for a phone: `COMPACT_WIDTH`)
 * @returns {object} the option: a `bar` series whose label writes the share and, when the row has a change, a second
 *   line with it, bold and coloured (rich text); the axis leaves room over the tallest column for the two lines
 */
export function columnsOption(rows, {
  colors = {}, fullnames = {}, formatter = String, axisFormatter = formatter, shortFormatter = formatter, compact = false,
} = {}) {
  const changes = rows.map((row) => changeText(row.change));
  const size = compact ? {value: 9, change: 9, axis: 9} : {value: 12, change: 11, axis: null};
  const top = Math.max(0, ...rows.map((row) => row.value));
  const label = {
    show: true,
    position: 'top',
    distance: 4,
    formatter(params) {
      const change = changes[params.dataIndex];
      const value = `{v|${(compact ? shortFormatter : formatter)(params.value)}}`;
      return change ? `${value}\n{d${params.dataIndex}|${change.text}}` : value;
    },
    rich: {
      v: {fontFamily: THEME.fontFamily, fontSize: size.value, fontWeight: 600, color: THEME.textColor, align: 'center'},
      ...Object.fromEntries(changes.map((change, i) => [`d${i}`, change ? {
        fontFamily: THEME.fontFamily, fontSize: size.change, fontWeight: 700, color: change.color, align: 'center',
        padding: [compact ? 1 : 2, 0, 0, 0],
      } : {}])),
    },
  };
  return {
    ...baseOption(),
    grid: {left: 8, right: 8, top: 16, bottom: 8, containLabel: true},
    xAxis: {
      type: 'category',
      data: rows.map((row) => row.name),
      axisTick: {show: false},
      axisLabel: {
        ...axisLabelStyle(), interval: 0, hideOverlap: false,
        ...(compact ? {fontSize: size.axis, rotate: 45} : {}),
      },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: niceMax(top * 1.25),
      // the end of the axis is only room for the labels over the tallest column: neither its line nor its label (a
      // «44 %» next to the «40 %» tick)
      splitLine: {lineStyle: {color: THEME.gridColor}, showMaxLine: false},
      axisLabel: {...axisLabelStyle(), formatter: (x) => axisFormatter(x), showMaxLabel: false},
    },
    tooltip: {
      ...tooltipStyle(),
      trigger: 'axis',
      axisPointer: {type: 'shadow'},
      formatter(params) {
        const i = params[0].dataIndex;
        const row = rows[i];
        if (!row) {
          return '';
        }
        const lines = [`<strong>${escapeHtml(fullnames[row.name] || row.name)}</strong>`, escapeHtml(formatter(row.value))];
        if (changes[i]) {
          lines.push(escapeHtml(`${changes[i].text} frente a la entrega anterior`));
        }
        return lines.join('<br>');
      },
    },
    series: [{
      type: 'bar',
      barMaxWidth: 44,
      barCategoryGap: '28%',
      data: rows.map((row) => ({value: row.value, itemStyle: {color: colors[row.name] || THEME.barColor, borderRadius: [3, 3, 0, 0]}})),
      label,
    }],
  };
}

/**
 * Render the columns of a poll in `el` (see `columnsOption`), compact when the container is narrower than
 * `COMPACT_WIDTH` and redrawn when it crosses it. The container keeps `__imageOption`, the option at the size the
 * image of the card gives it (`card-image.js`), so the image is the same from a phone and from a desktop.
 *
 * @param {HTMLElement} el chart container
 * @param {object[]} rows the rows of `columnsOption`
 * @param {object} [opts] the options of `columnsOption` (but `compact`, which follows the width)
 * @returns {object} the ECharts instance
 */
export function renderColumns(el, rows, opts = {}) {
  const optionAt = (width) => columnsOption(rows, {...opts, compact: width < COMPACT_WIDTH});
  let compact = (el.clientWidth || 800) < COMPACT_WIDTH;
  const chart = mountChart(el, optionAt(el.clientWidth || 800));
  el.__imageOption = (width) => optionAt(width);
  if (el.__columnsObserver) {
    el.__columnsObserver.disconnect();
  }
  el.__columnsObserver = new ResizeObserver(() => {
    const now = (el.clientWidth || 800) < COMPACT_WIDTH;
    if (el.__chart && now !== compact) {
      compact = now;
      el.__chart.setOption(optionAt(el.clientWidth || 800), true);
    }
  });
  el.__columnsObserver.observe(el);
  return chart;
}
