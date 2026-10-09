// Hemicycle: seats per party as a half doughnut, left to right.
import {baseOption, escapeHtml, mountChart, THEME} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtPct} from '../format.js';

/**
 * Pie data of the hemicycle: parties with seats, ordered left to right, labelled when large enough.
 *
 * @param {object} seats `{name: seats}`
 * @param {object} opts `colors` ({name: colour}), `order` (names → ordered names; object order when
 *   missing), `nSeats` (seats of the chamber, for the 3 % label threshold)
 * @returns {object[]} `{name, value, itemStyle, label}` items
 */
export function hemicycleData(seats, {colors = {}, order = null, nSeats = 0} = {}) {
  const names = Object.keys(seats).filter((name) => seats[name] > 0);
  const ordered = order ? order(names) : names;
  return ordered.map((name) => ({
    name,
    value: seats[name],
    itemStyle: {color: colors[name] || OTHERS_COLOR},
    label: {show: seats[name] >= nSeats * 0.03},
  }));
}

/**
 * Render the hemicycle of `seats` with the majority line written in its centre.
 *
 * @param {HTMLElement} el chart container
 * @param {object} seats `{name: seats}`
 * @param {object} opts `colors`, `order`, `nSeats`, `majority`, `fullnames` ({name: full name}, tooltip)
 * @returns {object} the ECharts instance
 */
export function renderHemicycle(el, seats, {colors = {}, order = null, nSeats = 0, majority = null, fullnames = {}} = {}) {
  const data = hemicycleData(seats, {colors, order, nSeats});
  const total = nSeats || data.reduce((sum, item) => sum + item.value, 0);
  const option = {
    ...baseOption(),
    tooltip: {
      trigger: 'item',
      formatter(params) {
        const share = total ? fmtPct((params.value / total) * 100) : '';
        const name = fullnames[params.name] || params.name;
        return escapeHtml(`${name}: ${params.value} escaños (${share})`);
      },
    },
    graphic: majority === null || majority === undefined ? [] : [{
      type: 'text',
      left: 'center',
      bottom: '26%',
      style: {
        // Two lines so the text fits inside the hole of the doughnut.
        text: `${majority} escaños\npara la mayoría`,
        align: 'center',
        lineHeight: 17,
        fill: THEME.textColor,
        font: `13px ${THEME.fontFamily}`,
      },
    }],
    series: [{
      type: 'pie',
      startAngle: 180,
      endAngle: 0,
      radius: ['45%', '85%'],
      center: ['50%', '75%'],
      avoidLabelOverlap: true,
      label: {formatter: '{b}\n{c}', lineHeight: 14},
      labelLine: {length: 6, length2: 6},
      itemStyle: {borderColor: '#fff', borderWidth: 1},
      data,
    }],
  };
  return mountChart(el, option);
}
