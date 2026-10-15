// Hemicycle: seats per party as a half doughnut, left to right, with the seats written inside the big arcs, the
// total in the centre and the absolute majority marked by a dark wedge and a pill above the arc.
import {baseOption, escapeHtml, mountChart, THEME, tooltipStyle} from './base.js';
import {OTHERS_COLOR} from '../catalog.js';
import {fmtPct} from '../format.js';

/** Share of the chamber from which an arc carries its label inside (the smaller ones only have the tooltip). */
const LABEL_SHARE = 0.05;
/** Width of the majority wedge, in seats. */
const MARK_SEATS = 0.7;
/** Outer radius from which the hole holds the long subtitle. */
const SUBTITLE_RADIUS = 190;

/**
 * Pie data of the hemicycle: parties with seats, ordered left to right, labelled when large enough.
 *
 * @param {object} seats `{name: seats}`
 * @param {object} opts `colors` ({name: colour}), `order` (names → ordered names; object order when
 *   missing), `nSeats` (seats of the chamber, for the label threshold)
 * @returns {object[]} `{name, value, itemStyle, label}` items
 */
export function hemicycleData(seats, {colors = {}, order = null, nSeats = 0} = {}) {
  const names = Object.keys(seats).filter((name) => seats[name] > 0);
  const ordered = order ? order(names) : names;
  const total = nSeats || names.reduce((sum, name) => sum + seats[name], 0);
  return ordered.map((name) => ({
    name,
    value: seats[name],
    itemStyle: {color: colors[name] || OTHERS_COLOR},
    label: {show: seats[name] >= total * LABEL_SHARE},
  }));
}

/**
 * ECharts option of the hemicycle of `seats`, drawn in a box of `width` x `height` pixels: the outer radius is
 * the largest half circle that fits, centred near the bottom edge.
 *
 * @param {object} seats `{name: seats}`
 * @param {object} opts `colors`, `order`, `nSeats`, `majority` (null for none), `fullnames` ({name: full name},
 *   tooltip), `width` and `height` of the container
 * @returns {object} the option
 */
export function hemicycleOption(seats, {
  colors = {}, order = null, nSeats = 0, majority = null, fullnames = {}, width = 600, height = 330,
} = {}) {
  const data = hemicycleData(seats, {colors, order, nSeats});
  const total = nSeats || data.reduce((sum, item) => sum + item.value, 0);
  const radius = Math.max(40, Math.min(width / 2 - 10, height - 36));
  const center = [width / 2, height - 12];
  const withMajority = majority !== null && majority !== undefined;
  const series = [{
    type: 'pie',
    startAngle: 180,
    endAngle: 0,
    radius: [Math.round(0.52 * radius * 10) / 10, radius],
    center,
    avoidLabelOverlap: false,
    label: {
      position: 'inside',
      formatter: '{name|{b}}\n{seats|{c}}',
      rich: {
        name: {fontFamily: THEME.headFamily, fontWeight: 700, fontSize: 15, color: '#fff', align: 'center'},
        seats: {fontFamily: THEME.headFamily, fontWeight: 800, fontSize: 24, color: '#fff', align: 'center'},
      },
    },
    labelLine: {show: false},
    itemStyle: {borderColor: '#fff', borderWidth: 1.5},
    emphasis: {scale: false},
    data,
  }];
  if (withMajority) {
    // A silent second ring, a little wider than the first, whose only visible datum is a thin dark wedge at the
    // angle of the majority
    series.push({
      type: 'pie',
      silent: true,
      startAngle: 180,
      endAngle: 0,
      radius: [Math.round(0.5 * radius * 10) / 10, radius + 10],
      center,
      label: {show: false},
      labelLine: {show: false},
      tooltip: {show: false},
      emphasis: {disabled: true},
      itemStyle: {borderWidth: 0},
      data: [
        {value: majority - MARK_SEATS / 2, itemStyle: {color: 'transparent'}},
        {value: MARK_SEATS, itemStyle: {color: THEME.textColor}},
        {value: total - majority - MARK_SEATS / 2, itemStyle: {color: 'transparent'}},
      ],
    });
  }
  const text = (x, y, content, style) => ({
    type: 'text', left: null, top: null, x, y, silent: true, z: 10,
    style: {text: content, textAlign: 'center', textVerticalAlign: 'bottom', ...style},
  });
  const graphic = [];
  if (withMajority) {
    const top = center[1] - radius - 24;
    graphic.push({
      type: 'rect', silent: true, z: 10, shape: {x: center[0] - 16, y: top, width: 32, height: 16, r: 8},
      style: {fill: THEME.textColor},
    });
    graphic.push(text(center[0], top + 12.5, String(majority), {
      fontFamily: THEME.fontFamily, fontWeight: 700, fontSize: 10, fill: '#fff',
    }));
  }
  graphic.push(text(center[0], center[1] - 30, String(total), {
    fontFamily: THEME.headFamily, fontWeight: 800, fontSize: 34, fill: THEME.textColor,
  }));
  // The long subtitle only fits the hole of a big hemicycle; the pill already names the majority
  const subtitle = withMajority && radius >= SUBTITLE_RADIUS ? `ESCAÑOS · MAYORÍA ABSOLUTA ${majority}` : 'ESCAÑOS';
  graphic.push(text(center[0], center[1] - 14, subtitle, {
    fontFamily: THEME.fontFamily, fontWeight: 600, fontSize: 10.5, fill: THEME.mutedColor, letterSpacing: 1,
  }));
  return {
    ...baseOption(),
    tooltip: {
      ...tooltipStyle(),
      trigger: 'item',
      formatter(params) {
        const share = total ? fmtPct((params.value / total) * 100) : '';
        const name = fullnames[params.name] || params.name;
        return escapeHtml(`${name}: ${params.value} escaños (${share})`);
      },
    },
    graphic,
    series,
  };
}

/**
 * Render the hemicycle of `seats` in `el`, sized to the element and redrawn when it changes size (the radius is
 * in pixels, so a plain `resize` would not do).
 *
 * @param {HTMLElement} el chart container
 * @param {object} seats `{name: seats}`
 * @param {object} opts `colors`, `order`, `nSeats`, `majority`, `fullnames` (see `hemicycleOption`)
 * @returns {object} the ECharts instance
 */
export function renderHemicycle(el, seats, opts = {}) {
  const draw = () => mountChart(el, hemicycleOption(seats, {
    ...opts, width: el.clientWidth || 600, height: el.clientHeight || 330,
  }));
  const chart = draw();
  if (el.__hemiObserver) {
    el.__hemiObserver.disconnect();
  }
  el.__hemiObserver = new ResizeObserver(() => {
    if (el.__chart) {
      el.__chart.resize();
      el.__chart.setOption(hemicycleOption(seats, {
        ...opts, width: el.clientWidth || 600, height: el.clientHeight || 330,
      }), true);
    }
  });
  el.__hemiObserver.observe(el);
  return chart;
}
