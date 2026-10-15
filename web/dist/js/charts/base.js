// Shared ECharts setup. ECharts is a classic UMD script loaded before the page module.

const echarts = window.echarts;

/** Whether the reader asked the system for less motion: charts then draw without animation. */
const REDUCED_MOTION = Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/**
 * Shared chart style: the body and heading fonts of `site.css` (`--font-body`, `--font-head`), text, muted, grid
 * and axis colours, the neutral blue of the bars without a party colour, and the animation length (zero under
 * `prefers-reduced-motion`).
 */
export const THEME = {
  fontFamily: "'DM Sans', system-ui, sans-serif",
  headFamily: "'Outfit', 'DM Sans', sans-serif",
  textColor: '#1a1c22',
  mutedColor: '#5a5f72',
  gridColor: 'rgba(0,0,0,.08)',
  axisColor: 'rgba(0,0,0,.15)',
  intervalColor: '#333',
  markColor: '#555',
  accentColor: '#1d4f86',  // the site accent (`--color-accent`): the points of a single-series bullseye
  barColor: '#3a6ea5',
  animationDuration: REDUCED_MOTION ? 0 : 300,
};

/** Style of the tooltips: a white card with the radius and shadow of the site (`--radius-sm`, `--shadow`). */
export function tooltipStyle() {
  return {
    backgroundColor: '#fff',
    borderColor: 'rgba(0,0,0,.08)',
    borderWidth: 1,
    borderRadius: 10,
    padding: [8, 12],
    textStyle: {color: THEME.textColor, fontSize: 12.5},
    extraCssText: 'box-shadow: 0 8px 24px rgba(16,24,40,.08);',
  };
}

/** Style of the labels of the axes: small and muted. */
export function axisLabelStyle() {
  return {color: THEME.mutedColor, fontSize: 10.5};
}

/** Create (or reuse) the chart of `el`, apply `option` and keep it resized. */
export function mountChart(el, option) {
  if (el.__chart) {
    el.__chart.setOption(option, true);
    return el.__chart;
  }
  const chart = echarts.init(el, null, {renderer: 'canvas'});
  chart.setOption(option, true);
  new ResizeObserver(() => chart.resize()).observe(el);
  el.__chart = chart;
  return chart;
}

/** Escape text for the HTML tooltips of ECharts. */
export function escapeHtml(text) {
  return window.echarts.format.encodeHTML(String(text));
}

/** Option fragment shared by every chart: animation length and text style. */
export function baseOption() {
  return {
    animation: !REDUCED_MOTION,
    animationDuration: THEME.animationDuration,
    textStyle: {fontFamily: THEME.fontFamily, color: THEME.textColor, fontSize: 11},
  };
}
