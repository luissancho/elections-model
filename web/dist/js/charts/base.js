// Shared ECharts setup. ECharts is a classic UMD script loaded before the page module.

const echarts = window.echarts;

/** Whether the reader asked the system for less motion: charts then draw without animation. */
const REDUCED_MOTION = Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/**
 * Shared chart style: the body font of `site.css` (`--font-body`), text and line colours, the neutral blue of
 * the bars without a party colour, and the animation length (zero under `prefers-reduced-motion`).
 */
export const THEME = {
  fontFamily: "'Source Sans 3', system-ui, sans-serif",
  textColor: '#1a1c22',
  gridColor: 'rgba(0,0,0,.15)',
  intervalColor: '#333',
  markColor: '#555',
  barColor: '#3a6ea5',
  animationDuration: REDUCED_MOTION ? 0 : 300,
};

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
    textStyle: {fontFamily: THEME.fontFamily, color: THEME.textColor},
  };
}
