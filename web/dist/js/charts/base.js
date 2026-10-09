// Shared ECharts setup. ECharts is a classic UMD script loaded before the page module.

const echarts = window.echarts;

export const THEME = {
  fontFamily: 'system-ui, sans-serif',
  textColor: '#333',
  gridColor: '#e5e5e5',
  animationDuration: 300,
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
    animationDuration: THEME.animationDuration,
    textStyle: {fontFamily: THEME.fontFamily, color: THEME.textColor},
  };
}
