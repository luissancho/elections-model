// Downloadable image of a chart: the chart drawn off screen at a fixed size under a title and a subtitle, with a
// discreet signature (site and author) at the bottom right, saved as a PNG. A page module wires the download
// button of a card (`wireDownload`) with a function that returns the ECharts option to draw.
import {THEME} from './charts/base.js';

/**
 * Size of the image in CSS pixels (the PNG is `scale` times bigger): its width, the height of the chart, the
 * padding (around the texts and at the sides of the chart) and the line heights of the texts.
 */
export const IMAGE = {
  width: 800, chartHeight: 440, scale: 2,
  pad: 28, titleLine: 28, subtitleLine: 20, gap: 12, signGap: 8, signLine: 16, bottom: 20,
};

/**
 * Layout of the image: the texts given (title, subtitle, signature) with their position, font and colour, the
 * box of the chart and the size of the whole. The title and the subtitle go at the top left, the chart under
 * them between the same side margins and the signature, small and muted, under the chart at the right. The `y`
 * of a text is its top edge.
 *
 * @param {object} opts `title`, `subtitle`, `sign` (texts, each optional), `width` (of the image), `chartHeight`
 *   and `scale` (`IMAGE` by default)
 * @returns {object} `{width, height, scale, texts: [{role, text, x, y, align, font, color}], chart: {x, y, width,
 *   height}}`
 */
export function imageLayout({
  title, subtitle, sign, width = IMAGE.width, chartHeight = IMAGE.chartHeight, scale = IMAGE.scale,
} = {}) {
  const texts = [];
  let y = title || subtitle ? IMAGE.pad : 0;
  if (title) {
    texts.push({role: 'title', text: title, x: IMAGE.pad, y, align: 'left', font: `700 22px ${THEME.headFamily}`, color: THEME.textColor});
    y += IMAGE.titleLine;
  }
  if (subtitle) {
    texts.push({role: 'subtitle', text: subtitle, x: IMAGE.pad, y, align: 'left', font: `500 14px ${THEME.fontFamily}`, color: THEME.mutedColor});
    y += IMAGE.subtitleLine;
  }
  if (title || subtitle) {
    y += IMAGE.gap;
  }
  const chart = {x: IMAGE.pad, y, width: width - 2 * IMAGE.pad, height: chartHeight};
  y += chartHeight;
  if (sign) {
    y += IMAGE.signGap;
    texts.push({role: 'sign', text: sign, x: width - IMAGE.pad, y, align: 'right', font: `600 12px ${THEME.fontFamily}`, color: THEME.mutedColor});
    y += IMAGE.signLine + IMAGE.bottom;
  }
  return {width, height: y, scale, texts, chart};
}

/**
 * PNG data URL of `option` drawn off screen: a chart of `width` x `height` CSS pixels at `scale` device pixels
 * per CSS pixel, on white and without animation (so the frame taken is the final one).
 *
 * @param {object} option ECharts option
 * @param {object} size `{width, height, scale}`
 * @returns {string} the data URL
 */
export function chartDataUrl(option, {width, height, scale}) {
  const chart = window.echarts.init(document.createElement('div'), null, {renderer: 'canvas', width, height, devicePixelRatio: scale});
  try {
    chart.setOption({...option, animation: false}, true);
    return chart.getDataURL({type: 'png', pixelRatio: scale, backgroundColor: '#fff'});
  } finally {
    chart.dispose();
  }
}

/**
 * Draw `layout` (its texts and the chart image of `chartUrl`) on a white canvas and return it as a PNG blob. The
 * fonts of the texts are loaded first, so the canvas draws them with the site fonts.
 *
 * @param {object} layout `imageLayout` result
 * @param {string} chartUrl data URL of the chart (`chartDataUrl`)
 * @returns {Promise<Blob>} the PNG
 */
export async function composeImage(layout, chartUrl) {
  if (document.fonts) {
    await Promise.all(layout.texts.map((text) => document.fonts.load(text.font).catch(() => null)));
  }
  const image = new Image();
  image.src = chartUrl;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = layout.width * layout.scale;
  canvas.height = layout.height * layout.scale;
  const ctx = canvas.getContext('2d');
  ctx.scale(layout.scale, layout.scale);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, layout.width, layout.height);
  ctx.drawImage(image, layout.chart.x, layout.chart.y, layout.chart.width, layout.chart.height);
  ctx.textBaseline = 'top';
  for (const text of layout.texts) {
    ctx.font = text.font;
    ctx.fillStyle = text.color;
    ctx.textAlign = text.align;
    ctx.fillText(text.text, text.x, text.y);
  }
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('empty image'))), 'image/png');
  });
}

/**
 * Offer `blob` to the reader as a file named `file`.
 *
 * @param {Blob} blob the file
 * @param {string} file its name
 */
export function saveBlob(blob, file) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = file;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Reveal `button` (hidden without JS) and, when clicked, download the chart as a PNG: the option of
 * `optionOf(box)` (`box` is the chart box of the layout, `{width, height}` among others) drawn under the texts of
 * the button's data attributes (`title`, `subtitle`, `sign`) and saved as `file`.
 *
 * @param {HTMLButtonElement|null} button the download button (nothing happens without it)
 * @param {function} optionOf chart box → ECharts option
 * @param {function} [onError] called with a message when the image cannot be made
 */
export function wireDownload(button, optionOf, onError = null) {
  if (!button) {
    return;
  }
  button.hidden = false;
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      const {title, subtitle, sign, file} = button.dataset;
      const layout = imageLayout({title, subtitle, sign});
      const url = chartDataUrl(optionOf(layout.chart), {width: layout.chart.width, height: layout.chart.height, scale: layout.scale});
      saveBlob(await composeImage(layout, url), file || 'grafico.png');
    } catch (error) {
      console.error('Could not make the image', error);
      if (onError) {
        onError('No se pudo generar la imagen del gráfico.');
      }
    } finally {
      button.disabled = false;
    }
  });
}
