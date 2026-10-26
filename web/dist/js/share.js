// Downloadable image of a chart: the chart (or several, one under another) drawn off screen at a fixed size under
// a title and a subtitle, with a discreet signature (site and author) at the bottom right, saved as a PNG. A page
// module wires the download button of a card (`wireDownload`) with a function that returns the ECharts option to
// draw and, optionally, a note chip under the subtitle (the text of the majority calculator) and a panel at the
// right of the chart or under it: a table of the page read from the DOM (`tableImage`), drawn on the canvas as the
// site's tables look.
import {THEME} from './charts/base.js';
import {fileSlug} from './format.js';

export {fileSlug};

/**
 * Size of the image in CSS pixels (the PNG is `scale` times bigger): its width, the height of the chart, the
 * padding (around the texts and at the sides of the chart) and the line heights of the texts.
 */
export const IMAGE = {
  width: 800, chartHeight: 440, scale: 2,
  pad: 28, titleLine: 28, subtitleLine: 20, gap: 12, signGap: 8, signLine: 16, bottom: 20, sideGap: 32,
};

/**
 * A table of an image (`tableDrawing`) as the site's tables look: the heights of the header and of a row, the
 * side padding of a cell, the party dot, the range bar of 110 px and the fonts and colours.
 */
export const TABLE = {
  headRow: 30, row: 30, padX: 8, dot: 9, dotGap: 7, barWidth: 110, barHeight: 12, barGap: 8,
  headFont: `600 10.5px ${THEME.fontFamily}`,
  font: `400 13px ${THEME.fontFamily}`,
  boldFont: `700 13px ${THEME.fontFamily}`,
  smallFont: `400 12px ${THEME.fontFamily}`,
  lineColor: 'rgba(0,0,0,.08)', headLineColor: 'rgba(0,0,0,.15)', barColor: '#e3e6ec', faintColor: '#9298ac',
};

/**
 * A note chip of an image (`imageLayout`) as the chips of the site look (`.chip`): the side padding, the height,
 * the corner radius, the font and the colours of the plain chip and of the `ok` (green) one.
 */
export const CHIP = {
  padX: 11, height: 30, r: 10,
  font: `600 13px ${THEME.fontFamily}`,
  color: THEME.textColor, background: '#eceef2', okColor: '#15803d', okBackground: '#ecfdf3',
};

/**
 * Layout of the image: the texts given (title, subtitle, signature) with their position, font and colour, the
 * boxes of the charts and the size of the whole. The title and the subtitle go at the top left, the charts under
 * them, one under another between the same side margins, and the signature, small and muted, under everything at
 * the right. The `y` of a text is its top edge. With a `note`, a chip with its text goes between the subtitle and
 * the charts. With a `side` panel, the charts narrow to leave it room at the right, and the shorter of the two
 * (the column of charts, the panel) is centred vertically against the taller. With a `bottom` panel, it goes
 * under the charts, centred, and the image widens if the panel needs it.
 *
 * @param {object} opts `title`, `subtitle`, `sign` (texts, each optional), `width` (of the image), `chartHeight`
 *   (of the single chart, when `charts` is not given) and `scale` (`IMAGE` by default), `charts` (the heights of
 *   the charts stacked, `[{height}]`), `note` (`{text, width, ok}`, the text of the chip, its measured width and
 *   whether it is green; null for none), `side` (a panel at the right of the charts, `{width, height, items}` as
 *   `tableDrawing` returns it; null for none) and `bottom` (a panel under the charts, the same shape)
 * @returns {object} `{width, height, scale, texts: [{role, text, x, y, align, font, color}], chart: {x, y, width,
 *   height}, charts: [{x, y, width, height}], note: {x, y, width, height, r, text, font, color, background}|null,
 *   side: {x, y, width, height, items}|null, bottom: {x, y, width, height, items}|null}` (`chart` is the first
 *   of `charts`)
 */
export function imageLayout({
  title, subtitle, sign, width = IMAGE.width, chartHeight = IMAGE.chartHeight, scale = IMAGE.scale, side = null,
  charts = null, note = null, bottom = null,
} = {}) {
  const heights = (charts || [{height: chartHeight}]).map((chart) => chart.height);
  if (bottom) {
    width = Math.max(width, bottom.width + 2 * IMAGE.pad);
  }
  const texts = [];
  let y = title || subtitle || note ? IMAGE.pad : 0;
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
  let chip = null;
  if (note) {
    chip = {
      x: IMAGE.pad, y, width: note.width + 2 * CHIP.padX, height: CHIP.height, r: CHIP.r, text: note.text, font: CHIP.font,
      color: note.ok ? CHIP.okColor : CHIP.color, background: note.ok ? CHIP.okBackground : CHIP.background,
    };
    y += CHIP.height + IMAGE.gap;
  }
  const column = heights.reduce((sum, height) => sum + height, 0) + (heights.length - 1) * IMAGE.gap;
  const body = side ? Math.max(column, side.height) : column;
  const chartWidth = width - 2 * IMAGE.pad - (side ? side.width + IMAGE.sideGap : 0);
  let top = y + Math.round((body - column) / 2);
  const boxes = heights.map((height) => {
    const box = {x: IMAGE.pad, y: top, width: chartWidth, height};
    top += height + IMAGE.gap;
    return box;
  });
  const panel = side ? {
    x: width - IMAGE.pad - side.width, y: y + Math.round((body - side.height) / 2), width: side.width,
    height: side.height, items: side.items,
  } : null;
  y += body;
  let foot = null;
  if (bottom) {
    y += IMAGE.gap;
    foot = {x: Math.round((width - bottom.width) / 2), y, width: bottom.width, height: bottom.height, items: bottom.items};
    y += bottom.height;
  }
  if (sign) {
    y += IMAGE.signGap;
    texts.push({role: 'sign', text: sign, x: width - IMAGE.pad, y, align: 'right', font: `600 12px ${THEME.fontFamily}`, color: THEME.mutedColor});
    y += IMAGE.signLine + IMAGE.bottom;
  }
  return {width, height: y, scale, texts, chart: boxes[0], charts: boxes, note: chip, side: panel, bottom: foot};
}

/**
 * The cells of a site table (`<table>` with a `thead` and a `tbody`), with what `tableDrawing` needs to draw
 * them: the header labels (the first column aligned left, the others right) and per cell its text, whether it
 * is bold, its colour (the arrows of a difference) and its party dot (`.pn`) or range bar (`.rng`, its
 * percentages from the inline styles the server writes).
 *
 * @param {HTMLTableElement} table the table
 * @returns {{columns: {label: string, align: string}[], rows: object[][]}} the table
 */
export function readTable(table) {
  const percent = (style) => {
    const match = /(-?[\d.]+)%/.exec(style || '');
    return match ? Number(match[1]) : null;
  };
  const columns = [...table.querySelectorAll('thead th')].map((th, i) => ({
    label: th.textContent.trim(), align: i === 0 ? 'left' : 'right',
  }));
  const rows = [...table.querySelectorAll('tbody tr')].filter((tr) => !tr.hidden).map((tr) => [...tr.cells].map((td) => {
    const party = td.querySelector('.pn');
    const badge = td.querySelector('.dlt');
    const range = td.querySelector('.rng');
    if (party) {
      const dot = party.querySelector('.dot');
      return {text: party.textContent.trim(), bold: true, dot: dot ? dot.style.backgroundColor : null};
    }
    if (badge) {
      return {text: badge.textContent.trim(), bold: true, color: window.getComputedStyle(badge).color};
    }
    if (range) {
      const [band, mark, line] = ['i', 'b', 'u'].map((tag) => range.querySelector(tag));
      const muted = td.querySelector('.mut');
      return {
        text: muted ? muted.textContent.trim() : '',
        range: {
          left: percent(band && band.style.left), width: percent(band && band.style.width),
          mark: percent(mark && mark.style.left), line: line ? percent(line.style.left) : null,
          color: band ? band.style.backgroundColor : THEME.textColor,
        },
      };
    }
    return {text: td.textContent.trim(), bold: Boolean(td.querySelector('strong, b'))};
  }));
  return {columns, rows};
}

/**
 * Drawing of a table (`readTable`) as the site draws it: upper-case muted headers over a stronger line, a thin
 * line under each row, the party dot before the name, the range bar (light track, the band of the range, the
 * central mark and the dashed majority line) before its numbers, the bars of a column lined up. Each column is
 * as wide as its widest cell.
 * Coordinates are relative to the top left corner of the table; a text item is vertically centred on its `y`.
 *
 * @param {object} table `{columns, rows}` as `readTable` returns it
 * @param {function} measure `(text, font) => width` in CSS pixels
 * @returns {{width: number, height: number, items: object[]}} the size of the table and its items: `text`
 *   (`{type, text, x, y, align, baseline, font, color}`), `rect` (`{type, x, y, width, height, r, color,
 *   opacity}`), `circle` (`{type, x, y, r, color}`) and `line` (`{type, x1, y1, x2, y2, color, width, dash}`)
 */
export function tableDrawing({columns, rows}, measure) {
  const fontOf = (cell) => (cell.bold ? TABLE.boldFont : TABLE.font);
  const contentWidth = (cell) => {
    if (cell.range) {
      return TABLE.barWidth + (cell.text ? TABLE.barGap + measure(cell.text, TABLE.smallFont) : 0);
    }
    return (cell.dot ? TABLE.dot + TABLE.dotGap : 0) + measure(cell.text || '', fontOf(cell));
  };
  const widths = columns.map((column, i) => 2 * TABLE.padX + Math.ceil(Math.max(
    measure(column.label.toUpperCase(), TABLE.headFont), ...rows.map((row) => contentWidth(row[i] || {})),
  )));
  const width = widths.reduce((sum, w) => sum + w, 0);
  const height = TABLE.headRow + rows.length * TABLE.row;
  const items = [];
  const text = (content, x, y, align, font, color) => items.push({type: 'text', text: content, x, y, align, baseline: 'middle', font, color});
  const rect = (x, y, w, h, r, color, opacity = 1) => items.push({type: 'rect', x, y, width: w, height: h, r, color, opacity});

  /** Draw `cell` in the row centred on `cy`, from `left` (its content's left edge; a range bar starts at the left
   * of its column, so the bars of a column line up, and its numbers end at `right`). */
  function drawCell(cell, left, right, cy) {
    if (cell.range) {
      const {left: from, width: span, mark, line, color} = cell.range;
      const at = (pct) => left + (pct / 100) * TABLE.barWidth;
      const top = cy - TABLE.barHeight / 2;
      rect(left, top, TABLE.barWidth, TABLE.barHeight, TABLE.barHeight / 2, TABLE.barColor);
      if (from !== null && span !== null) {
        rect(at(from), top, (span / 100) * TABLE.barWidth, TABLE.barHeight, TABLE.barHeight / 2, color, 0.45);
      }
      if (mark !== null) {
        rect(at(mark) - 1, top - 2, 3, TABLE.barHeight + 4, 2, color);
      }
      if (line !== null) {
        items.push({type: 'line', x1: at(line), y1: top - 3, x2: at(line), y2: top + TABLE.barHeight + 3, color: TABLE.faintColor, width: 1.5, dash: [3, 2]});
      }
      if (cell.text) {
        text(cell.text, right, cy, 'right', TABLE.smallFont, THEME.mutedColor);
      }
      return;
    }
    let x = left;
    if (cell.dot) {
      items.push({type: 'circle', x: x + TABLE.dot / 2, y: cy, r: TABLE.dot / 2, color: cell.dot});
      x += TABLE.dot + TABLE.dotGap;
    }
    text(cell.text || '', x, cy, 'left', fontOf(cell), cell.color || THEME.textColor);
  }

  let x = 0;
  columns.forEach((column, i) => {
    const left = column.align === 'left';
    const label = column.label.toUpperCase();
    text(label, left ? x + TABLE.padX : x + widths[i] - TABLE.padX, TABLE.headRow / 2, left ? 'left' : 'right', TABLE.headFont, THEME.mutedColor);
    rows.forEach((row, r) => {
      const cell = row[i] || {};
      const cy = TABLE.headRow + r * TABLE.row + TABLE.row / 2;
      const right = x + widths[i] - TABLE.padX;
      drawCell(cell, left || cell.range ? x + TABLE.padX : right - contentWidth(cell), right, cy);
    });
    x += widths[i];
  });
  items.push({type: 'line', x1: 0, y1: TABLE.headRow, x2: width, y2: TABLE.headRow, color: TABLE.headLineColor, width: 1.5, dash: null});
  rows.forEach((_, r) => {
    const y = TABLE.headRow + (r + 1) * TABLE.row;
    items.push({type: 'line', x1: 0, y1: y, x2: width, y2: y, color: TABLE.lineColor, width: 1, dash: null});
  });
  return {width, height, items};
}

/**
 * Drawing of the site table `table` for the side panel of an image (`tableDrawing`), measured on a canvas once
 * its fonts are loaded.
 *
 * @param {HTMLTableElement} table the table
 * @returns {Promise<object>} `{width, height, items}`
 */
export async function tableImage(table) {
  const fonts = [TABLE.headFont, TABLE.font, TABLE.boldFont, TABLE.smallFont];
  if (document.fonts) {
    await Promise.all(fonts.map((font) => document.fonts.load(font).catch(() => null)));
  }
  const ctx = document.createElement('canvas').getContext('2d');
  return tableDrawing(readTable(table), (content, font) => {
    ctx.font = font;
    return ctx.measureText(content).width;
  });
}

/**
 * Draw an item of a panel (`tableDrawing`) on `ctx`, moved by `dx`, `dy`.
 *
 * @param {CanvasRenderingContext2D} ctx the canvas
 * @param {object} item the item
 * @param {number} dx horizontal offset
 * @param {number} dy vertical offset
 */
function drawItem(ctx, item, dx, dy) {
  ctx.save();
  if (item.type === 'text') {
    ctx.font = item.font;
    ctx.fillStyle = item.color;
    ctx.textAlign = item.align;
    ctx.textBaseline = item.baseline;
    ctx.fillText(item.text, item.x + dx, item.y + dy);
  } else if (item.type === 'rect') {
    ctx.globalAlpha = item.opacity;
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.roundRect(item.x + dx, item.y + dy, item.width, item.height, Math.min(item.r, item.width / 2, item.height / 2));
    ctx.fill();
  } else if (item.type === 'circle') {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(item.x + dx, item.y + dy, item.r, 0, 2 * Math.PI);
    ctx.fill();
  } else if (item.type === 'line') {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.width;
    ctx.setLineDash(item.dash || []);
    ctx.beginPath();
    ctx.moveTo(item.x1 + dx, item.y1 + dy);
    ctx.lineTo(item.x2 + dx, item.y2 + dy);
    ctx.stroke();
  }
  ctx.restore();
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
 * Draw `layout` (its texts, the chart images of `chartUrls`, its note chip and its panels) on a white canvas and
 * return it as a PNG blob. The fonts of the texts are loaded first, so the canvas draws them with the site fonts.
 *
 * @param {object} layout `imageLayout` result
 * @param {string|string[]} chartUrls data URL of the chart (`chartDataUrl`), or one per chart of the layout
 * @returns {Promise<Blob>} the PNG
 */
export async function composeImage(layout, chartUrls) {
  const urls = Array.isArray(chartUrls) ? chartUrls : [chartUrls];
  const fonts = [...layout.texts.map((text) => text.font), ...(layout.note ? [layout.note.font] : [])];
  if (document.fonts) {
    await Promise.all(fonts.map((font) => document.fonts.load(font).catch(() => null)));
  }
  const images = await Promise.all(urls.map(async (url) => {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  }));
  const canvas = document.createElement('canvas');
  canvas.width = layout.width * layout.scale;
  canvas.height = layout.height * layout.scale;
  const ctx = canvas.getContext('2d');
  ctx.scale(layout.scale, layout.scale);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, layout.width, layout.height);
  layout.charts.forEach((box, i) => {
    if (images[i]) {
      ctx.drawImage(images[i], box.x, box.y, box.width, box.height);
    }
  });
  ctx.textBaseline = 'top';
  for (const text of layout.texts) {
    ctx.font = text.font;
    ctx.fillStyle = text.color;
    ctx.textAlign = text.align;
    ctx.fillText(text.text, text.x, text.y);
  }
  if (layout.note) {
    const {x, y, width, height, r, text, font, color, background} = layout.note;
    drawItem(ctx, {type: 'rect', x, y, width, height, r, color: background, opacity: 1}, 0, 0);
    drawItem(ctx, {type: 'text', text, x: x + CHIP.padX, y: y + height / 2, align: 'left', baseline: 'middle', font, color}, 0, 0);
  }
  for (const panel of [layout.side, layout.bottom]) {
    if (panel) {
      for (const item of panel.items) {
        drawItem(ctx, item, panel.x, panel.y);
      }
    }
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
 * The note chip of an image (`imageLayout`) with its text measured in the chip font, once the font is loaded.
 *
 * @param {{text: string, ok: boolean}|null} note the text and whether the chip is green
 * @returns {Promise<object|null>} `{text, ok, width}`, or null without a note
 */
async function measuredNote(note) {
  if (!note) {
    return null;
  }
  if (document.fonts) {
    await document.fonts.load(CHIP.font).catch(() => null);
  }
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.font = CHIP.font;
  return {text: note.text, ok: Boolean(note.ok), width: ctx.measureText(note.text).width};
}

/**
 * Reveal `button` (hidden without JS) and, when clicked, download the chart as a PNG: the option of
 * `optionOf(box)` (`box` is the chart box of the layout, `{width, height}` among others) drawn under the texts of
 * the button's data attributes (`title`, `subtitle`, `sign`) and saved as `file`, with the note chip of `note()`
 * under the subtitle, the panel of `side()` at its right and the panel of `bottom()` under it when given. With
 * `charts`, several charts go one under another instead of the single one of `optionOf`.
 *
 * @param {HTMLButtonElement|null} button the download button (nothing happens without it)
 * @param {function|null} optionOf chart box → ECharts option (unused with `charts`)
 * @param {function} [onError] called with a message when the image cannot be made
 * @param {object} [image] `width` and `chartHeight` of the image (`IMAGE` by default), `charts` (the charts
 *   stacked, `[{height, option}]` with `option` as `optionOf`, or a function returning them at the moment of the
 *   click), `note` (`() => {text, ok}|null`, the chip of the moment), `side` and `bottom` (async, return a panel,
 *   `tableImage`) and `texts` (`(dataset) => {title, subtitle, sign, file}`, the texts of the moment when they
 *   follow a control of the card, over those of the data attributes)
 */
export function wireDownload(button, optionOf, onError = null, {
  width, chartHeight, side = null, texts = null, charts = null, note = null, bottom = null,
} = {}) {
  if (!button) {
    return;
  }
  button.hidden = false;
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      const {title, subtitle, sign, file} = {...button.dataset, ...(texts ? texts({...button.dataset}) : {})};
      const stack = charts ? (typeof charts === 'function' ? charts() : charts) : null;
      const options = stack ? stack.map((chart) => chart.option) : [optionOf];
      const layout = imageLayout({
        title, subtitle, sign, width, chartHeight, side: side ? await side() : null, bottom: bottom ? await bottom() : null,
        charts: stack ? stack.map(({height}) => ({height})) : null, note: await measuredNote(note ? note() : null),
      });
      const urls = layout.charts.map((box, i) => chartDataUrl(options[i](box), {width: box.width, height: box.height, scale: layout.scale}));
      saveBlob(await composeImage(layout, urls), file || 'grafico.png');
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
