// Downloadable image of a whole card (`.card.shot`): the card as the page shows it, laid out at a fixed width and
// rasterised with html2canvas (vendored, loaded on the first click), with its charts redrawn at that width from
// their ECharts options, without its buttons and its folded «Cómo leerlo», and with a signature line at the bottom
// right. A page module reveals and wires the square button of each card (`wireCardImages`); the file name and the
// signature travel in the button's data attributes.
import {chartDataUrl, saveBlob} from './share.js';

/** Width of the layout the card is drawn at (CSS pixels; the PNG is `scale` times bigger) and the vendored library. */
export const CARD_IMAGE = {width: 1200, scale: 2, vendor: '/dist/vendor/html2canvas-1.4.1.min.js'};

/** What leaves the image: the tools of the card, the «Ver más» buttons and the folded «Cómo leerlo». */
export const STRIPPED = '.card-img-btn, .tool-btn, button.more, details.how';

let loading = null;

/**
 * html2canvas, loaded once from the vendor folder the first time an image is asked for.
 *
 * @param {string} [src] the script
 * @returns {Promise<function>} the library
 */
export function loadHtml2canvas(src = CARD_IMAGE.vendor) {
  if (window.html2canvas) {
    return Promise.resolve(window.html2canvas);
  }
  if (!loading) {
    loading = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.onload = () => resolve(window.html2canvas);
      script.onerror = () => {
        loading = null;
        reject(new Error(`could not load ${src}`));
      };
      document.head.appendChild(script);
    });
  }
  return loading;
}

/**
 * The charts of `card` with an ECharts instance mounted (`mountChart` keeps it in `el.__chart`): their container id
 * and their current option, to redraw them at the size the image gives them.
 *
 * @param {HTMLElement} card the card
 * @returns {{id: string, option: object}[]} the charts
 */
export function chartOptions(card) {
  return [...card.querySelectorAll('.chart')]
    .filter((el) => el.id && el.__chart)
    .map((el) => ({id: el.id, option: el.__chart.getOption()}));
}

/**
 * Prepare the copy of the card in the document html2canvas renders: strip the tools (`STRIPPED`), replace each
 * chart by an image of its option at the size the copy's layout gives the container, append the signature and wait
 * for the fonts.
 *
 * @param {Document} doc the cloned document
 * @param {string} cardId id of the card
 * @param {{id: string, option: object}[]} charts the charts of the card (`chartOptions`)
 * @param {string} sign the signature line (nothing without it)
 * @param {number} scale device pixels per CSS pixel of the chart images
 */
export async function prepareClone(doc, cardId, charts, sign, scale) {
  const card = doc.getElementById(cardId);
  if (!card) {
    throw new Error(`no card #${cardId} in the clone`);
  }
  card.classList.add('shot-image');
  card.querySelectorAll(STRIPPED).forEach((el) => el.remove());
  for (const {id, option} of charts) {
    const el = doc.getElementById(id);
    if (!el) {
      continue;
    }
    el.replaceChildren();
    const width = Math.round(el.getBoundingClientRect().width);
    const height = Math.round(el.getBoundingClientRect().height);
    if (width > 0 && height > 0) {
      const image = doc.createElement('img');
      image.src = chartDataUrl(option, {width, height, scale});
      image.width = width;
      image.height = height;
      image.alt = '';
      image.style.cssText = 'display:block;width:100%;height:100%';
      el.appendChild(image);
    }
  }
  if (sign) {
    const line = doc.createElement('p');
    line.className = 'shot-sign';
    line.textContent = sign;
    card.appendChild(line);
  }
  if (doc.fonts) {
    await doc.fonts.ready;
  }
}

/**
 * PNG of `card` as the page shows it at `CARD_IMAGE.width`.
 *
 * @param {HTMLElement} card the card (with an id)
 * @param {object} [opts] `sign` (the signature line), `width` and `scale` (`CARD_IMAGE` by default)
 * @returns {Promise<Blob>} the PNG
 */
export async function cardImage(card, {sign = '', width = CARD_IMAGE.width, scale = CARD_IMAGE.scale} = {}) {
  const html2canvas = await loadHtml2canvas();
  const charts = chartOptions(card);
  const canvas = await html2canvas(card, {
    scale,
    backgroundColor: '#ffffff',
    logging: false,
    windowWidth: width,
    windowHeight: Math.max(window.innerHeight, 900),
    onclone: (doc) => prepareClone(doc, card.id, charts, sign, scale),
  });
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('empty image'))), 'image/png');
  });
}

/**
 * Reveal every card-image button of `root` (hidden without JS) and, when clicked, download its card as a PNG named
 * by the button's `data-file`, signed with its `data-sign`.
 *
 * @param {ParentNode} [root] where the buttons are
 * @param {function} [onError] called with a message when the image cannot be made
 */
export function wireCardImages(root = document, onError = null) {
  for (const button of root.querySelectorAll('button.card-img-btn')) {
    const card = button.closest('.card');
    if (!card || !card.id) {
      continue;
    }
    button.hidden = false;
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        saveBlob(await cardImage(card, {sign: button.dataset.sign}), button.dataset.file || `${card.id}.png`);
      } catch (error) {
        console.error('Could not make the image of the card', error);
        if (onError) {
          onError('No se pudo generar la imagen de la tarjeta.');
        }
      } finally {
        button.disabled = false;
      }
    });
  }
}
