// The page of the herd study (`/analisis/rebano`): the bars and the squares are HTML and CSS; the script draws the
// separation between houses by weeks before the election (question 2) from the data embedded by the server
// (`initial`, built by `herd_context` of `mtpy/lib/pages/analisis.py`) and wires the buttons that download each card
// as an image (the download redraws the chart at the width of the image).
import {renderWeeks} from '../charts/weeks.js';
import {wireCardImages} from '../card-image.js';
import {mountWhenReady, readInitial, showError, tryDraw} from './common.js';

/**
 * Question 2: the separation of every general election by weeks before the election, over the line of chance; only
 * when the container is on the page (the server leaves it out without rows) and the part has buckets and rows.
 *
 * @param {object} weeks the `weeks` part (`{buckets, rows, mean}`)
 */
function drawWeeks(weeks) {
  const el = document.getElementById('weeks-chart');
  if (el && weeks && weeks.buckets && weeks.buckets.length && weeks.rows && weeks.rows.length) {
    tryDraw(() => renderWeeks(el, weeks), 'the separation by weeks');
  }
}

/** Draw the charts whose container is present. */
function paint() {
  drawWeeks(readInitial().weeks);
}

mountWhenReady(paint);
wireCardImages(document, showError);
