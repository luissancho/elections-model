// The page of the herd study (`/analisis/rebano`): the bars and the squares are HTML and CSS; the script draws the
// separation between houses by weeks before the election (question 2) and what chance would give against what was
// observed (question 3) from the data embedded by the server (`initial`, built by `herd_context` of
// `mtpy/lib/pages/analisis.py`) and wires the buttons that download each card as an image (the download redraws the
// charts at the width of the image).
import {renderChance} from '../charts/chance.js';
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

/**
 * Question 3: the histogram of the shuffles of the null-hypothesis test with the observed change; only when the
 * container is on the page (the server leaves card 3 out without the `chance` part) and the block has bins.
 *
 * @param {object} chance the `chance` block of the part (`{change, null: {bins, ...}, ...}`)
 */
function drawChance(chance) {
  const el = document.getElementById('chance-chart');
  if (el && chance && chance.null && chance.null.bins && chance.null.bins.length) {
    tryDraw(() => renderChance(el, chance), 'what chance would give');
  }
}

/** Draw the charts whose container is present. */
function paint() {
  const initial = readInitial();
  drawWeeks(initial.weeks);
  drawChance(initial.chance);
}

mountWhenReady(paint);
wireCardImages(document, showError);
