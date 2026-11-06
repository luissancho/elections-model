// The page of the hidden-vote study (`/analisis/voto-oculto`): the signed error of the left and of Vox by election
// and the strip of every party, drawn from the data embedded by the server (`initial`, `island` of
// `mtpy/lib/pages/shy.py`), and the buttons that download each card as an image.
import {renderErrorBars, renderErrorStrip} from '../charts/errors.js';
import {wireCardImages} from '../card-image.js';
import {mountWhenReady, readInitial, showError, tryDraw} from './common.js';

/** The container `#id` when it exists and there is something to draw in it. */
function container(id, ready) {
  const el = document.getElementById(id);
  return el && ready ? el : null;
}

/** Question 1: the error of the left in every election, with its mean. */
function drawLeft(data) {
  const el = container('left-chart', data.left_bars && data.left_bars.length);
  if (el) {
    tryDraw(() => renderErrorBars(el, data.left_bars, {mean: data.left_mean, rotate: 90}), 'the error of the left');
  }
}

/** Question 2: the error of Vox in every election it ran, with its mean. */
function drawVox(data) {
  const el = container('vox-chart', data.vox_bars && data.vox_bars.length);
  if (el) {
    tryDraw(() => renderErrorBars(el, data.vox_bars, {mean: data.vox_mean, yMin: 5, step: 2.5}), 'the error of Vox');
  }
}

/** Question 3: the error of each party in every election, with its mean. */
function drawParties(strip) {
  const el = container('parties-chart', strip && strip.rows && strip.rows.length);
  if (el) {
    tryDraw(() => renderErrorStrip(el, strip), 'the error by party');
  }
}

/** Draw the charts whose container is present. */
function paint() {
  const data = readInitial();
  drawLeft(data);
  drawVox(data);
  drawParties(data.party_strip);
}

mountWhenReady(paint);
wireCardImages(document, showError);
