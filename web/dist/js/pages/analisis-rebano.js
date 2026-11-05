// The page of the herd study (`/analisis/rebano`): the dispersion of the polls by weeks before the election and
// the house × cycle matrix, drawn from the data embedded by the server (`initial`, `island` of
// `mtpy/lib/pages/herd.py`), and the buttons that download each card as an image.
import {renderMatrix} from '../charts/matrix.js';
import {renderWeeks} from '../charts/weeks.js';
import {wireCardImages} from '../card-image.js';
import {fmtNum} from '../format.js';
import {mountWhenReady, readInitial, showError, tryDraw} from './common.js';

/** The container `#id` when it exists and there is something to draw in it. */
function container(id, ready) {
  const el = document.getElementById(id);
  return el && ready ? el : null;
}

/** Question 1: the dispersion of every cycle by weeks before the election, over the line of chance. */
function drawWeeks(weeks) {
  const el = container('weeks-chart', weeks && weeks.buckets && weeks.buckets.length && weeks.rows && weeks.rows.length);
  if (el) {
    tryDraw(() => renderWeeks(el, weeks), 'the dispersion by weeks');
  }
}

/** Question 3: the herd of each house in the final weeks of each cycle; once drawn, its table without JavaScript leaves. */
function drawMatrix(matrix) {
  const el = container('matrix-chart', matrix && matrix.houses && matrix.houses.length);
  if (!el) {
    return;
  }
  const drawn = tryDraw(() => renderMatrix(el, matrix, {
    center: 1,
    label: true,
    cellText: (house, event, h) => `${house} · ${event.label}: rebaño ${fmtNum(h, 2)}`,
    bold: (event) => event.current,
    rotate: 0,
  }), 'the matrix of the houses');
  const table = document.getElementById('matrix-table');
  if (drawn && table) {
    (table.closest('.table-wrap') || table).hidden = true;
  }
}

/** Draw the charts whose container is present. */
function paint() {
  const data = readInitial();
  drawWeeks(data.weeks);
  drawMatrix(data.matrix);
}

mountWhenReady(paint);
wireCardImages(document, showError);
