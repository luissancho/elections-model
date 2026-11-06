// The page of the clandestine polls study (`/analisis/andorra`): the dumbbell of question 1, drawn from the data
// embedded by the server (`initial.pairs`), and the buttons that download each card as an image.
import {renderDumbbell} from '../charts/dumbbell.js';
import {wireCardImages} from '../card-image.js';
import {mountWhenReady, readInitial, showError, tryDraw} from './common.js';

/** Question 1: the error of each house with its last legal poll and with its last clandestine one. */
function paint() {
  const {pairs} = readInitial();
  const el = document.getElementById('c-dumb');
  if (el && pairs && pairs.length) {
    tryDraw(() => renderDumbbell(el, pairs), 'the dumbbell of the pairs');
  }
}

mountWhenReady(paint);
wireCardImages(document, showError);
