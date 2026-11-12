// Page «Escaños que bailan» (`/{scope}/circunscripciones`): the server draws everything, the tokens included, and the
// list of districts sorts itself with CSS. The script only submits the dropdown of districts of the phones on change
// (`#district-pick`; without JavaScript its «Ver» button does it) and wires the download of the card as an image
// (`card-image.js`).
import {wireCardImages} from '../card-image.js';
import {mountWhenReady, showError, tryDraw, wireForm} from './common.js';

/** Wire the page, each part on its own so that a failure does not stop the other. */
function wire() {
  tryDraw(() => wireForm('district-pick'), 'the dropdown of districts');
  tryDraw(() => wireCardImages(document, showError), 'the download of the card');
}

mountWhenReady(wire);
