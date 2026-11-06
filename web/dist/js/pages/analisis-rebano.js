// The page of the herd study (`/analisis/rebano`): the bars and the squares are HTML and CSS, so the only script
// wires the buttons that download each card as an image.
import {wireCardImages} from '../card-image.js';
import {showError} from './common.js';

wireCardImages(document, showError);
