// The page of the clandestine polls study (`/analisis/andorra`): for now only the buttons that download each card as
// an image; the dumbbell of question 1 (`#c-dumb`, from `initial.pairs`) comes with `charts/dumbbell.js`.
import {wireCardImages} from '../card-image.js';
import {showError} from './common.js';

wireCardImages(document, showError);
