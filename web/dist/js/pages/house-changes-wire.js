// Wiring of «Los cambios de cada casa» on the event page: the % Voto / Escaños switch between the two tables the
// server renders (the seats one, and the switch, hidden without JS) and the download button of the card, whose image
// holds the whole table of the view shown (`card-image.js` widens the copy by what the table hides).
import {wireCardImages} from '../card-image.js';
import {markSegments} from './common.js';

/** The tables of the card by value of the switch (`input[name=hcmetric]`). */
const VIEWS = {pct: 'changes-pct', seats: 'changes-seats'};

/**
 * Wire the card of the changes of each house (`#house-changes`; nothing without it): reveal the switch
 * (`#changes-metric`, only rendered when some house gives seats) and, on its change, leave in view the table of the
 * value chosen (`#changes-pct` or `#changes-seats`) and hide the other one; reveal and wire the image button.
 *
 * @param {Document} [doc] the page
 * @param {function} [onError] called with a message when the image cannot be made
 */
export function wireHouseChanges(doc = document, onError = null) {
  const card = doc.getElementById('house-changes');
  if (!card) {
    return;
  }
  const metricSet = doc.getElementById('changes-metric');
  const tables = Object.fromEntries(Object.entries(VIEWS).map(([key, id]) => [key, doc.getElementById(id)]));
  if (metricSet && tables.pct && tables.seats) {
    // the tables start in the vote view, whatever radio the browser restored
    for (const input of metricSet.querySelectorAll('input[name=hcmetric]')) {
      input.checked = input.value === 'pct';
    }
    markSegments(metricSet);
    metricSet.hidden = false;
    metricSet.addEventListener('change', (event) => {
      const key = event.target.value;
      if (!tables[key]) {
        return;
      }
      for (const [name, table] of Object.entries(tables)) {
        table.hidden = name !== key;
      }
    });
  }
  wireCardImages(card, onError);
}
