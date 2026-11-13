// Pure helpers of «Escaños que bailan» (the card of the event page and the page `/{scope}/circunscripciones`): the
// name of the image of a card, the line that says the filters of the map, after what is on view, whether a box is out
// of the window, and the canonical of the tile checked. No imports, so node tests them as they are.

/**
 * Name of the image of a card: its base and the parts of what is on view that are not empty, joined by `-`.
 *
 * @param {string} base the name without extension (`data-file-base` of the card)
 * @param {string[]} parts slugs of what is on view (the party and the state of the map, the chart); empty ones skip
 * @returns {string} the file name, ending in `.png`
 */
export function imageFile(base, parts) {
  return [base, ...parts.filter(Boolean)].join('-') + '.png';
}

/**
 * The line of the image of the map that says its filters: the labels of the party and of the state chosen, those
 * not empty joined by « · ».
 *
 * @param {string} party label of the party chosen (empty for «Todos»)
 * @param {string} state label of the state chosen (empty for «Todos»)
 * @returns {string} the line, empty when both filters are on «Todos»
 */
export function filterLine(party, state) {
  return [party, state].filter(Boolean).join(' · ');
}

/**
 * Whether a box is out of view: none of it shows in a window of that height (an edge on the border of the window shows
 * nothing either).
 *
 * @param {{top: number, bottom: number}} rect the box, relative to the window (`getBoundingClientRect`)
 * @param {number} height the height of the window (`innerHeight`)
 * @returns {boolean} true when the box is wholly below or wholly above the window
 */
export function outOfView(rect, height) {
  return rect.top >= height || rect.bottom <= 0;
}

/**
 * The canonical of the address of a tile: the origin of the canonical of the page and the path of the tile, without
 * its query (the mode and the run, which the canonical never keeps) nor its fragment.
 *
 * @param {string} current the canonical of the page, absolute (`https://demoscopio.es/es/circunscripciones`)
 * @param {string} href the `data-href` of the radio of the tile (`/es/circunscripciones/cadiz?mode=forecast`)
 * @returns {string} the canonical of the tile (`https://demoscopio.es/es/circunscripciones/cadiz`)
 */
export function canonicalFor(current, href) {
  return new URL(current).origin + href.split(/[?#]/)[0];
}
