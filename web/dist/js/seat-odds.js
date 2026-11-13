// Pure helpers of the page «Escaños que bailan» (`/{scope}/circunscripciones`): the name of the image of a card and
// the line that says the filters of the map, after what is on view. No imports, so node tests them as they are.

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
