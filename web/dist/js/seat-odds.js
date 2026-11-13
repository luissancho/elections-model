// Pure helpers of «Escaños que bailan», the card of the event page (`seat-odds-wire.js`): the name of its image and
// the line that says the filters of the map, after what is on view, and whether a box is out of the free band of the
// window, between the bars stuck to it. No imports, so node tests them as they are.

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
 * Whether a box is out of view: none of it shows in the free band of a window of that height, between what the bars
 * stuck to its top and to its foot cover (an edge on the border of the band shows nothing either).
 *
 * @param {{top: number, bottom: number}} rect the box, relative to the window (`getBoundingClientRect`)
 * @param {number} height the height of the window (`innerHeight`)
 * @param {number} [top=0] the pixels the bars stuck to the top of the window cover (`coveredBand`)
 * @param {number} [foot=0] the pixels the bars stuck to its foot cover
 * @returns {boolean} true when the box is wholly below or wholly above the free band
 */
export function outOfView(rect, height, top = 0, foot = 0) {
  return rect.top >= height - foot || rect.bottom <= top;
}

/**
 * What the bars stuck to the window cover of it: from its top down to the lowest bottom of the top bars (the freeze
 * banner and the header), and from its foot up to the highest top of the foot bars (the tab bar of the phones), each
 * within the window.
 *
 * @param {Array<{top: number, bottom: number}>} tops the boxes of the bars stuck to the top, relative to the window
 * @param {Array<{top: number, bottom: number}>} foots the boxes of the bars stuck to the foot
 * @param {number} height the height of the window (`innerHeight`)
 * @returns {{top: number, foot: number}} the pixels covered at the top and at the foot, 0 without bars
 */
export function coveredBand(tops, foots, height) {
  const clamp = (value) => Math.min(Math.max(value, 0), height);
  return {
    top: clamp(Math.max(0, ...tops.map((box) => box.bottom))),
    foot: clamp(Math.max(0, ...foots.map((box) => height - box.top))),
  };
}
