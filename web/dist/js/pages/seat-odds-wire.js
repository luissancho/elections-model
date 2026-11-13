// Wiring of «Escaños que bailan» on the event page (`#seat-odds`, the macro `card` of `_seat_odds.html`). The server
// draws everything and radios with `:has()` do the rest without JavaScript (the view, the two filters of the map and
// the block of the district checked). The script adds what CSS cannot: the image of the card is named after what is on
// view and the line of its filters says them (`seat-odds.js`), the dropdown of the phones checks a tile, a tap on a
// tile scrolls its block into view when the card is stacked and the block is out of the window, and the image button
// downloads the card (`card-image.js`). Every element it looks for may be missing; a page without the card is left
// as it is.
import {wireCardImages} from '../card-image.js';
import {filterLine, imageFile, outOfView} from '../seat-odds.js';
import {showError, tryDraw} from './common.js';

/** The radio groups of the card: the view, the party and the state of the map, and the district. */
const RADIOS = {view: 'sv', party: 'mp', state: 'ms', district: 'd'};

/** Width of the window from which the map and the block of the district sit side by side (`site.css`). */
const SIDE_BY_SIDE = 1200;

/**
 * The radio checked of a group of the card.
 *
 * @param {HTMLElement} card the card
 * @param {string} name the `name` of the radios
 * @returns {?HTMLInputElement} the radio checked, `null` when none is
 */
function checkedRadio(card, name) {
  return card.querySelector(`input[type=radio][name="${name}"]:checked`);
}

/**
 * Name the image of the card after what is on view and write the line of the filters. In the view of the map, the
 * image is named after the district, the party and the state checked, and the line says the party and the state
 * (hidden on «Todos» and «Todos»); in the view of the columns, which the filters do not touch, the image is named
 * after the view (its `data-slug`) and the line is hidden.
 *
 * @param {HTMLElement} card the card, with its `data-file-base`
 */
function syncImage(card) {
  const view = checkedRadio(card, RADIOS.view);
  const columns = Boolean(view && view.dataset.slug);
  const party = checkedRadio(card, RADIOS.party);
  const state = checkedRadio(card, RADIOS.state);
  const district = checkedRadio(card, RADIOS.district);
  const line = document.getElementById('map-filter-line');
  if (line) {
    line.textContent = columns ? '' : filterLine(party ? party.dataset.label : '', state ? state.dataset.label : '');
    line.hidden = !line.textContent;
  }
  const button = card.querySelector('button.card-img-btn');
  if (button && card.dataset.fileBase) {
    const parts = columns ? [view.dataset.slug] : [
      district ? district.value : '', party ? party.dataset.slug : '', state ? state.dataset.slug : ''];
    button.dataset.file = imageFile(card.dataset.fileBase, parts);
  }
}

/**
 * Keep the image of the card and the line of its filters in step (`syncImage`): now, on every change of a radio of
 * the card, and when the page comes back from the back/forward cache with its radios as they were left.
 *
 * @param {HTMLElement} card the card
 */
function wireImageName(card) {
  const names = Object.values(RADIOS);
  syncImage(card);
  card.addEventListener('change', (event) => {
    if (names.includes(event.target.name)) {
      syncImage(card);
    }
  });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      syncImage(card);
    }
  });
}

/**
 * Reveal the dropdown of districts of the phones (`#district-select`; the stylesheet shows it only under 720 px) and
 * keep it in step with the tiles: choosing a district checks its tile and dispatches its `change`, so that the block,
 * the image and everything else follow as on a tap; a tile checked sets the dropdown, which dispatches nothing back.
 * It never navigates.
 *
 * @param {HTMLElement} card the card
 */
function wireDropdown(card) {
  const select = document.getElementById('district-select');
  if (!select) {
    return;
  }
  const follow = () => {
    const radio = checkedRadio(card, RADIOS.district);
    if (radio) {
      select.value = radio.value;
    }
  };
  follow();
  select.hidden = false;
  select.addEventListener('change', () => {
    const radio = document.getElementById(`d-${select.value}`);
    if (!radio || radio.name !== RADIOS.district || radio.checked) {
      return;
    }
    radio.checked = true;
    radio.dispatchEvent(new Event('change', {bubbles: true}));
  });
  card.addEventListener('change', (event) => {
    if (event.target.name === RADIOS.district && event.target.checked) {
      select.value = event.target.value;
    }
  });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      follow();
    }
  });
}

/**
 * Scroll the block of a district into view when the card is stacked (a window narrower than `SIDE_BY_SIDE`) and none
 * of the block shows (`outOfView`): as little as needed (`nearest`), smoothly unless the reader asks for less motion.
 *
 * @param {HTMLInputElement} radio the radio of the tile, checked
 */
function revealBlock(radio) {
  if (!radio.checked || window.innerWidth >= SIDE_BY_SIDE) {
    return;
  }
  const block = document.getElementById(`ficha-${radio.value}`);
  if (!block || !outOfView(block.getBoundingClientRect(), window.innerHeight)) {
    return;
  }
  const still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  block.scrollIntoView({block: 'nearest', behavior: still ? 'auto' : 'smooth'});
}

/**
 * After a tap or a click on a tile (a `click` on its label with `detail` above 0: the arrow keys move the radio
 * group with `click`s of `detail` 0, on the radio itself), bring the block of its district into view
 * (`revealBlock`). The label checks its radio, and the stylesheet shows its block, only once its own `click` is over:
 * the block is read on the next frame.
 *
 * @param {HTMLElement} card the card
 */
function wireReveal(card) {
  card.addEventListener('click', (event) => {
    const tile = event.detail > 0 && event.target.closest ? event.target.closest('label.so-tile') : null;
    const radio = tile ? tile.querySelector(`input[name="${RADIOS.district}"]`) : null;
    if (!radio || event.target === radio) {
      return;
    }
    window.requestAnimationFrame(() => tryDraw(() => revealBlock(radio), 'the block of the district'));
  });
}

/**
 * Wire the card «Escaños que bailan» of the event page (`#seat-odds`; nothing without it), each part on its own so
 * that a failure does not stop the others: the name of its image and the line of its filters, the dropdown of the
 * phones, the scroll to the block of a district tapped and the download of the card as an image.
 */
export function wireSeatOdds() {
  const card = document.getElementById('seat-odds');
  if (!card) {
    return;
  }
  tryDraw(() => wireImageName(card), 'the name of the image of the seats that dance');
  tryDraw(() => wireDropdown(card), 'the dropdown of districts');
  tryDraw(() => wireReveal(card), 'the scroll to the block of a district');
  tryDraw(() => wireCardImages(card, showError), 'the download of the seats that dance');
}
