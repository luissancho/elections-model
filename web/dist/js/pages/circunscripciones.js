// Page «Escaños que bailan» (`/{scope}/circunscripciones`): the server draws everything, and radios with `:has()` do
// the rest without JavaScript (the card of the tile checked, the two filters of the map, the switch of the chart of
// the parties). The script adds what CSS cannot: the address, the title and the canonical follow the tile checked
// (`history.replaceState`, no reload, no scroll), the dropdown of the phones checks a tile instead of submitting its
// form, the images of the map and of the chart are named after what is on view (`seat-odds.js`) and the line of the
// image of the map says its filters; and it wires the download of each card as an image (`card-image.js`). It runs on
// the three views, so every element it looks for may be missing.
import {wireCardImages} from '../card-image.js';
import {canonicalFor, filterLine, imageFile} from '../seat-odds.js';
import {mountWhenReady, showError, tryDraw} from './common.js';

/**
 * The radio checked of a group.
 *
 * @param {string} name the `name` of the radios
 * @returns {?HTMLInputElement} the radio checked, `null` when none is (or the group is not on the page)
 */
function checkedRadio(name) {
  return document.querySelector(`input[type=radio][name="${name}"]:checked`);
}

/**
 * Point the image button of a card at the file named after what is on view.
 *
 * @param {?HTMLElement} card the card, with its `data-file-base`
 * @param {string[]} parts slugs of what is on view (empty ones skip, `imageFile`)
 */
function nameImage(card, parts) {
  const button = card && card.querySelector('button.card-img-btn');
  if (button && card.dataset.fileBase) {
    button.dataset.file = imageFile(card.dataset.fileBase, parts);
  }
}

/**
 * Follow the two filters of the map (`mp`, `ms`): the line of its image says them (hidden on «Todos» and «Todos») and
 * its image is named after them, from the `data-label` and `data-slug` of the radios checked.
 */
function syncMap() {
  const card = document.getElementById('mapa');
  if (!card) {
    return;
  }
  const party = checkedRadio('mp');
  const state = checkedRadio('ms');
  const line = document.getElementById('map-filter-line');
  if (line) {
    line.textContent = filterLine(party ? party.dataset.label : '', state ? state.dataset.label : '');
    line.hidden = !line.textContent;
  }
  nameImage(card, [party ? party.dataset.slug : '', state ? state.dataset.slug : '']);
}

/** Follow the switch of the chart of the parties (`pf`): its image is named after the chart on view. */
function syncChart() {
  const form = checkedRadio('pf');
  nameImage(document.getElementById('grafico'), [form ? form.dataset.slug : '']);
}

/**
 * Run `sync` now and whenever a radio of one of the `names` groups changes.
 *
 * @param {string[]} names the `name` of the radios
 * @param {Function} sync what follows them
 */
function follow(names, sync) {
  sync();
  document.addEventListener('change', (event) => {
    if (names.includes(event.target.name)) {
      sync();
    }
  });
}

/**
 * Point the canonical of the page (and its `og:url`) at the address of a tile, as the server writes it for that
 * address (`canonicalFor`); a page without them is left as it is.
 *
 * @param {string} href the `data-href` of the radio of the tile
 */
function followCanonical(href) {
  const link = document.querySelector('link[rel="canonical"]');
  if (link && link.href) {
    link.href = canonicalFor(link.href, href);
  }
  const og = document.querySelector('meta[property="og:url"]');
  if (og && og.content) {
    og.content = canonicalFor(og.content, href);
  }
}

/**
 * The address, the title and the canonical of the page follow the tile checked (`d`), without reloading nor
 * scrolling: the `data-href` and `data-title` of its radio; the dropdown of the phones shows the same district. The
 * title, the dropdown and the canonical go first, so they follow even when the browser refuses to replace the address.
 */
function wireDistricts() {
  document.addEventListener('change', (event) => {
    const radio = event.target;
    if (radio.name !== 'd' || !radio.checked) {
      return;
    }
    if (radio.dataset.title) {
      document.title = radio.dataset.title;
    }
    const select = document.getElementById('district-select');
    if (select) {
      select.value = radio.value;
    }
    if (radio.dataset.href) {
      followCanonical(radio.dataset.href);
      try {
        history.replaceState(null, '', radio.dataset.href);
      } catch (error) {
        // Safari refuses history calls past ~100 in 30 s (an arrow key held over the tiles): keep the old address
      }
    }
  });
}

/**
 * The dropdown of the phones (`#district-pick`) no longer submits its form (`.js` hides its «Ver» button): it checks
 * the tile of the district chosen and dispatches its `change`, so that the address and the card follow as on a tap.
 * A district without its tile (never expected) still submits the form, as without JavaScript.
 */
function wireDropdown() {
  const form = document.getElementById('district-pick');
  const select = document.getElementById('district-select');
  if (!form || !select) {
    return;
  }
  form.classList.add('js');
  select.addEventListener('change', () => {
    const radio = document.getElementById(`d-${select.value}`);
    if (!radio) {
      form.submit();
      return;
    }
    radio.checked = true;
    radio.dispatchEvent(new Event('change', {bubbles: true}));
  });
}

/** Wire the page, each part on its own so that a failure does not stop the others. */
function wire() {
  tryDraw(wireDistricts, 'the address of the tiles');
  tryDraw(wireDropdown, 'the dropdown of districts');
  tryDraw(() => follow(['mp', 'ms'], syncMap), 'the image of the map');
  tryDraw(() => follow(['pf'], syncChart), 'the image of the chart');
  // a page restored from the back/forward cache keeps the radios as they were left: name the images after them again
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      tryDraw(syncMap, 'the image of the map');
      tryDraw(syncChart, 'the image of the chart');
    }
  });
  tryDraw(() => wireCardImages(document, showError), 'the download of the card');
}

mountWhenReady(wire);
