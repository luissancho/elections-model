// Header, footer, freeze banner and error rendering shared by all pages.
import {writeState} from './state.js';
import {fmtDate} from './format.js';

function el(tag, text = null, attrs = {}) {
  const node = document.createElement(tag);
  if (text !== null) {
    node.textContent = text;
  }
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

/**
 * Render title, navigation, scope selector and mode switch into `root`, once per page load.
 *
 * Later state changes go through `updateHeader`, which only updates values: rebuilding the header inside
 * the controls' own `change` handlers would drop the keyboard focus.
 */
export function renderHeader(root, {pages, active, scopes, state}) {
  root.replaceChildren();
  const title = el('h1');
  title.append(el('a', 'Pronóstico electoral', {href: '/'}));
  root.append(title);

  const nav = el('nav', null, {'aria-label': 'Principal'});
  for (const page of pages) {
    const link = el('a', page.label, {href: page.href, 'data-href': page.href});
    if (page.href === active) {
      link.setAttribute('aria-current', 'page');
    }
    nav.append(link);
  }
  root.append(nav);

  const controls = el('div', null, {class: 'controls'});
  const label = el('label', 'Ámbito ');
  const select = el('select', null, {id: 'scope'});
  select.addEventListener('change', () => writeState({scope: select.value}));
  label.append(select);
  controls.append(label);

  const fieldset = el('fieldset', null, {id: 'mode'});
  fieldset.append(el('legend', 'Modo'));
  for (const [value, text] of [['nowcast', 'Hoy'], ['forecast', 'Elección']]) {
    const radioLabel = el('label');
    const radio = el('input', null, {type: 'radio', name: 'mode', value});
    radio.addEventListener('change', () => {
      if (radio.checked) {
        writeState({mode: value});
      }
    });
    radioLabel.append(radio, ` ${text}`);
    fieldset.append(radioLabel);
  }
  controls.append(fieldset);
  root.append(controls);
  updateHeader(root, {scopes, state});
}

/**
 * Sync the header built by `renderHeader` with `state`: selected scope, checked mode and the query string
 * of the navigation links. With `scopes`, the scope options are (re)built first (the simulable ones).
 * Nodes are updated in place, so the focused control keeps the focus.
 */
export function updateHeader(root, {scopes = null, state}) {
  const select = root.querySelector('#scope');
  if (scopes) {
    select.replaceChildren(...scopes
      .filter((scope) => scope.simulable)
      .map((row) => el('option', row.name, {value: row.code})));
  }
  select.value = state.scope;
  for (const radio of root.querySelectorAll('input[name=mode]')) {
    radio.checked = radio.value === state.mode;
  }
  for (const link of root.querySelectorAll('nav a[data-href]')) {
    link.setAttribute('href', link.getAttribute('data-href') + location.search);
  }
}

/** Render the run line, the attribution lines and the manifest link into `root`. */
export function renderFooter(root, {meta, manifest}) {
  root.replaceChildren();
  const data = meta.data;
  const dirty = data.dirty ? ' (sucio)' : '';
  root.append(el(
    'p',
    `Run ${data.run_id} · Estimación a ${fmtDate(data.as_of)} · último sondeo ${fmtDate(data.date_last)}` +
      ` · commit ${data.commit}${dirty}`,
  ));
  const attribution = manifest.data.attribution || {};
  for (const key of ['polls', 'results', 'model']) {
    if (attribution[key]) {
      root.append(el('p', attribution[key]));
    }
  }
  const links = el('p');
  links.append(el('a', 'Manifiesto de datos', {href: '/api/v1/manifest'}));
  root.append(links);
}

/** Show the freeze banner with its message when the freeze is active, hide it otherwise. */
export function renderFreeze(root, manifest) {
  const freeze = manifest.data.freeze || {};
  if (freeze.active) {
    root.textContent = freeze.message || '';
    root.hidden = false;
  } else {
    root.textContent = '';
    root.hidden = true;
  }
}

/** Show a Spanish message for an ApiError (or any error) in `root`. */
export function renderError(root, error) {
  let message;
  if (error.status === 503) {
    message = 'Todavía no hay ningún pronóstico publicado.';
  } else if (error.status === 404) {
    message = 'No hay datos para esta selección.';
  } else if (error.status === 0) {
    message = error.message;
  } else {
    message = `Error ${error.status}: ${error.message}`;
  }
  root.textContent = message;
  root.hidden = false;
  root.classList.add('error');
}
