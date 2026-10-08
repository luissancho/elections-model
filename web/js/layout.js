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

/** Render title, navigation, scope selector and mode switch into `root`. */
export function renderHeader(root, {pages, active, scopes, state}) {
  root.replaceChildren();
  const title = el('h1');
  title.append(el('a', 'Pronóstico electoral', {href: '/'}));
  root.append(title);

  const nav = el('nav', null, {'aria-label': 'Principal'});
  for (const page of pages) {
    const link = el('a', page.label, {href: page.href + location.search});
    if (page.href === active) {
      link.setAttribute('aria-current', 'page');
    }
    nav.append(link);
  }
  root.append(nav);

  const controls = el('div', null, {class: 'controls'});
  const label = el('label', 'Ámbito ');
  const select = el('select', null, {id: 'scope'});
  for (const row of scopes.filter((scope) => scope.simulable)) {
    const option = el('option', row.name, {value: row.code});
    option.selected = row.code === state.scope;
    select.append(option);
  }
  select.addEventListener('change', () => writeState({scope: select.value}));
  label.append(select);
  controls.append(label);

  const fieldset = el('fieldset', null, {id: 'mode'});
  fieldset.append(el('legend', 'Modo'));
  for (const [value, text] of [['nowcast', 'Hoy'], ['forecast', 'Elección']]) {
    const radioLabel = el('label');
    const radio = el('input', null, {type: 'radio', name: 'mode', value});
    radio.checked = state.mode === value;
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
