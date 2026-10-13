// Helpers shared by the page modules: embedded data, auto-submitted forms and error display.

/**
 * Parse the page data embedded by the server in `<script type="application/json" id="initial-data">`.
 *
 * @returns {object} the page's initial data
 */
export function readInitial() {
  const node = document.getElementById('initial-data');
  if (!node) {
    throw new Error('Missing #initial-data');
  }
  return JSON.parse(node.textContent);
}

/**
 * Submit the page controls on change: the mode form of the event bar (`#event-controls`, which keeps a pinned
 * `run`) and the scope filter of the global pages (`#scope-filter`). A page has at most one of them.
 */
export function wireControls() {
  wireForm('event-controls');
  wireForm('scope-filter');
}

/**
 * Submit the form `#id` whenever any of its controls changes. A page restored from the back/forward cache
 * resets the form to the values of the page it belongs to.
 *
 * @param {string} id form id
 */
export function wireForm(id) {
  const form = document.getElementById(id);
  if (!form) {
    return;
  }
  form.classList.add('js');
  form.addEventListener('change', () => {
    if (form.requestSubmit) {
      form.requestSubmit();
    } else {
      form.submit();
    }
  });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
      form.reset();
    }
  });
}

/**
 * `{name: value}` for every name, from a catalogue method.
 *
 * @param {string[]} names party names
 * @param {Function} fn maps a name to its value
 * @returns {object} values keyed by name
 */
export function mapNames(names, fn) {
  return Object.fromEntries(names.map((name) => [name, fn(name)]));
}

/** Show `message` as an error in the status line. */
export function showError(message) {
  const status = document.getElementById('status');
  if (!status) {
    return;
  }
  status.textContent = message;
  status.classList.add('error');
  status.hidden = false;
}

/**
 * Run one drawing call, so that a failing chart does not stop the ones drawn after it.
 *
 * @param {Function} fn draws one part of the page
 * @param {string} label part name, for the console
 * @returns {boolean} false if `fn` threw (the error goes to the console)
 */
export function tryDraw(fn, label) {
  try {
    fn();
    return true;
  } catch (error) {
    console.error(`Could not draw ${label}:`, error);
    return false;
  }
}

/**
 * Run `fn` once ECharts is available: now if it is loaded, otherwise on the window `load` event, which
 * can only report the missing library (`charts/base.js` captures `window.echarts` when it is evaluated). Exceptions go to the console and the status line.
 *
 * @param {Function} fn draws the page
 * @param {string} message text shown if `fn` throws
 */
export function mountWhenReady(fn, message = 'No se pudieron dibujar los gráficos.') {
  const run = () => {
    try {
      fn();
    } catch (error) {
      console.error(error);
      showError(message);
    }
  };
  if (window.echarts) {
    run();
  } else {
    window.addEventListener('load', run);
  }
}
