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
 * Submit the page controls on change: the scope filter of the global pages (`#scope-filter`). The event pages
 * have no control of their own in the event bar (the mode comes from the server).
 */
export function wireControls() {
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

/**
 * Mirror the state of the checkboxes of a group of pills (`.pill`) on the class `is-on` of their labels, for the
 * browsers without `:has()` (the CSS colours a pill with either).
 *
 * @param {Element|null} root element holding the pills (a `fieldset.pills`)
 */
export function markPills(root) {
  if (!root) {
    return;
  }
  for (const box of root.querySelectorAll('.pill input')) {
    box.closest('.pill').classList.toggle('is-on', box.checked);
  }
}

/**
 * Mirror the checked radio of every segmented control (`.seg`) in `root` on the class `is-on` of its label, for the
 * browsers without `:has()` (the CSS paints a segment with either).
 *
 * @param {Document|Element} root where to look for segments
 */
export function markSegments(root = document) {
  for (const input of root.querySelectorAll('.seg input[type=radio]')) {
    input.closest('label').classList.toggle('is-on', input.checked);
  }
}

// Every page imports this module: keep the segments marked from the start and after each change.
markSegments();
document.addEventListener('change', (event) => {
  if (event.target.closest && event.target.closest('.seg')) {
    markSegments();
  }
});
window.addEventListener('pageshow', () => markSegments());

/**
 * "Ver más" / "Cargar más" of a table: the server renders every row, hides those beyond the first ones and
 * renders the button hidden (it only works with JavaScript); show the button and, on each click, reveal the next
 * batch of hidden rows (`data-batch` of the button; all of them without it), hiding the button when none is left.
 *
 * @param {string} buttonId id of the button
 * @param {string} tableId id of the table
 */
export function wireMore(buttonId, tableId) {
  const button = document.getElementById(buttonId);
  const table = document.getElementById(tableId);
  if (!button || !table) {
    return;
  }
  const batch = Number(button.dataset.batch) || Infinity;
  const hidden = () => [...table.querySelectorAll('tbody tr[hidden]')];
  button.hidden = hidden().length === 0;
  button.addEventListener('click', () => {
    for (const row of hidden().slice(0, batch)) {
      row.hidden = false;
    }
    button.hidden = hidden().length === 0;
  });
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
