// Helpers shared by the page modules: embedded data, controls form and error display.

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
 * Submit the controls form whenever the scope or the mode changes. A run id belongs to one scope, so a
 * scope change drops the pinned `run`; a mode change keeps it.
 */
export function wireControls() {
  const form = document.getElementById('controls');
  if (!form) {
    return;
  }
  form.classList.add('js');
  form.addEventListener('change', (event) => {
    const target = event.target;
    if (target.matches('select[name=scope]')) {
      const run = form.querySelector('input[name=run]');
      if (run) {
        run.disabled = true;
      }
    } else if (!target.matches('input[name=mode]')) {
      return;
    }
    form.requestSubmit();
  });
}

/** `{name: value}` for every name, from a catalogue method. */
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
 * Run `fn` once ECharts is available: now if it is loaded, otherwise on the window `load` event (the
 * vendor script is `defer`red). Exceptions go to the console and the status line.
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
