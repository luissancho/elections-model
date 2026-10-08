// Page state (scope, mode, run) kept in the query string.

export const DEFAULT_MODE = 'forecast';
const MODES = ['nowcast', 'forecast'];

/** Read and validate the state from the current URL. */
export function readState() {
  const params = new URLSearchParams(location.search);
  const mode = params.get('mode');
  const run = params.get('run');
  const scope = params.get('scope');
  return {
    scope: scope && /^[0-9a-z_-]+$/.test(scope) ? scope : 'es',
    mode: MODES.includes(mode) ? mode : DEFAULT_MODE,
    run: run && /^\d{8}-\d{6}$/.test(run) ? run : null,
  };
}

/** Merge `partial` into the state, update the URL and notify listeners. */
export function writeState(partial) {
  const state = {...readState(), ...partial};
  const params = new URLSearchParams(location.search);
  params.set('scope', state.scope);
  params.set('mode', state.mode);
  if (state.run) {
    params.set('run', state.run);
  } else {
    params.delete('run');
  }
  history.replaceState(null, '', `${location.pathname}?${params.toString()}`);
  document.dispatchEvent(new CustomEvent('statechange', {detail: state}));
  return state;
}

export function onStateChange(handler) {
  document.addEventListener('statechange', handler);
}
