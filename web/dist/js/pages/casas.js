// Houses page and pollster profile: the "Cargar más" of the ranking, the house-effects heat map, the rating path
// (every election the house polled, up to today) and the bullseye, drawn from the data embedded by the server. The
// bullseye of a profile has a scope and a years filter (no hash: the profile has no other state), which redraws it
// and its table.
import {THEME} from '../charts/base.js';
import {renderHeatmap} from '../charts/heatmap.js';
import {renderRatingPath} from '../charts/path.js';
import {KIND_GENERAL, KIND_REGIONAL, REGIONAL_COLOR, wireBullseye} from './bullseye-wire.js';
import {markSegments, mountWhenReady, readInitial, tryDraw, wireControls, wireMore} from './common.js';

/**
 * Wire the switch between the two views of a profile card, each in its own wrap (the server renders both, the
 * second hidden): the deviation of the last poll from the result or from the consensus, and the house effects
 * from the result or from the consensus of the cycle.
 *
 * @param {string} fieldsetId id of the `.seg` fieldset of radios
 * @param {string} inputName name of its radios
 * @param {Object<string, string>} wrapIds id of the wrap of each view, by radio value
 */
function wireViewSwitch(fieldsetId, inputName, wrapIds) {
  const fieldset = document.getElementById(fieldsetId);
  const wraps = Object.fromEntries(Object.entries(wrapIds).map(([view, id]) => [view, document.getElementById(id)]));
  if (!fieldset || Object.values(wraps).some((wrap) => !wrap)) {
    return;
  }
  fieldset.hidden = false;
  markSegments(fieldset);
  fieldset.addEventListener('change', (event) => {
    if (event.target.matches(`input[name=${inputName}]`) && wraps[event.target.value]) {
      for (const [view, wrap] of Object.entries(wraps)) {
        wrap.hidden = view !== event.target.value;
      }
      markSegments(fieldset);
    }
  });
}

/** Wire the "Cargar más" of the ranking and draw the charts whose container is present and has data. */
function paint() {
  wireMore('ranking-more', 'ranking-table');
  wireViewSwitch('last-view', 'lview', {resultado: 'profile-last-wrap', consenso: 'profile-last-consensus-wrap'});
  wireViewSwitch('effects-view', 'eview', {resultado: 'profile-effects-wrap', ciclo: 'profile-effects-cycle-wrap'});
  const {grid = null, path = [], today = null, bullseye = [], blocks = []} = readInitial();

  const heatmap = document.getElementById('effects-heatmap');
  if (heatmap && grid && grid.rows.length) {
    tryDraw(() => renderHeatmap(heatmap, grid), 'the house effects');
  }

  const pathEl = document.getElementById('rating-path');
  if (pathEl && path.length) {
    const colors = {[KIND_GENERAL]: THEME.accentColor, [KIND_REGIONAL]: REGIONAL_COLOR};
    tryDraw(() => renderRatingPath(pathEl, path, {today, colors: {general: colors[KIND_GENERAL], regional: colors[KIND_REGIONAL]}}),
      'the rating path');
  }

  wireBullseye(bullseye, blocks);
}

wireControls();
mountWhenReady(paint);
