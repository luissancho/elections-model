// Houses page and pollster profile: the "Cargar más" of the ranking, the house-effects heat map, the rating path
// (every election the house polled, up to today) and the bullseye, drawn from the data embedded by the server. The
// bullseye of a profile has a scope and a years filter (no hash: the profile has no other state), which redraws it
// and its table.
import {fmtDate, fmtNum, shadeAround} from '../format.js';
import {fadeByDate, filterRows} from '../errors.js';
import {THEME} from '../charts/base.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderHeatmap} from '../charts/heatmap.js';
import {renderRatingPath} from '../charts/path.js';
import {rankingVisibility} from '../ranking.js';
import {markSegments, mountWhenReady, readInitial, tryDraw, wireControls, wireMore} from './common.js';

// The bullseye and the rating path of a profile: general elections in the accent, regional ones in a contrasting
// tone; a point of the bullseye fades with the age of its election down to this opacity.
const NATIONAL_SCOPE = 'es';
const KIND_GENERAL = 'Generales';
const KIND_REGIONAL = 'Autonómicas';
const REGIONAL_COLOR = '#c0672e';
const OLDEST_ALPHA = 0.3;
// The double gradient of the block errors of the bullseye table, the same the server paints (`casa.html`): red the
// more the poll fell short of the block, green the more it overshot it, fullest at this many points.
const SHORT_COLOR = '#d62728';
const OVER_COLOR = '#2ca02c';
const ERROR_SPAN = 8;
// Leading columns of the bullseye table before the block columns (election, date, days).
const BULLSEYE_FIXED_COLUMNS = 3;

/**
 * Rebuild the body of the bullseye table with the rows shown. The block columns come from the table header, in its
 * order (the server puts the left block first, which may differ from the order of the axes of the bullseye), and
 * their cells carry the same double gradient the server paints.
 *
 * @param {HTMLTableElement|null} table the `#bullseye-table`
 * @param {object[]} rows bullseye rows (`name`, `event_date`, `days` and the error of each block)
 */
function fillBullseyeTable(table, rows) {
  if (!table) {
    return;
  }
  const blocks = [...table.tHead.rows[0].cells].slice(BULLSEYE_FIXED_COLUMNS).map((th) => th.textContent.trim());
  const cell = (text) => {
    const td = document.createElement('td');
    td.textContent = text;
    return td;
  };
  const shaded = (value) => {
    const td = cell(fmtNum(value, 1));
    td.className = 'cell-shade';
    td.style.backgroundColor = shadeAround(value, 0, ERROR_SPAN, SHORT_COLOR, OVER_COLOR);
    return td;
  };
  table.tBodies[0].replaceChildren(...rows.map((row) => {
    const tr = document.createElement('tr');
    tr.append(cell(row.name || row.event), cell(fmtDate(row.event_date)), cell(fmtNum(row.days)),
      ...blocks.map((block) => shaded(row[block])));
    return tr;
  }));
}

/**
 * Bullseye of a profile with its filters: draw it and its table from the rows that pass the chosen scope and
 * years; the form only works with JavaScript, so the server renders it hidden and this shows it.
 *
 * @param {object[]} bullseye the profile's bullseye rows (`scope`, `year` and the block errors)
 * @param {string[]} blocks the two blocks of the axes
 */
function wireBullseye(bullseye, blocks) {
  const target = document.getElementById('bullseye-chart');
  if (!target || !bullseye.length || blocks.length < 2) {
    return;
  }
  const form = document.getElementById('bullseye-filter');
  const table = document.getElementById('bullseye-table');
  const count = document.getElementById('bullseye-count');
  // The fading is fixed over every election of the house, so a filter does not change the tone of a point.
  const alphas = new Map(fadeByDate(bullseye, {min: OLDEST_ALPHA}).map((alpha, i) => [bullseye[i], alpha]));
  const options = {
    blocks,
    groupOf: (row) => (row.scope === NATIONAL_SCOPE ? KIND_GENERAL : KIND_REGIONAL),
    colors: {[KIND_GENERAL]: THEME.accentColor, [KIND_REGIONAL]: REGIONAL_COLOR},
    legend: true,
    labelOf: (row) => `${row.event}-${row.scope}`,
    alphaOf: (row) => alphas.get(row) ?? 1,
    onTop: (group) => group === KIND_GENERAL,  // the general elections over the regional ones
  };
  const draw = (rows) => {
    tryDraw(() => renderBullseye(target, rows, options), 'the bullseye');
    fillBullseyeTable(table, rows);
    if (count) {
      count.textContent = `${rows.length} ${rows.length === 1 ? 'elección' : 'elecciones'}`;
    }
  };
  if (form) {
    form.addEventListener('submit', (event) => event.preventDefault());
    // The years offered run from the first election of the house to today (the server fills both selects with the
    // same list); an extreme chosen is an open bound.
    const {desde, hasta} = form.elements;
    const first = Number(desde.options[0]?.value);
    const last = Number(hasta.options[hasta.options.length - 1]?.value);
    form.addEventListener('change', () => {
      const {ambito} = form.elements;
      draw(filterRows(bullseye, {
        scope: ambito.value || null,
        from: Number(desde.value) === first ? null : Number(desde.value),
        to: Number(hasta.value) === last ? null : Number(hasta.value),
      }));
    });
    form.hidden = false;
  }
  draw(bullseye);
}

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

/**
 * Wire the "Solo con experiencia en generales" filter of the ranking: on change, show the first batch of the rows
 * with general elections (`data-generals`), mark the others as filtered (so "Cargar más" skips them) and show the
 * button only while rows are left. The control works with JavaScript only, so the server renders it hidden.
 */
function wireGeneralsFilter() {
  const input = document.getElementById('ranking-generals');
  const table = document.getElementById('ranking-table');
  if (!input || !table) {
    return;
  }
  const button = document.getElementById('ranking-more');
  const batch = Number(button?.dataset.batch) || Infinity;
  const rows = [...table.querySelectorAll('tbody tr')];
  const generals = rows.map((row) => Number(row.dataset.generals) || 0);
  input.addEventListener('change', () => {
    const {visible, filtered, more} = rankingVisibility(generals, input.checked, batch);
    rows.forEach((row, i) => {
      row.hidden = !visible[i];
      row.toggleAttribute('data-filtered', filtered[i]);
    });
    if (button) {
      button.hidden = !more;
    }
  });
  document.getElementById('ranking-controls').hidden = false;
}

/** Wire the "Cargar más" of the ranking and draw the charts whose container is present and has data. */
function paint() {
  wireMore('ranking-more', 'ranking-table');
  wireGeneralsFilter();
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
