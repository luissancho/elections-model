// Wiring of the "Todas las encuestas" table (Promedios and the detail of a past election): party and pollster
// dropdowns, month range, the % de voto / Escaños switch, sortable headers and pages of `PAGE_SIZE` rows, all
// computed in the browser from the polls of the page (`polls-table.js`) and drawn with `polls-dom.js`.
import {PAGE_SIZE, filterPolls, meanOf, pageLabel, pageOf, pollsCount, sortPolls} from '../polls-table.js';
import {markSegments} from './common.js';
import {renderBody, renderFoot, renderHead} from './polls-dom.js';

/**
 * Wire the controls of a dropdown of checkboxes (`details.dd`): "Seleccionar todos" and "Quitar todos" (which
 * leaves the first one checked), at least one box always checked, and the counter of its summary.
 *
 * @param {HTMLElement} dd the `details` element
 * @param {Function} onChange called after any change
 * @returns {function(): HTMLInputElement[]} the boxes that are checked now
 */
export function wireDropdown(dd, onChange) {
  const boxes = [...dd.querySelectorAll('input[type="checkbox"]')];
  const sync = () => {
    dd.querySelector('[data-count]').textContent = String(boxes.filter((box) => box.checked).length);
    boxes.forEach((box) => box.closest('.chip').classList.toggle('is-on', box.checked));
  };
  const set = (checked) => {
    boxes.forEach((box) => { box.checked = checked; });
    if (!checked && boxes.length) {
      boxes[0].checked = true;
    }
    sync();
    onChange();
  };
  dd.querySelector('[data-all]').addEventListener('click', () => set(true));
  dd.querySelector('[data-none]').addEventListener('click', () => set(false));
  dd.addEventListener('change', (event) => {
    if (!boxes.some((box) => box.checked)) {
      event.target.checked = true;
    }
    sync();
    onChange();
  });
  sync();
  return () => boxes.filter((box) => box.checked);
}

/** Sort keys of the fixed columns of the polls table; any other key is a party. */
const FIXED_KEYS = ['pollster', 'date', 'sample_size'];

/**
 * Wire the "Todas las encuestas" table: party and pollster dropdowns, month range, sortable headers and pages of
 * `PAGE_SIZE` rows, all computed in the browser from `polls`. The server renders the 15 most recent rows and
 * hides the controls, which this function shows; without the table in the page it does nothing.
 *
 * @param {object[]} polls polls of the cycle (`chart_polls`: one value per party, `seats` when published)
 * @param {Catalog} catalog party catalogue
 */
export function wirePollsTable(polls, catalog) {
  const controls = document.getElementById('polls-controls');
  const table = document.getElementById('polls-table');
  const pag = document.getElementById('polls-pag');
  if (!controls || !table || !pag) {
    return;
  }
  const from = document.getElementById('polls-from');
  const to = document.getElementById('polls-to');
  const state = {sort: {key: 'date', asc: false}, page: 0, mode: 'pct'};
  const modeSwitch = document.getElementById('polls-mode');
  const [prev, next] = pag.querySelectorAll('button');
  const pick = (checked, total) => (checked.length === total ? null : checked.map((box) => box.value));

  const render = (focusKey = null) => {
    const parties = checkedParties().map((box) => box.value);
    if (!FIXED_KEYS.includes(state.sort.key) && !parties.includes(state.sort.key)) {
      state.sort = {key: 'date', asc: false};
    }
    const pollsters = pick(checkedPollsters(), pollsterTotal);
    const rows = sortPolls(filterPolls(polls, {
      pollsters,
      from: from.selectedIndex <= 0 ? null : from.value,
      to: to.selectedIndex === to.options.length - 1 ? null : to.value,
      mode: state.mode,
    }), state.sort.key, state.sort.asc, state.mode);
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const shown = pageOf(rows, state.page);
    renderHead(table.tHead, parties, catalog, state.sort);
    renderBody(table.tBodies[0], shown, parties, catalog, state.mode);
    renderFoot(table.tFoot, rows.length, meanOf(rows, parties, state.mode), parties, catalog, state.mode);
    document.getElementById('polls-count').textContent = pollsCount(rows.length) + (state.mode === 'esc' ? ' con escaños' : '');
    const recent = state.sort.key === 'date' && !state.sort.asc;
    prev.textContent = recent ? '‹ Más recientes' : '‹ Anteriores';
    next.textContent = recent ? 'Más antiguos ›' : 'Siguientes ›';
    prev.disabled = state.page === 0;
    next.disabled = state.page >= pages - 1;
    document.getElementById('polls-range').textContent = pageLabel(state.page, shown.length, rows.length);
    pag.hidden = pages <= 1;
    if (focusKey) {
      const button = table.querySelector(`th[data-key="${CSS.escape(focusKey)}"] .sort`);
      if (button) {
        button.focus();
      }
    }
  };
  const refilter = () => { state.page = 0; render(); };
  const checkedParties = wireDropdown(document.getElementById('polls-parties'), refilter);
  const checkedPollsters = wireDropdown(document.getElementById('polls-pollsters'), refilter);
  const pollsterTotal = document.querySelectorAll('#polls-pollsters input[type="checkbox"]').length;

  from.addEventListener('change', refilter);
  to.addEventListener('change', refilter);
  if (modeSwitch) {
    modeSwitch.addEventListener('change', (event) => {
      if (event.target.matches('input[name=pmode]')) {
        state.mode = event.target.value === 'esc' ? 'esc' : 'pct';
        markSegments(modeSwitch);
        refilter();
      }
    });
  }
  table.tHead.addEventListener('click', (event) => {
    const th = event.target.closest('th[data-key]');
    if (!th) {
      return;
    }
    const key = th.dataset.key;
    state.sort = key === state.sort.key ? {key, asc: !state.sort.asc} : {key, asc: key === 'pollster'};
    state.page = 0;
    render(key);
  });
  pag.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-page]');
    if (button && !button.disabled) {
      state.page += Number(button.dataset.page);
      render();
    }
  });
  document.addEventListener('click', (event) => {
    for (const dd of controls.querySelectorAll('details.dd[open]')) {
      if (!dd.contains(event.target)) {
        dd.open = false;
      }
    }
  });
  controls.hidden = false;
  render();
}
