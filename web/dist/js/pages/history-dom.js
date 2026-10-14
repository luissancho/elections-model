// DOM writers of the error history (filters, figures, complements and the table of elections), built with
// nodes from the data, never with HTML strings.
import {fmtNum} from '../format.js';

const DASH = '—';

/**
 * Rebuild the options of a `select` and choose a value.
 *
 * @param {HTMLSelectElement} select the control
 * @param {Array<{value: string, label: string, scope: (string|undefined)}>} options the options after the first
 * @param {string} first label of the leading option, whose value is `''`
 * @param {(string|null)} value the wanted value
 * @returns {(string|null)} the value now selected: `null` when the wanted one is not among the options
 */
export function fillSelect(select, options, first, value) {
  const nodes = [{value: '', label: first}, ...options].map((item) => {
    const option = document.createElement('option');
    option.value = item.value;
    option.textContent = item.label;
    if (item.scope) {
      option.dataset.scope = item.scope;
    }
    return option;
  });
  select.replaceChildren(...nodes);
  const found = value !== null && options.some((item) => item.value === value);
  select.value = found ? value : '';
  return found ? value : null;
}

/**
 * Format a mean for the figures: one decimal, or a dash without value.
 *
 * @param {(number|null)} value the mean
 * @param {boolean} signed whether to put a plus sign on positive values
 * @returns {string} the text
 */
function figure(value, signed = false) {
  if (value === null || !Number.isFinite(value)) {
    return DASH;
  }
  return `${signed && value > 0 ? '+' : ''}${fmtNum(value, 1)}`;
}

/**
 * Write the four figures of the selection.
 *
 * @param {{polls: HTMLElement, events: HTMLElement, mean: HTMLElement, bias: HTMLElement, meanLabel: HTMLElement}} figures
 *   the `dd` of each figure and the `dt` of the mean
 * @param {{polls: number, events: number, mean: (number|null), bias: (number|null), meanLabel: string}} data the numbers
 */
export function fillFigures(figures, data) {
  figures.polls.textContent = String(data.polls);
  figures.events.textContent = String(data.events);
  figures.mean.textContent = figure(data.mean);
  figures.bias.textContent = figure(data.bias, true);
  figures.meanLabel.textContent = data.meanLabel;
}

/**
 * Write one line per complement under the figures (the list is emptied when there are none).
 *
 * @param {HTMLElement} list the `ul`
 * @param {Array<{label: string, n: number, mean: (number|null)}>} items the complements
 */
export function fillComplements(list, items) {
  list.replaceChildren(...items.map((item) => {
    const li = document.createElement('li');
    li.textContent = `${item.label}: media ${figure(item.mean)} · ${item.n} ${item.n === 1 ? 'sondeo' : 'sondeos'}`;
    return li;
  }));
}

/**
 * Hide the rows of the table of elections outside the filters and write the count.
 *
 * @param {HTMLTableElement} table the `#events-table`, rows with `data-scope`, `data-date` and `data-year`
 * @param {HTMLElement|null} count the `#events-count`
 * @param {Function} passes `(scope, date, year) => boolean`
 */
export function filterEventsTable(table, count, passes) {
  let shown = 0;
  for (const row of table.tBodies[0].rows) {
    const keep = passes(row.dataset.scope, row.dataset.date, Number(row.dataset.year));
    row.hidden = !keep;
    shown += keep ? 1 : 0;
  }
  if (count) {
    count.textContent = `${shown} ${shown === 1 ? 'elección' : 'elecciones'}`;
  }
}
