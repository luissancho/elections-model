// DOM builders of the "Todas las encuestas" table of Promedios: header, body and footer, built with
// `createElement`/`textContent` (never HTML with data).
import {fmtDate, fmtInt, fmtNum} from '../format.js';
import {leaderOf, pollsCount, shortDate} from '../polls-table.js';

const FIXED = [['pollster', 'Encuestadora'], ['date', 'Fecha'], ['sample_size', 'Muestra']];

/**
 * Create an element.
 *
 * @param {string} tag tag name
 * @param {string} [text] text content
 * @param {string} [className] class attribute
 * @returns {HTMLElement} the element
 */
function el(tag, text = '', className = '') {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) {
    node.className = className;
  }
  return node;
}

/**
 * Value cells of a row: one per party, the leader shaded in its colour.
 *
 * @param {Object<string, (number|null)>} values value by party name
 * @param {string[]} parties columns
 * @param {Catalog} catalog party catalogue
 * @returns {HTMLElement[]} the `td` cells
 */
function valueCells(values, parties, catalog) {
  const leaders = leaderOf(Object.fromEntries(parties.map((name) => [name, values[name]])));
  return parties.map((name) => {
    const missing = typeof values[name] !== 'number';
    const td = el('td', missing ? '—' : fmtNum(values[name], 1), missing ? 'p na' : 'p');
    if (leaders.includes(name)) {
      td.className = 'p mx';
      td.style.background = `${catalog.color(name)}24`;
    }
    return td;
  });
}

/**
 * Rebuild the header row.
 *
 * @param {HTMLTableSectionElement} thead header section
 * @param {string[]} parties columns shown
 * @param {Catalog} catalog party catalogue
 * @param {{key: string, asc: boolean}} sort active sort, marked with `aria-sort`
 */
export function renderHead(thead, parties, catalog, sort) {
  const tr = el('tr');
  const add = (th, key) => {
    th.dataset.key = key;
    th.scope = 'col';
    const button = el('button', '', 'sort');
    button.type = 'button';
    button.append(...th.childNodes);
    th.appendChild(button);
    if (key === sort.key) {
      th.setAttribute('aria-sort', sort.asc ? 'ascending' : 'descending');
    }
    tr.appendChild(th);
  };
  FIXED.forEach(([key, label], index) => add(el('th', label, index === 0 ? 'f l' : ''), key));
  for (const name of parties) {
    const th = el('th', '', 'p');
    th.title = catalog.fullname(name);
    const dot = el('i', '', 'pd');
    dot.style.background = catalog.color(name);
    th.append(dot, name);
    add(th, name);
  }
  thead.replaceChildren(tr);
}

/**
 * Rebuild the body with one page of polls.
 *
 * @param {HTMLTableSectionElement} tbody body section
 * @param {Array} polls polls of the page
 * @param {string[]} parties columns shown
 * @param {Catalog} catalog party catalogue
 */
export function renderBody(tbody, polls, parties, catalog) {
  if (!polls.length) {
    const td = el('td', 'Ningún sondeo con estos filtros.', 'empty');
    td.colSpan = FIXED.length + parties.length;
    const tr = el('tr');
    tr.appendChild(td);
    tbody.replaceChildren(tr);
    return;
  }
  tbody.replaceChildren(...polls.map((poll) => {
    const first = el('td', '', 'f l');
    first.append(el('b', poll.pollster || '–'), el('small', poll.sponsor || '—'));
    const date = el('td', shortDate(poll.date));
    if (poll.start_date || poll.end_date) {
      date.title = `Trabajo de campo: ${fmtDate(poll.start_date || poll.end_date)} – ${fmtDate(poll.end_date || poll.start_date)}`;
    }
    const tr = el('tr');
    tr.append(first, date, el('td', fmtInt(poll.sample_size)), ...valueCells(poll, parties, catalog));
    return tr;
  }));
}

/**
 * Rebuild the footer with the simple mean of the filtered polls.
 *
 * @param {HTMLTableSectionElement} tfoot footer section
 * @param {number} count polls averaged
 * @param {Object<string, (number|null)>} means mean by party name
 * @param {string[]} parties columns shown
 * @param {Catalog} catalog party catalogue
 */
export function renderFoot(tfoot, count, means, parties, catalog) {
  const first = el('td', '', 'f l');
  first.append(el('b', 'Media simple'), el('small', pollsCount(count)));
  const tr = el('tr');
  tr.append(first, el('td'), el('td'), ...valueCells(means, parties, catalog));
  tfoot.replaceChildren(tr);
}
