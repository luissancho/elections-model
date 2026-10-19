// DOM builders of the "Todas las encuestas" table (Promedios and the detail of a past election): header, body and footer, built with
// `createElement`/`textContent` (never HTML with data).
import {fmtDate, fmtInt, fmtNum, gradient} from '../format.js';
import {leaderOf, pollsCount, seatsText, shortDate, valuesOf} from '../polls-table.js';

/** Fixed columns, in order (key and label); any other column is a party. The server renders the same ones. */
export const FIXED_COLUMNS = [['pollster', 'Encuestadora'], ['rating', 'Rating (0-100)'], ['date', 'Fecha'],
  ['sample_size', 'Muestra']];
/** Stops of the rating badge, from 0 to 100: red, grey at 50 and green (`pages.RATING_COLORS` of the server). */
export const RATING_COLORS = ['#d62728', '#7f7f7f', '#2ca02c'];
const RATING_TITLE = 'Rating de la encuestadora, de 0 a 100: cuanto más alto, más pesa su sondeo en el promedio. Rojo por debajo de 50, verde por encima.';

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
 * @param {Object<string, (number|null)>} values value by party name (what sorts and picks the leader)
 * @param {string[]} parties columns
 * @param {Catalog} catalog party catalogue
 * @param {Object<string, (string|null)>} [texts] text by party name, replacing the share at one decimal
 * @returns {HTMLElement[]} the `td` cells
 */
function valueCells(values, parties, catalog, texts = null) {
  const leaders = leaderOf(Object.fromEntries(parties.map((name) => [name, values[name]])));
  return parties.map((name) => {
    const missing = typeof values[name] !== 'number';
    const text = texts ? texts[name] : fmtNum(values[name], 1);
    const td = el('td', missing ? '—' : text, missing ? 'p na' : 'p');
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
  FIXED_COLUMNS.forEach(([key, label], index) => {
    const th = el('th', label, index === 0 ? 'f l' : '');
    if (key === 'rating') {
      th.title = RATING_TITLE;
    }
    add(th, key);
  });
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
 * First cell of a poll: the house, linked to its profile when the poll carries `href`, over the sponsor.
 *
 * @param {object} poll a poll of the cycle
 * @returns {HTMLElement} the `td` cell
 */
function houseCell(poll) {
  const td = el('td', '', 'f l');
  const name = el('b', poll.pollster || '–');
  if (poll.href) {
    const link = el('a');
    link.href = poll.href;
    link.appendChild(name);
    td.appendChild(link);
  } else {
    td.appendChild(name);
  }
  td.appendChild(el('small', poll.sponsor || '—'));
  return td;
}

/**
 * Rating cell of a poll: the badge with the rating at one decimal on its colour (red at 0, grey at 50, green at
 * 100), or a dash without rating.
 *
 * @param {object} poll a poll of the cycle
 * @returns {HTMLElement} the `td` cell
 */
function ratingCell(poll) {
  const color = gradient(poll.rating, 0, 100, RATING_COLORS);
  if (color === 'transparent') {
    return el('td', '—', 'na');
  }
  const badge = el('span', fmtNum(poll.rating, 1), 'rt');
  badge.style.background = color;
  const td = el('td');
  td.appendChild(badge);
  return td;
}

/**
 * Rebuild the body with one page of polls.
 *
 * @param {HTMLTableSectionElement} tbody body section
 * @param {Array} polls polls of the page
 * @param {string[]} parties columns shown
 * @param {Catalog} catalog party catalogue
 * @param {string} [mode='pct'] view: shares, or `esc` for the seat projections (`mín–máx` for a range)
 */
export function renderBody(tbody, polls, parties, catalog, mode = 'pct') {
  if (!polls.length) {
    const td = el('td', 'Ningún sondeo con estos filtros.', 'empty');
    td.colSpan = FIXED_COLUMNS.length + parties.length;
    const tr = el('tr');
    tr.appendChild(td);
    tbody.replaceChildren(tr);
    return;
  }
  tbody.replaceChildren(...polls.map((poll) => {
    const date = el('td', shortDate(poll.date));
    if (poll.start_date || poll.end_date) {
      date.title = `Trabajo de campo: ${fmtDate(poll.start_date || poll.end_date)} – ${fmtDate(poll.end_date || poll.start_date)}`;
    }
    const tr = el('tr');
    const texts = mode === 'esc' ? Object.fromEntries(parties.map((name) => [name, seatsText(poll, name)])) : null;
    tr.append(houseCell(poll), ratingCell(poll), date, el('td', fmtInt(poll.sample_size)),
      ...valueCells(valuesOf(poll, parties, mode), parties, catalog, texts));
    return tr;
  }));
}

/**
 * Rebuild the footer with the simple mean of the filtered polls and, in the detail of a past election, the rows
 * pinned under it (`fixedRows`: the model at its shortest horizon and the official result).
 *
 * @param {HTMLTableSectionElement} tfoot footer section
 * @param {number} count polls averaged
 * @param {Object<string, (number|null)>} means mean by party name
 * @param {string[]} parties columns shown
 * @param {Catalog} catalog party catalogue
 * @param {string} [mode='pct'] view: shares at one decimal, or `esc` for whole seats
 * @param {Array<{label: string, small: string, title: (string|null), values: object, texts: (object|null)}>}
 *   [fixed=[]] the pinned rows, already in the view (`fixedRows`)
 */
export function renderFoot(tfoot, count, means, parties, catalog, mode = 'pct', fixed = []) {
  const first = el('td', '', 'f l');
  first.append(el('b', 'Media simple'), el('small', pollsCount(count) + (mode === 'esc' ? ' con escaños' : '')));
  const tr = el('tr');
  const texts = mode === 'esc' ? Object.fromEntries(parties.map((name) => [name, fmtInt(means[name])])) : null;
  tr.append(first, el('td'), el('td'), el('td'), ...valueCells(means, parties, catalog, texts));
  const pinned = fixed.map((row) => {
    const head = el('td', '', 'f l');
    head.append(el('b', row.label), el('small', row.small));
    if (row.title) {
      head.title = row.title;
    }
    const line = el('tr', '', 'fixed');
    line.append(head, el('td'), el('td'), el('td'), ...valueCells(row.values, parties, catalog, row.texts));
    return line;
  });
  tfoot.replaceChildren(tr, ...pinned);
}
