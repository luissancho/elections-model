// DOM builders of the "Resultados por territorio" table of Escaños: header and body, built with
// `createElement`/`textContent` (never HTML with data).
import {fmtInt, fmtNum} from '../format.js';
import {columnRange, tintAlpha, winnerOf} from '../territory.js';

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
 * Text of a cell value.
 *
 * @param {(number|null)} value seats or vote
 * @param {string} mode `esc` or `pct`
 * @returns {string} the formatted value, or a dash without value or equal to zero
 */
function cellText(value, mode) {
  if (typeof value !== 'number' || value <= 0) {
    return '—';
  }
  return mode === 'pct' ? `${fmtNum(value, 1)} %` : fmtInt(value);
}

/**
 * Party colour with an alpha, as a `#rrggbbaa` colour.
 *
 * @param {string} color `#rrggbb` colour
 * @param {number} alpha alpha in [0, 1]
 * @returns {string} the colour with a two-digit hexadecimal alpha
 */
function withAlpha(color, alpha) {
  return `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
}

/**
 * Rebuild the header row.
 *
 * @param {HTMLTableSectionElement} thead header section
 * @param {string[]} parties party columns
 * @param {Catalog} catalog party catalogue
 * @param {{key: string, asc: boolean}} sort active sort, marked with `aria-sort`
 * @param {string} first label of the territory column
 */
export function renderHead(thead, parties, catalog, sort, first) {
  const tr = el('tr');
  const add = (th, key) => {
    th.dataset.key = key;
    const button = el('button', '', 'sort');
    button.type = 'button';
    button.append(...th.childNodes);
    th.appendChild(button);
    if (key === sort.key) {
      th.setAttribute('aria-sort', sort.asc ? 'ascending' : 'descending');
    }
    tr.appendChild(th);
  };
  add(el('th', first, 'l'), 'name');
  for (const name of parties) {
    const th = el('th', '', 'p');
    th.scope = 'col';
    th.title = catalog.fullname(name);
    const dot = el('span', '', 'dot');
    dot.style.background = catalog.color(name);
    th.append(dot, name);
    add(th, name);
  }
  const total = el('th', 'Total');
  total.scope = 'col';
  add(total, 'total');
  thead.replaceChildren(tr);
}

/**
 * Rebuild the body: one row per territory and the TOTAL row.
 *
 * @param {HTMLTableSectionElement} tbody body section
 * @param {{rows: Array, parties: string[], total: object}} data sorted rows (`rows`), columns and total row
 * @param {string} mode `esc` or `pct`
 * @param {Catalog} catalog party catalogue
 * @param {function(object): (string|null)} hrefOf link of a row, or null when it has none
 */
export function renderBody(tbody, data, mode, catalog, hrefOf) {
  const {rows, parties} = data;
  const ranges = Object.fromEntries(parties.map((name) => [name, columnRange(rows, name)]));
  const body = rows.map((row) => {
    const first = el('td');
    const href = hrefOf(row);
    if (href) {
      const link = el('a', row.name);
      link.href = href;
      first.appendChild(link);
    } else {
      first.textContent = row.name;
    }
    const winners = winnerOf(row.values);
    for (const name of winners) {
      const dot = el('span', '', 'tdot');
      dot.style.background = catalog.color(name);
      dot.title = name;
      first.append(' ', dot);
    }
    const tr = el('tr');
    tr.appendChild(first);
    for (const name of parties) {
      const value = row.values[name];
      const cell = el('span', cellText(value, mode), winners.includes(name) ? 'cell is-win' : 'cell');
      const alpha = tintAlpha(value, ranges[name].min, ranges[name].max);
      if (alpha > 0) {
        cell.style.background = withAlpha(catalog.color(name), alpha);
      }
      if (mode === 'esc' && row.ranges[name]) {
        cell.title = row.ranges[name];
      }
      const td = el('td');
      td.appendChild(cell);
      tr.appendChild(td);
    }
    tr.appendChild(el('td', mode === 'pct' ? row.total : cellText(row.total, 'esc')));
    return tr;
  });
  const foot = el('tr', '', 'total');
  foot.appendChild(el('td', 'TOTAL'));
  for (const name of parties) {
    foot.appendChild(el('td', cellText(data.total.values[name], mode)));
  }
  foot.appendChild(el('td', mode === 'pct' ? data.total.total : cellText(data.total.total, 'esc')));
  tbody.replaceChildren(...body, foot);
}
