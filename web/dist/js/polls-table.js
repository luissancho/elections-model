// Pure logic of the "Todas las encuestas" table of Promedios: months, filtering, sorting, pagination, simple means
// and leaders. The page script only reads the controls, calls these functions and writes the DOM.

/** Rows per page. */
export const PAGE_SIZE = 15;

const MONTHS = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
const DASH = '–';

/**
 * Count of polls with its noun.
 *
 * @param {number} n number of polls
 * @returns {string} `1 sondeo`, or `{n} sondeos` otherwise
 */
export function pollsCount(n) {
  return `${n} ${n === 1 ? 'sondeo' : 'sondeos'}`;
}

/**
 * Month of a poll.
 *
 * @param {object} poll poll with an ISO `date`
 * @returns {string|null} `YYYY-MM`, or null without a valid date
 */
export function monthOf(poll) {
  const date = poll && poll.date;
  return typeof date === 'string' && /^\d{4}-\d{2}/.test(date) ? date.slice(0, 7) : null;
}

/**
 * Months with polls.
 *
 * @param {Array} polls polls of the cycle
 * @returns {string[]} the distinct `YYYY-MM` months, ascending
 */
export function monthsOf(polls) {
  return [...new Set(polls.map(monthOf).filter((month) => month !== null))].sort();
}

/**
 * Label of a month.
 *
 * @param {string} month `YYYY-MM`
 * @returns {string} e.g. `Ago 2023`
 */
export function monthLabel(month) {
  const [year, number] = month.split('-');
  return `${MONTHS[Number(number) - 1]} ${year}`;
}

/**
 * Numeric short date.
 *
 * @param {string|null} iso ISO date
 * @returns {string} `dd/mm/aa`, or `–` without a valid date
 */
export function shortDate(iso) {
  const match = typeof iso === 'string' ? /^(\d{4})-(\d{2})-(\d{2})/.exec(iso) : null;
  return match ? `${match[3]}/${match[2]}/${match[1].slice(2)}` : DASH;
}

/**
 * Polls of the chosen pollsters and months.
 *
 * @param {Array} polls polls of the cycle
 * @param {{pollsters: (string[]|null), from: (string|null), to: (string|null)}} filter pollster names (null: all)
 *   and inclusive `YYYY-MM` months (null or empty: no limit); a poll without a date fails any month limit
 * @returns {Array} the polls that pass, in their order
 */
export function filterPolls(polls, {pollsters, from, to}) {
  const names = pollsters === null ? null : new Set(pollsters);
  return polls.filter((poll) => {
    if (names !== null && !names.has(poll.pollster)) {
      return false;
    }
    const month = monthOf(poll);
    if (from && (month === null || month < from)) {
      return false;
    }
    return !(to && (month === null || month > to));
  });
}

/**
 * Sort polls by a column.
 *
 * @param {Array} polls polls to sort, in publication order (not modified)
 * @param {string} key `date`, `pollster`, `sample_size` or a party name
 * @param {boolean} asc ascending order
 * @returns {Array} a sorted copy: missing values last in both directions; ties by date descending and then by
 *   publication, the later published first (the order of the server); sorting by `date` ascending is the exact
 *   reverse of descending
 */
export function sortPolls(polls, key, asc) {
  const sign = asc ? 1 : -1;
  const byDate = (a, b) => (b.poll.date || '').localeCompare(a.poll.date || '') || b.index - a.index;
  const indexed = polls.map((poll, index) => ({poll, index}));
  const missing = (value) => value === null || value === undefined || value === '';
  indexed.sort((a, b) => {
    const x = a.poll[key];
    const y = b.poll[key];
    if (missing(x) || missing(y)) {
      return missing(x) === missing(y) ? byDate(a, b) : (missing(x) ? 1 : -1);
    }
    if (key === 'date') {
      return sign * (x.localeCompare(y) || a.index - b.index);
    }
    const order = typeof x === 'string' ? x.localeCompare(y, 'es') : x - y;
    return order === 0 ? byDate(a, b) : sign * order;
  });
  return indexed.map((entry) => entry.poll);
}

/**
 * One page of rows.
 *
 * @param {Array} rows rows in display order
 * @param {number} page zero-based page
 * @param {number} [size=PAGE_SIZE] rows per page
 * @returns {Array} the rows of the page
 */
export function pageOf(rows, page, size = PAGE_SIZE) {
  return rows.slice(page * size, (page + 1) * size);
}

/**
 * Range label of a page.
 *
 * @param {number} page zero-based page
 * @param {number} shown rows of the page
 * @param {number} total rows in all pages
 * @returns {string} e.g. `1–15 de 315`; `0 de 0` without rows
 */
export function pageLabel(page, shown, total) {
  if (shown === 0) {
    return `${total} de ${total}`;
  }
  const first = page * PAGE_SIZE + 1;
  return `${first}–${first + shown - 1} de ${total}`;
}

/**
 * Unweighted mean of each party over the polls that give it.
 *
 * @param {Array} rows polls
 * @param {string[]} parties party names
 * @returns {Object<string, (number|null)>} the mean at one decimal, null when no poll has the party
 */
export function meanOf(rows, parties) {
  const means = {};
  for (const name of parties) {
    const values = rows.map((row) => row[name]).filter((value) => typeof value === 'number' && Number.isFinite(value));
    means[name] = values.length ? Math.round((values.reduce((sum, v) => sum + v, 0) / values.length) * 10) / 10 : null;
  }
  return means;
}

/**
 * Parties with the highest value.
 *
 * @param {Object<string, (number|null)>} values value by party name
 * @returns {string[]} every name tied at the maximum among the present values; empty without values
 */
export function leaderOf(values) {
  const present = Object.entries(values).filter(([, value]) => typeof value === 'number' && Number.isFinite(value));
  if (!present.length) {
    return [];
  }
  const top = Math.max(...present.map(([, value]) => value));
  return present.filter(([, value]) => value === top).map(([name]) => name);
}
