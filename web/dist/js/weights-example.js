// The weights example of /metodo/promedio («Un cálculo completo, con datos publicados»), without the DOM: load the
// `weights` and `meta` parts of a scope (pinned to its run by the server), order its parties, slice the view of one
// party that `charts/weights.js` draws, and write the figures of the text and the rows of the table exactly as the
// server writes them (`pages/filters.py`), so going back to the general elections and their leader leaves the card
// as it was served. `pages/metodo.js` wires it to the card.
import {DASH, fmtCount, fmtDate, fmtDateShort, fmtNum, fmtPct} from './format.js';

// The scope whose texts name no region («el promedio del PP»).
const NATIONAL_SCOPE = 'es';
// The weight that fills the bar of its cell (`bar(0, 3)` in the template the table came from).
const WEIGHT_BAR_MAX = 3;
// The factors of a weight, after the weight itself, in the order of the columns.
const FACTORS = ['weight_kernel', 'weight_over', 'weight_sample', 'weight_rating'];

/** Escape a text for HTML: the names of pollsters and sponsors come from the data. */
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

/** Ask for one JSON part and return its `data`; rejected when the response is not OK or has no `data`. */
async function loadPart(href, fetchImpl) {
  const response = await fetchImpl(href, {headers: {Accept: 'application/json'}});
  if (!response.ok) {
    throw new Error(`Could not load ${href}: HTTP ${response.status}`);
  }
  const body = await response.json();
  if (!body || typeof body.data !== 'object' || body.data === null) {
    throw new Error(`Could not load ${href}: no data in the response`);
  }
  return body.data;
}

/**
 * Load the parts the example of one scope needs, in parallel.
 *
 * @param {object} scope a row of `example_scopes` (`weights_href`, `meta_href`)
 * @param {function} [fetchImpl] `fetch` or a stand-in with its signature (the tests)
 * @returns {Promise<{weights: object, meta: object}>} the `data` of both parts; rejected when a request fails, a
 *   response is not OK or has no `data`, or the `weights` part has no parties
 */
export async function loadExample(scope, fetchImpl = fetch) {
  const [weights, meta] = await Promise.all([loadPart(scope.weights_href, fetchImpl), loadPart(scope.meta_href, fetchImpl)]);
  if (!Array.isArray(weights.parties) || !weights.parties.length) {
    throw new Error(`Could not load ${scope.weights_href}: the part has no parties`);
  }
  return {weights, meta};
}

/**
 * The parties of a `weights` part, the highest average first; those without one keep their order at the end.
 *
 * @param {object} weights the `weights` part
 * @returns {string[]} party names
 */
export function partyOptions(weights) {
  const parties = weights.parties || [];
  const mean = (name) => {
    const value = ((weights.stats || {})[name] || {}).mean;
    return value === null || value === undefined || Number.isNaN(value) ? null : value;
  };
  return parties
    .map((name, index) => ({name, index, mean: mean(name)}))
    .sort((a, b) => {
      if (a.mean === null || b.mean === null) {
        return a.mean === b.mean ? a.index - b.index : (a.mean === null ? 1 : -1);
      }
      return b.mean - a.mean || a.index - b.index;
    })
    .map((row) => row.name);
}

/** The party shown when a scope is picked: the first of `partyOptions`, `null` without parties. */
export function defaultParty(weights) {
  return partyOptions(weights)[0] ?? null;
}

/**
 * The view of one party: what `renderWeights` draws plus what the texts and the table need.
 *
 * @param {object} weights the `weights` part
 * @param {object} meta the `meta` part (the colour of the party)
 * @param {string} party party name
 * @returns {object|null} `{name, color, date, stats, rows, line, kernel, n_rows, top}`, the rows newest first and
 *   `top` the heaviest poll (`{pollster, sponsor, date, weight}`, the newest on a tie; `null` without a positive
 *   weight); `null` when the part does not have the party
 */
export function exampleView(weights, meta, party) {
  if (!(weights.parties || []).includes(party)) {
    return null;
  }
  const rows = (weights.rows || []).filter((row) => row.name === party)
    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  let heaviest = null;
  rows.forEach((row) => {
    const weight = row.weight || 0;
    if (weight > 0 && (heaviest === null || weight > heaviest.weight)) {   // newest first: a tie keeps the newest
      heaviest = row;
    }
  });
  const catalogue = ((meta && meta.parties) || []).find((row) => row.name === party) || {};
  return {
    name: party,
    color: catalogue.color ?? null,
    date: weights.date,
    stats: (weights.stats || {})[party] || {},
    rows,
    line: (weights.line || {})[party],
    kernel: (weights.kernel || {})[party],
    n_rows: rows.length,
    top: heaviest && {pollster: heaviest.pollster, sponsor: heaviest.sponsor ?? null, date: heaviest.date, weight: heaviest.weight},
  };
}

/** Who the average is of: the party, and its region outside the general elections («PSOE en Andalucía»). */
export function subjectOf(name, scope) {
  return scope.code === NATIONAL_SCOPE ? name : `${name} en ${scope.name}`;
}

/**
 * The figures of the text of the card, one per `data-w` span, as the template writes them.
 *
 * @param {object} view an `exampleView`
 * @param {object} scope its row of `example_scopes` (`code`, `name`, `published`)
 * @returns {object} `published`, `subject`, `date`, `mean`, `lo`, `hi`, `nobs`, `bandwidth`, `neff`, `n_rows`,
 *   `top` (the sentence on the heaviest poll with its leading space, `''` without one) and `dates_hidden` (the
 *   paragraph on the two dates only shows when the day read is after the run)
 */
export function exampleTexts(view, scope) {
  const stats = view.stats || {};
  const top = view.top;
  return {
    published: fmtDate(scope.published),
    subject: subjectOf(view.name, scope),
    date: fmtDate(view.date),
    mean: fmtPct(stats.mean),
    lo: fmtPct(stats.lo),
    hi: fmtPct(stats.hi),
    nobs: fmtCount(stats.nobs),   // `int` of the template groups four digits too ('1.200'), as fmtCount
    bandwidth: fmtNum(stats.bandwidth, 0),
    neff: fmtNum(stats.neff, 1),
    n_rows: String(view.n_rows),
    top: top ? ` El sondeo que más pesa es el de ${top.pollster} del ${fmtDateShort(top.date)}, con ${fmtNum(top.weight, 2)}.` : '',
    dates_hidden: !(view.date > scope.published),
  };
}

/** Width of the bar of a weight, as `bar_width` of `pages/filters.py`: 0 to 100 with one decimal, 0 if missing. */
function barWidth(x, vmax) {
  if (x === null || x === undefined || Number.isNaN(x)) {
    return 0;
  }
  return Math.round(1000 * Math.min(1, Math.max(0, x / vmax))) / 10;
}

/**
 * The rows of the table of polls, with the markup the template used to write.
 *
 * @param {object[]} rows the `rows` of an `exampleView`
 * @returns {string} the `<tr>` elements, every text escaped
 */
export function tableRowsHtml(rows) {
  return rows.map((row) => {
    const house = `<b>${escapeHtml(row.pollster ?? DASH)}</b>${row.sponsor ? ` / ${escapeHtml(row.sponsor)}` : ''}`;
    const weight = `<td class="cell-bar"><i class="fill w" style="left: 0; width: ${barWidth(row.weight, WEIGHT_BAR_MAX)}%"></i>`
      + `<span>${fmtNum(row.weight, 2)}</span></td>`;
    const factors = FACTORS.map((key) => `<td>${fmtNum(row[key], 2)}</td>`).join('');
    return `<tr><td class="f l">${fmtDateShort(row.date)}</td><td>${house}</td><td>${fmtCount(row.sample_size)}</td>`
      + `<td>${fmtPct(row.pct)}</td><td>${fmtPct(row.pct_adj)}</td>${weight}${factors}</tr>`;
  }).join('');
}
