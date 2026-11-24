// Table of changes of the evolution of the updates (Luis, 2026-11-04): under the chart, the figure of each name on
// the last day and its change against the day before and against two key dates of the election, the start of its
// campaign and its call (Luis, 2026-11-23, in place of a week back and the first day of the series). The changes are
// between the figures shown (`shownValue`), the rule of the donuts of the chart, so the table and the chart agree.
import {escapeHtml} from './charts/base.js';
import {FIELD_DIGITS, runDay, shownValue} from './charts/evolution.js';
import {fmtBadge, fmtDate, fmtDateShort, fmtPct} from './format.js';

/** Columns of the key dates, in the order of the table, and what the title of each header says after its date. */
const DATE_COLUMNS = [['campaign', 'inicio de la campaña'], ['call', 'convocatoria']];

/** Header date of a column: «4-nov» (Luis, 2026-11-04). */
const headDate = (iso) => fmtDateShort(iso).replace(' ', '-');

/** The value of `name` in the `nowcast` headline of `run`, null when missing. */
function valueOf(run, name, field) {
  const party = ((run.nowcast || {}).parties || []).find((row) => row.name === name);
  const value = party ? party[field] : null;
  return value === undefined ? null : value;
}

/**
 * Changes of each name between the last daily run and up to three earlier days: the day before (the previous day
 * with a value of the name, as the donut), the start of the campaign and the call of the election. The reference of
 * a key date is the last daily run on that date or before it, and its column is left out (null) without the date,
 * while the last day is not after it (the campaign has not started: a day is never compared with itself), when no
 * run is that old, or when its day already heads an earlier column (the day after the call, the day before is the
 * call).
 *
 * @param {object[]} runs daily runs of the view (`dailyRuns`), ascending by day
 * @param {string[]} names party or block names, in legend order
 * @param {string} [field='pct'] party field (`pct` or `seats`)
 * @param {{campaign: (string|null), call: (string|null)}} [dates] ISO dates of the start of the campaign and of the
 *   call (the key dates of the election); none by default
 * @returns {object|null} `{days: {last, previous, campaign, call}, rows: [{name, value, day, campaign, call}]}`
 *   (`campaign` and `call` null without their column; a change null without a value on either day), or null with
 *   fewer than two days
 */
export function evolutionChanges(runs, names, field = 'pct', dates = {}) {
  if (!runs || runs.length < 2) {
    return null;
  }
  const lastRun = runs[runs.length - 1];
  const last = runDay(lastRun);
  const previous = runDay(runs[runs.length - 2]);
  // the reference run of each key date, by column; the days already heading a column are not repeated
  const taken = new Set([previous]);
  const references = {};
  for (const [key] of DATE_COLUMNS) {
    const date = (dates || {})[key] || null;
    const run = date && last > date ? [...runs].reverse().find((entry) => runDay(entry) <= date) || null : null;
    references[key] = run && !taken.has(runDay(run)) ? run : null;
    if (references[key]) {
      taken.add(runDay(run));
    }
  }
  const change = (now, then) => (now === null || then === null ? null : shownValue(now - then, field));

  const rows = names.map((name) => {
    const value = shownValue(valueOf(lastRun, name, field), field);
    const before = runs.slice(0, -1).reverse().find((run) => valueOf(run, name, field) !== null);
    const shownOf = (run) => (run ? shownValue(valueOf(run, name, field), field) : null);
    return {
      name,
      value,
      day: change(value, shownOf(before)),
      campaign: change(value, shownOf(references.campaign)),
      call: change(value, shownOf(references.call)),
    };
  });
  const dayOf = (run) => (run ? runDay(run) : null);
  return {days: {last, previous, campaign: dayOf(references.campaign), call: dayOf(references.call)}, rows};
}

/**
 * HTML of the table of changes (`evolutionChanges`): a site table with the name (dot and full name in the title),
 * the figure of the last day in bold and the arrows of the changes (`.dlt`), each column headed by its date
 * (`headDate`: the last run, then «vs» the day before, the start of the campaign and the call; Luis, 2026-11-04 and
 * 2026-11-23), the title of the header saying what its date is, one row per name with `data-name`: every name of
 * the view, whatever the legend shows. A key date without a day (`days.campaign`, `days.call`) has no column.
 *
 * @param {object} changes `evolutionChanges` result
 * @param {object} view `colors` and `fullnames` by name
 * @param {object} [options] `field` (`pct` or `seats`) and `formatter` (value → text)
 * @returns {string} the `<table>` markup
 */
export function changesTableHtml(changes, view, {field = 'pct', formatter = fmtPct} = {}) {
  const digits = FIELD_DIGITS[field] ?? 0;
  const {days} = changes;
  const columns = [
    {key: 'day', day: days.previous, what: ''},
    ...DATE_COLUMNS.filter(([key]) => days[key]).map(([key, what]) => ({key, day: days[key], what: `, ${what}`})),
  ];
  const head = ['<th scope="col">Partido</th>', `<th scope="col">${escapeHtml(headDate(days.last))}</th>`,
    ...columns.map((col) => `<th scope="col" title="Frente al ${escapeHtml(fmtDate(col.day) + col.what)}">vs ${escapeHtml(headDate(col.day))}</th>`)];
  const body = changes.rows.map((row) => {
    const color = (view.colors || {})[row.name] || '';
    const fullname = (view.fullnames || {})[row.name] || row.name;
    const cells = [
      `<td title="${escapeHtml(fullname)}"><span class="pn"><span class="dot" style="background-color: ${escapeHtml(color)}"></span>${escapeHtml(row.name)}</span></td>`,
      `<td><strong>${escapeHtml(formatter(row.value))}</strong></td>`,
      ...columns.map((col) => {
        const badge = fmtBadge(row[col.key], digits);
        return `<td><span class="dlt ${badge.dir}">${escapeHtml(badge.text)}</span></td>`;
      }),
    ];
    return `<tr data-name="${escapeHtml(row.name)}">${cells.join('')}</tr>`;
  });
  return `<table><thead><tr>${head.join('')}</tr></thead><tbody>${body.join('')}</tbody></table>`;
}
