// Table of changes of the evolution of the updates (Luis, 2026-11-04): under the chart, the figure of each name on
// the last day and its change against the day before, a week back and the first day of the series. The changes are
// between the figures shown (`shownValue`), the rule of the donuts of the chart, so the table and the chart agree.
import {escapeHtml} from './charts/base.js';
import {FIELD_DIGITS, runDay, shownValue} from './charts/evolution.js';
import {dayTime, isoDay} from './charts/series.js';
import {fmtBadge, fmtDate, fmtDateShort, fmtPct} from './format.js';

const DAY_MS = 24 * 3600 * 1000;
const WEEK_DAYS = 7;

/** ISO date of the browser's local day now (`isoDay` rounds to the nearest midnight, so it gets the day's own). */
function localToday() {
  return isoDay(new Date(Date.now()).setHours(0, 0, 0, 0));
}

/** The value of `name` in the `nowcast` headline of `run`, null when missing. */
function valueOf(run, name, field) {
  const party = ((run.nowcast || {}).parties || []).find((row) => row.name === name);
  const value = party ? party[field] : null;
  return value === undefined ? null : value;
}

/**
 * Changes of each name between the last daily run and three earlier days: the day before (the previous day with a
 * value of the name, as the donut), the last day on or before a week back and the first day of the series.
 *
 * @param {object[]} runs daily runs of the view (`dailyRuns`), ascending by day
 * @param {string[]} names party or block names, in legend order
 * @param {string} [field='pct'] party field (`pct` or `seats`)
 * @param {string} [today] ISO date of today (the browser's local day), to title the first change column «Hoy» (else
 *   «1 día»)
 * @returns {object|null} `{days: {last, previous, week, start}, todayLabel, rows: [{name, value, day, week, start}]}`
 *   (`week` null without a day a week back; a change null without a value on either day), or null with fewer than
 *   two days
 */
export function evolutionChanges(runs, names, field = 'pct', today = localToday()) {
  if (!runs || runs.length < 2) {
    return null;
  }
  const lastRun = runs[runs.length - 1];
  const last = runDay(lastRun);
  const weekBack = isoDay(dayTime(last) - WEEK_DAYS * DAY_MS);
  const weekRun = [...runs].reverse().find((run) => runDay(run) <= weekBack) || null;
  const startRun = runs[0];
  const change = (now, then) => (now === null || then === null ? null : shownValue(now - then, field));

  const rows = names.map((name) => {
    const value = shownValue(valueOf(lastRun, name, field), field);
    const before = runs.slice(0, -1).reverse().find((run) => valueOf(run, name, field) !== null);
    const shownOf = (run) => (run ? shownValue(valueOf(run, name, field), field) : null);
    return {
      name,
      value,
      day: change(value, shownOf(before)),
      week: change(value, shownOf(weekRun)),
      start: change(value, shownOf(startRun)),
    };
  });
  return {
    days: {last, previous: runDay(runs[runs.length - 2]), week: weekRun ? runDay(weekRun) : null, start: runDay(startRun)},
    // «Hoy» only when the last day is today; otherwise «1 día», not the date already heading the figure column
    todayLabel: last === today ? 'Hoy' : '1 día',
    rows,
  };
}

/**
 * HTML of the table of changes (`evolutionChanges`): a site table with the name (dot and full name in the title),
 * the figure of the last day in bold and the arrows of the changes (`.dlt`), one row per name with `data-name`,
 * hidden when not in `shown`. Without a day a week back the «7 días» column is left out.
 *
 * @param {object} changes `evolutionChanges` result
 * @param {object} view `colors` and `fullnames` by name
 * @param {object} [options] `field` (`pct` or `seats`), `formatter` (value → text) and `shown` (names shown; all
 *   when null)
 * @returns {string} the `<table>` markup
 */
export function changesTableHtml(changes, view, {field = 'pct', formatter = fmtPct, shown = null} = {}) {
  const digits = FIELD_DIGITS[field] ?? 0;
  const {days} = changes;
  const columns = [
    {key: 'day', label: changes.todayLabel, day: days.previous},
    ...(days.week ? [{key: 'week', label: `${WEEK_DAYS} días`, day: days.week}] : []),
    {key: 'start', label: `Desde ${fmtDateShort(days.start)}`, day: days.start},
  ];
  const head = ['<th scope="col">Partido</th>', `<th scope="col">${escapeHtml(fmtDateShort(days.last))}</th>`,
    ...columns.map((col) => `<th scope="col" title="Frente al ${escapeHtml(fmtDate(col.day))}">${escapeHtml(col.label)}</th>`)];
  const body = changes.rows.map((row) => {
    const hidden = shown && !shown.includes(row.name) ? ' hidden' : '';
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
    return `<tr data-name="${escapeHtml(row.name)}"${hidden}>${cells.join('')}</tr>`;
  });
  return `<table><thead><tr>${head.join('')}</tr></thead><tbody>${body.join('')}</tbody></table>`;
}
