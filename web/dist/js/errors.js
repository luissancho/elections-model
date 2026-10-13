// Error history of the polls (the `poll-errors` rows embedded in /elecciones): filters, means, the table by weeks
// before the election and the URL hash of the filters. Pure logic, no DOM and no ECharts, so node can test it.

/** Measures the histogram and the weeks table can show; the first one is the default. */
export const MEASURES = ['error_blocks_abs', 'error_avg'];
export const DEFAULT_MEASURE = MEASURES[0];

// Keys of the URL hash.
const EVENT_KEY = 'eleccion';
const POLLSTER_KEY = 'casa';
const MEASURE_KEY = 'medida';

/** Label of the row that sums up every election in the weeks table. */
export const ALL_LABEL = 'Todas';

/**
 * Rows of one election and/or one house.
 *
 * @param {object[]} rows poll-errors rows (`event_date`, `pollster`, …)
 * @param {{event: (string|null), pollster: (string|null)}} filters election date and house name; `null` (or
 *   missing) keeps every value
 * @returns {object[]} the matching rows, in their order
 */
export function filterRows(rows, {event = null, pollster = null} = {}) {
  return rows.filter((row) => (event === null || row.event_date === event)
    && (pollster === null || row.pollster === pollster));
}

/**
 * Mean of a measure over the rows that have a finite value of it.
 *
 * @param {object[]} rows poll-errors rows
 * @param {string} measure column name
 * @returns {number|null} the mean, or `null` without any finite value
 */
export function meanOf(rows, measure) {
  const values = rows.map((row) => row[measure]).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

/**
 * Week before the election of a poll published `days` days before it, as in the PollsErrors notebook.
 *
 * @param {number} days days between the poll and the election
 * @returns {number} `floor((days + 1) / 7)`
 */
export function weekOf(days) {
  return Math.floor((days + 1) / 7);
}

/**
 * Polls and mean of a measure per week, for a set of rows.
 *
 * @param {object[]} rows poll-errors rows
 * @param {string} measure column name
 * @returns {object} `{week: {n, mean}}`: `n` counts every poll of the week, `mean` only the finite values
 */
function weekCells(rows, measure) {
  const groups = new Map();
  for (const row of rows) {
    const week = weekOf(row.days);
    if (!groups.has(week)) {
      groups.set(week, []);
    }
    groups.get(week).push(row);
  }
  const cells = {};
  for (const [week, group] of [...groups].sort((a, b) => a[0] - b[0])) {
    cells[week] = {n: group.length, mean: meanOf(group, measure)};
  }
  return cells;
}

/**
 * Table of a measure by election and week before it.
 *
 * @param {object[]} rows poll-errors rows (`event`, `event_date`, `days` and the measure)
 * @param {string} measure column name
 * @returns {{weeks: number[], rows: object[]}} the weeks present (ascending) and one row per election, newest
 *   first, `{event, event_date, cells: {week: {n, mean}}}`, closed by the `Todas` row of every poll
 */
export function weeksTable(rows, measure) {
  const valid = rows.filter((row) => Number.isFinite(row.days));
  const weeks = [...new Set(valid.map((row) => weekOf(row.days)))].sort((a, b) => a - b);
  const events = new Map();
  for (const row of valid) {
    if (!events.has(row.event_date)) {
      events.set(row.event_date, {event: row.event, event_date: row.event_date, rows: []});
    }
    events.get(row.event_date).rows.push(row);
  }
  const byEvent = [...events.values()]
    .sort((a, b) => (a.event_date < b.event_date ? 1 : a.event_date > b.event_date ? -1 : 0))
    .map(({event, event_date: date, rows: group}) => ({event, event_date: date, cells: weekCells(group, measure)}));
  byEvent.push({event: ALL_LABEL, event_date: null, cells: weekCells(valid, measure)});
  return {weeks, rows: byEvent};
}

/**
 * Filters of the error history from the URL hash; a value not offered falls back to its default.
 *
 * @param {string} hash `location.hash`, e.g. `#eleccion=2023-07-23&casa=GAD3&medida=error_avg`
 * @param {{events: string[], pollsters: string[]}} options the election dates and house names offered
 * @returns {{event: (string|null), pollster: (string|null), measure: string}} the filters (`null` = all)
 */
export function parseErrorsHash(hash, {events, pollsters}) {
  const params = new Map();
  for (const entry of (hash || '').replace(/^#/, '').split('&')) {
    const at = entry.indexOf('=');
    if (at > 0) {
      let value;
      try {
        value = decodeURIComponent(entry.slice(at + 1));
      } catch {
        continue;
      }
      params.set(entry.slice(0, at), value);
    }
  }
  const event = params.get(EVENT_KEY);
  const pollster = params.get(POLLSTER_KEY);
  const measure = params.get(MEASURE_KEY);
  return {
    event: events.includes(event) ? event : null,
    pollster: pollsters.includes(pollster) ? pollster : null,
    measure: MEASURES.includes(measure) ? measure : DEFAULT_MEASURE,
  };
}

/**
 * URL hash of the filters, the inverse of `parseErrorsHash`.
 *
 * @param {{event: (string|null), pollster: (string|null), measure: string}} state the filters
 * @returns {string} `#eleccion=…&casa=…&medida=…` with only the filters set (`medida` only when it is not the
 *   default), or `''` when none is
 */
export function errorsHash({event, pollster, measure}) {
  const parts = [];
  if (event !== null && event !== undefined) {
    parts.push(`${EVENT_KEY}=${encodeURIComponent(event)}`);
  }
  if (pollster !== null && pollster !== undefined) {
    parts.push(`${POLLSTER_KEY}=${encodeURIComponent(pollster)}`);
  }
  if (measure && measure !== DEFAULT_MEASURE) {
    parts.push(`${MEASURE_KEY}=${encodeURIComponent(measure)}`);
  }
  return parts.length ? `#${parts.join('&')}` : '';
}
