// Error history of the polls (the `poll-errors` rows embedded in /elecciones): filters, means, the table by weeks
// before the election and the URL hash of the filters. Pure logic, no DOM and no ECharts, so node can test it.

/** Measures the histogram and the weeks table can show; the first one is the default. */
export const MEASURES = ['error_blocks_abs', 'error_avg'];
export const DEFAULT_MEASURE = MEASURES[0];

// Keys of the URL hash.
const SCOPE_KEY = 'ambito';
const EVENT_KEY = 'eleccion';
const POLLSTER_KEY = 'casa';
const FROM_KEY = 'desde';
const TO_KEY = 'hasta';
const MEASURE_KEY = 'medida';

/** Label of the row that sums up every election in the weeks table. */
export const ALL_LABEL = 'Todas';

/**
 * Key of the election a row belongs to; the date alone is not enough because several scopes vote the same day.
 *
 * @param {{scope: string, event_date: string}} row poll-errors row
 * @returns {string} `scope:date`, e.g. `es:2023-07-23`
 */
export function eventKey(row) {
  return `${row.scope}:${row.event_date}`;
}

/**
 * Inclusive year bounds in ascending order.
 *
 * @param {(number|null)} from first year or `null`
 * @param {(number|null)} to last year or `null`
 * @returns {{from: (number|null), to: (number|null)}} the bounds, swapped when `from > to`
 */
function orderedBounds(from, to) {
  if (from !== null && to !== null && from > to) {
    return {from: to, to: from};
  }
  return {from, to};
}

/**
 * Rows of one scope, election, house and/or range of years.
 *
 * @param {object[]} rows poll-errors rows (`scope`, `event_date`, `pollster`, `year`, …)
 * @param {{scope: (string|null), event: (string|null), pollster: (string|null), from: (number|null),
 *   to: (number|null)}} filters scope code, election key (`scope:date`), house name and inclusive years; `null`
 *   (or missing) keeps every value; `from > to` are swapped
 * @returns {object[]} the matching rows, in their order
 */
export function filterRows(rows, {scope = null, event = null, pollster = null, from = null, to = null} = {}) {
  const bounds = orderedBounds(from, to);
  return rows.filter((row) => (scope === null || row.scope === scope)
    && (event === null || eventKey(row) === event)
    && (pollster === null || row.pollster === pollster)
    && (bounds.from === null || row.year >= bounds.from)
    && (bounds.to === null || row.year <= bounds.to));
}

/**
 * Years present in the rows.
 *
 * @param {object[]} rows poll-errors rows
 * @returns {number[]} the distinct years, ascending
 */
export function yearsOf(rows) {
  return [...new Set(rows.map((row) => row.year).filter(Number.isFinite))].sort((a, b) => a - b);
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
 * @param {object[]} rows poll-errors rows (`event`, `event_date`, `days` and the measure; `name`, the full name of
 *   the election, when the page knows it)
 * @param {string} measure column name
 * @returns {{weeks: number[], rows: object[]}} the weeks present (ascending) and one row per election (`scope:date`,
 *   labelled with its `name` or, without one, "{event} · {scope_name}"), newest first,
 *   `{event, event_date, cells: {week: {n, mean}}}`, closed by the `Todas` row of every poll; both empty without any
 *   row with a finite `days`
 */
export function weeksTable(rows, measure) {
  const valid = rows.filter((row) => Number.isFinite(row.days));
  if (!valid.length) {
    return {weeks: [], rows: []};
  }
  const weeks = [...new Set(valid.map((row) => weekOf(row.days)))].sort((a, b) => a - b);
  const events = new Map();
  for (const row of valid) {
    const key = eventKey(row);
    if (!events.has(key)) {
      events.set(key, {event: row.name || `${row.event} · ${row.scope_name}`, event_date: row.event_date, rows: []});
    }
    events.get(key).rows.push(row);
  }
  const byEvent = [...events.values()]
    .sort((a, b) => (a.event_date < b.event_date ? 1 : a.event_date > b.event_date ? -1 : 0))
    .map(({event, event_date: date, rows: group}) => ({event, event_date: date, cells: weekCells(group, measure)}));
  byEvent.push({event: ALL_LABEL, event_date: null, cells: weekCells(valid, measure)});
  return {weeks, rows: byEvent};
}

/**
 * Elections of the rows, one entry each.
 *
 * @param {object[]} rows poll-errors rows
 * @param {string} measure column name
 * @returns {object[]} `{event_date, event, name, scope, scope_name, n, mean}` by date ascending (ties by order of
 *   appearance; `name` is the full name of the election when the rows carry it, else `null`); `n` counts the rows
 *   with a finite `measure` and `mean` is their mean (`null` without any)
 */
export function eventsOf(rows, measure) {
  const groups = new Map();
  for (const row of rows) {
    const key = eventKey(row);
    if (!groups.has(key)) {
      groups.set(key, {event_date: row.event_date, event: row.event, name: row.name ?? null, scope: row.scope,
                       scope_name: row.scope_name, rows: []});
    }
    groups.get(key).rows.push(row);
  }
  return [...groups.values()]
    .map(({rows: group, ...head}) => ({
      ...head,
      n: group.filter((row) => Number.isFinite(row[measure])).length,
      mean: meanOf(group, measure),
    }))
    .sort((a, b) => (a.event_date < b.event_date ? -1 : a.event_date > b.event_date ? 1 : 0));
}

/**
 * Rows outside the selection on one filter, with the rest of the filters applied.
 *
 * @param {object[]} rows poll-errors rows
 * @param {{scope: (string|null), event: (string|null), pollster: (string|null), from: (number|null),
 *   to: (number|null)}} state the active filters
 * @param {string} measure column name
 * @returns {object[]} one `{key, label, n, mean}` per active filter among scope, event, pollster and years, in
 *   that order: the rows that pass every other filter but not that one; empty complements are left out
 */
export function complements(rows, state, measure) {
  const {scope = null, event = null, pollster = null} = state;
  const years = yearsOf(rows);
  const bounds = orderedBounds(state.from ?? null, state.to ?? null);
  const first = years.length ? years[0] : null;
  const last = years.length ? years[years.length - 1] : null;
  const yearsActive = (bounds.from !== null && bounds.from !== first) || (bounds.to !== null && bounds.to !== last);
  const range = (row) => (bounds.from === null || row.year >= bounds.from) && (bounds.to === null || row.year <= bounds.to);
  const rules = [
    {key: 'scope', label: 'Otros ámbitos', active: scope !== null, pass: (row) => row.scope === scope},
    {key: 'event', label: 'Otras elecciones', active: event !== null, pass: (row) => eventKey(row) === event},
    {key: 'pollster', label: 'Otras encuestadoras', active: pollster !== null, pass: (row) => row.pollster === pollster},
    {key: 'years', label: 'Otros años', active: yearsActive, pass: range},
  ];
  const result = [];
  for (const rule of rules.filter((item) => item.active)) {
    const group = rows.filter((row) => rules.every((other) => other === rule || !other.active || other.pass(row))
      && !rule.pass(row));
    if (group.length) {
      result.push({key: rule.key, label: rule.label, n: group.length, mean: meanOf(group, measure)});
    }
  }
  return result;
}

/**
 * Moving average of a measure over the elections, weighted by their polls.
 *
 * @param {object[]} rows poll-errors rows
 * @param {string} measure column name
 * @param {{window: number}} options number of elections averaged, the current one included
 * @returns {Array<[string, number]>} `[event_date, value]` per election with a mean, ascending: the mean of the
 *   last `window` elections weighted by their `n`
 */
export function trendPoints(rows, measure, {window = 5} = {}) {
  const events = eventsOf(rows, measure).filter((item) => item.mean !== null);
  return events.map((item, index) => {
    const recent = events.slice(Math.max(0, index - window + 1), index + 1);
    const weight = recent.reduce((sum, other) => sum + other.n, 0);
    const total = recent.reduce((sum, other) => sum + other.mean * other.n, 0);
    return [item.event_date, total / weight];
  });
}

/**
 * Mean of the other houses in the elections where one house has polls.
 *
 * @param {object[]} rows poll-errors rows
 * @param {string} pollster house name
 * @param {string} measure column name
 * @returns {Array<[string, number]>} `[event_date, value]` ascending: the mean over the finite values of the
 *   other houses, only for the elections where `pollster` has a row and some other house has a value
 */
export function restPoints(rows, pollster, measure) {
  const own = new Set(rows.filter((row) => row.pollster === pollster).map(eventKey));
  const others = rows.filter((row) => row.pollster !== pollster && own.has(eventKey(row)));
  return eventsOf(others, measure)
    .filter((item) => item.mean !== null)
    .map((item) => [item.event_date, item.mean]);
}

/**
 * Filters of the error history from the URL hash; a value not offered falls back to its default.
 *
 * @param {string} hash `location.hash`, e.g. `#ambito=es-md&desde=2021&casa=CIS&medida=error_avg`; an old
 *   `eleccion=2023-07-23` without scope is resolved within the hash scope, or within `defaultScope`
 * @param {{scopes: string[], events: string[], pollsters: string[], years: number[], defaultScope: string}} options
 *   the scope codes, election keys (`scope:date`), house names and years offered; `defaultScope` (default `es`)
 *   is where a date-only `eleccion=` is looked up when the hash has no valid `ambito`
 * @returns {{scope: (string|null), event: (string|null), pollster: (string|null), from: (number|null),
 *   to: (number|null), measure: string}} the filters (`null` = all; `from`/`to` are `null` at the extremes)
 */
export function parseErrorsHash(hash, {scopes = [], events = [], pollsters = [], years = [], defaultScope = 'es'}) {
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
  const scope = scopes.includes(params.get(SCOPE_KEY)) ? params.get(SCOPE_KEY) : null;
  const pollster = params.get(POLLSTER_KEY);
  const measure = params.get(MEASURE_KEY);
  let event = params.get(EVENT_KEY);
  if (event !== undefined && !events.includes(event)) {
    const matches = events.filter((key) => key.slice(key.indexOf(':') + 1) === event
      && key.slice(0, key.indexOf(':')) === (scope ?? defaultScope));
    event = matches.length === 1 ? matches[0] : null;
  }
  const yearBound = (key, extreme) => {
    const year = Number(params.get(key));
    return params.has(key) && years.includes(year) && year !== extreme ? year : null;
  };
  const bounds = orderedBounds(yearBound(FROM_KEY, years[0]), yearBound(TO_KEY, years[years.length - 1]));
  return {
    scope,
    event: event ?? null,
    pollster: pollsters.includes(pollster) ? pollster : null,
    from: bounds.from,
    to: bounds.to,
    measure: MEASURES.includes(measure) ? measure : DEFAULT_MEASURE,
  };
}

/**
 * URL hash of the filters, the inverse of `parseErrorsHash`.
 *
 * @param {{scope: (string|null), event: (string|null), pollster: (string|null), from: (number|null),
 *   to: (number|null), measure: string}} state the filters (`from`/`to` already `null` at the extremes)
 * @returns {string} `#ambito=…&eleccion=…&casa=…&desde=…&hasta=…&medida=…` with only the filters set (`medida`
 *   only when it is not the default), or `''` when none is
 */
export function errorsHash({scope, event, pollster, from, to, measure}) {
  const parts = [];
  const add = (key, value) => {
    if (value !== null && value !== undefined) {
      parts.push(`${key}=${encodeURIComponent(value)}`);
    }
  };
  add(SCOPE_KEY, scope);
  add(EVENT_KEY, event);
  add(POLLSTER_KEY, pollster);
  add(FROM_KEY, from);
  add(TO_KEY, to);
  if (measure && measure !== DEFAULT_MEASURE) {
    add(MEASURE_KEY, measure);
  }
  return parts.length ? `#${parts.join('&')}` : '';
}
