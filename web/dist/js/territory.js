// Pure logic of "Resultados por territorio" of Escaños: rows by province or by community, in seats or in vote,
// winners, cell tints and sorting. The page script only reads the controls, calls these functions and writes the DOM.
import {fmtRange} from './format.js';

/** Highest alpha of a cell tint. */
export const MAX_ALPHA = 0.18;
const PERCENT_TOTAL = '100 %';

const isNumber = (x) => typeof x === 'number' && !Number.isNaN(x);

/**
 * Whether the regions can be grouped by community: all carry `community` and `votes`, and there are at least two
 * distinct communities.
 *
 * @param {Array} regions `districts.regions`
 * @returns {boolean} true when the community level makes sense
 */
export function canGroup(regions) {
  return regions.length > 0
    && regions.every((region) => region.community && isNumber(region.votes))
    && new Set(regions.map((region) => region.community.code)).size > 1;
}

/**
 * Province rows: one per region of `districts`, in its order.
 *
 * @param {object} districts the districts part
 * @param {object} scenario the scenario part
 * @param {string[]} parties columns
 * @param {string} mode `esc` or `pct`
 * @returns {Array} rows with `id`, `name`, `values`, `total` and `ranges`
 */
function provinceRows(districts, scenario, parties, mode) {
  const seats = new Map((scenario.rows || []).map((row) => [row.region_id, row.seats]));
  const index = new Map((scenario.parties || []).map((name, i) => [name, i]));
  const cells = new Map((districts.rows || []).map((row) => [`${row.region_id}|${row.name}`, row]));
  return (districts.regions || []).map((region) => {
    const values = {};
    const ranges = {};
    for (const name of parties) {
      const cell = cells.get(`${region.id}|${name}`);
      if (mode === 'pct') {
        values[name] = cell && isNumber(cell.pct) ? cell.pct : null;
      } else {
        const row = seats.get(region.id);
        values[name] = row && index.has(name) ? row[index.get(name)] : null;
        ranges[name] = cell && isNumber(cell.seats_lo) && isNumber(cell.seats_hi)
          ? fmtRange(cell.seats_lo, cell.seats_hi, 0) : null;
      }
    }
    return {
      id: region.id, name: region.name, values, ranges, total: mode === 'pct' ? PERCENT_TOTAL : region.seats,
      _weight: region.votes, _code: region.community ? region.community.code : null,
      _community: region.community ? region.community.name : null,
    };
  });
}

/**
 * Group province rows by community: seats add up, the vote is the mean weighted by the votes of each province
 * (only those with data).
 *
 * @param {Array} rows province rows
 * @param {string[]} parties columns
 * @param {string} mode `esc` or `pct`
 * @returns {Array} one row per community, in order of first appearance, with `id: null` and no ranges
 */
export function groupByCommunity(rows, parties, mode) {
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row._code)) {
      groups.set(row._code, {name: row._community, members: []});
    }
    groups.get(row._code).members.push(row);
  }
  return [...groups.values()].map(({name, members}) => {
    const values = {};
    for (const party of parties) {
      const known = members.filter((row) => isNumber(row.values[party]));
      if (!known.length) {
        values[party] = null;
      } else if (mode === 'pct') {
        const weight = known.reduce((sum, row) => sum + row._weight, 0);
        values[party] = weight > 0
          ? known.reduce((sum, row) => sum + row.values[party] * row._weight, 0) / weight
          : known.reduce((sum, row) => sum + row.values[party], 0) / known.length;
      } else {
        values[party] = known.reduce((sum, row) => sum + row.values[party], 0);
      }
    }
    const total = mode === 'pct' ? PERCENT_TOTAL : members.reduce((sum, row) => sum + (row.total || 0), 0);
    return {id: null, name, values, total, ranges: {}};
  });
}

/**
 * Rows, columns and total row of the table.
 *
 * @param {object} districts the districts part (`parties`, `regions`, `rows`)
 * @param {object} scenario the scenario part (`parties`, `rows`; `region_id` 0 is the whole scope)
 * @param {object} summary the summary part (`parties` with `pct`)
 * @param {{level: string, mode: string}} [options] `prov` or `ccaa` (falls back to `prov` without communities);
 *   `esc` or `pct`
 * @returns {{parties: string[], rows: Array, total: object}} columns, rows and the TOTAL row
 */
export function territoryRows(districts, scenario, summary, {level = 'prov', mode = 'esc'} = {}) {
  const scenarioRows = scenario.rows || [];
  const regionRows = scenarioRows.filter((row) => row.region_id !== 0);
  const scenarioParties = scenario.parties || [];
  const whole = scenarioRows.find((row) => row.region_id === 0);
  const seatTotals = {};
  scenarioParties.forEach((name, i) => {
    seatTotals[name] = whole ? whole.seats[i] : regionRows.reduce((sum, row) => sum + (row.seats[i] || 0), 0);
  });
  const voteTotals = Object.fromEntries((summary.parties || []).map((row) => [row.name, row.pct]));
  const rank = (name) => (mode === 'pct' ? voteTotals[name] : seatTotals[name]);
  const parties = mode === 'pct'
    ? [...(districts.parties || [])]
    : scenarioParties.filter((name, i) => regionRows.some((row) => (row.seats[i] || 0) > 0));
  parties.sort((a, b) => ((rank(b) ?? 0) - (rank(a) ?? 0)) || a.localeCompare(b, 'es'));

  let rows = provinceRows(districts, scenario, parties, mode);
  if (level === 'ccaa' && canGroup(districts.regions || [])) {
    rows = groupByCommunity(rows, parties, mode);
  } else {
    rows = rows.map(({id, name, values, total, ranges}) => ({id, name, values, total, ranges}));
  }
  const values = {};
  for (const name of parties) {
    values[name] = mode === 'pct' ? voteTotals[name] ?? null : seatTotals[name];
  }
  const total = mode === 'pct' ? PERCENT_TOTAL : parties.reduce((sum, name) => sum + (values[name] || 0), 0);
  return {parties, rows, total: {values, total}};
}

/**
 * Parties with the highest value of a row.
 *
 * @param {Object<string, (number|null)>} values value by party name
 * @returns {string[]} the names holding the maximum, ties included; empty when no value is above zero
 */
export function winnerOf(values) {
  let best = 0;
  let names = [];
  for (const [name, value] of Object.entries(values)) {
    if (!isNumber(value) || value <= 0) {
      continue;
    }
    if (value > best) {
      best = value;
      names = [name];
    } else if (value === best) {
      names.push(name);
    }
  }
  return names;
}

/**
 * Range of the positive values of a column.
 *
 * @param {Array} rows territory rows
 * @param {string} name party name
 * @returns {{min: (number|undefined), max: (number|undefined)}} smallest and largest value above zero
 */
export function columnRange(rows, name) {
  const values = rows.map((row) => row.values[name]).filter((value) => isNumber(value) && value > 0);
  return values.length ? {min: Math.min(...values), max: Math.max(...values)} : {min: undefined, max: undefined};
}

/**
 * Alpha of a cell tint, growing linearly from 0 at the column minimum to `MAX_ALPHA` at its maximum.
 *
 * @param {(number|null)} value cell value
 * @param {number} min column minimum
 * @param {number} max column maximum
 * @returns {number} 0 without a positive value; `MAX_ALPHA` when `max === min`
 */
export function tintAlpha(value, min, max) {
  if (!isNumber(value) || value <= 0) {
    return 0;
  }
  return max === min ? MAX_ALPHA : ((value - min) / (max - min)) * MAX_ALPHA;
}

/**
 * Sort territory rows.
 *
 * @param {Array} rows territory rows
 * @param {string} key `name`, `total` or a party name
 * @param {boolean} asc ascending when true
 * @returns {Array} a sorted copy; missing values last in both directions, ties by name
 */
export function sortTerritory(rows, key, asc) {
  const sign = asc ? 1 : -1;
  const valueOf = (row) => {
    const value = key === 'total' ? row.total : row.values[key];
    return isNumber(value) ? value : null;
  };
  return [...rows].sort((a, b) => {
    if (key === 'name') {
      return sign * a.name.localeCompare(b.name, 'es');
    }
    const [x, y] = [valueOf(a), valueOf(b)];
    if (x === null || y === null) {
      return x === y ? a.name.localeCompare(b.name, 'es') : (x === null ? 1 : -1);
    }
    return sign * (x - y) || a.name.localeCompare(b.name, 'es');
  });
}
