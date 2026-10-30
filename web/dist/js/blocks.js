// Block view of the poll average (series, polls, projection and official result) and the state of its controls in the URL hash.
// Pure functions without DOM nor ECharts, shared by the Promedios page and the parity test.
// A block map with `REST_BLOCK` (Otros) gets that block as the rest: 100 minus the other blocks, whatever
// parties the map lists under it; a map without it (runs published with four blocks) is summed as it comes.
import {DEFAULT_WINDOW} from './range.js';

const WINDOW_KEY = 'ventana';
const VIEW_KEY = 'vista';
const SELECTED_KEY = 'sel';
// Layers of the average drawn: the poll points, the average (lines and bands) or both, the default, which the hash
// omits (`mostrar=sondeos`, `mostrar=promedio`)
const LAYERS_KEY = 'mostrar';
const LAYERS = ['sondeos', 'promedio', 'ambos'];
const DEFAULT_LAYERS = 'ambos';
/** Name of the block that holds the rest of the vote (what Izquierda and Derecha leave out). */
export const REST_BLOCK = 'Otros';
const PARTIES_VIEW = 'partidos';
const BLOCKS_VIEW = 'bloques';

/** Whether a value of a part is a number (the parts write missing values as null). */
function present(value) {
  return value !== null && value !== undefined;
}

/**
 * Value of the rest block: `total` minus the other blocks, only when there are others and all of them have a value.
 *
 * @param {Array<number|null|undefined>} others values of the blocks other than the rest block
 * @param {number} [total=100] what the blocks add up to (100 for shares, the chamber for seats)
 * @returns {number|null} the complement to `total`, null when any value is missing
 */
function restOf(others, total = 100) {
  return others.length && others.every(present) ? total - others.reduce((sum, value) => sum + value, 0) : null;
}

/**
 * Poll average of every block: the sum of the means of its parties and a band that adds their half-widths in
 * quadrature (independent draws, as the notebook does: the blocks add the variances of their parties).
 *
 * Per date and block, `mean` is the sum of the non-null means of its parties (null when none has one) and
 * `lo`/`hi` are `mean ∓ sqrt(Σ((hi_i − lo_i) / 2)²)` over the parties with a band on that date. The `REST_BLOCK`,
 * if the map has one, is `100 − Σ` of the other blocks' means (null on a date when any of them is null) with
 * a half-width of `sqrt(Σ half²)` over them.
 *
 * @param {{dates: string[], mean: object, lo: object, hi: object}} series `data` of the series part
 * @param {Object<string, string[]>} blocks block → party names (`meta.bmaps.blocks`)
 * @returns {{dates: string[], parties: string[], mean: object, lo: object, hi: object}} the block series, with
 *   the block names as `parties`, ready for `renderSeries`
 */
export function blockSeries(series, blocks) {
  const names = Object.keys(blocks);
  const out = {dates: series.dates, parties: names, mean: {}, lo: {}, hi: {}};
  const others = names.filter((block) => block !== REST_BLOCK);
  for (const block of others) {
    const members = blocks[block];
    out.mean[block] = [];
    out.lo[block] = [];
    out.hi[block] = [];
    series.dates.forEach((date, i) => {
      let mean = null;
      let variance = 0;
      for (const name of members) {
        const value = (series.mean[name] || [])[i];
        const lo = (series.lo[name] || [])[i];
        const hi = (series.hi[name] || [])[i];
        if (present(value)) {
          mean = (mean ?? 0) + value;
        }
        if (present(lo) && present(hi)) {
          variance += ((hi - lo) / 2) ** 2;
        }
      }
      const half = Math.sqrt(variance);
      out.mean[block].push(mean);
      out.lo[block].push(mean === null ? null : mean - half);
      out.hi[block].push(mean === null ? null : mean + half);
    });
  }
  if (names.includes(REST_BLOCK)) {
    out.mean[REST_BLOCK] = [];
    out.lo[REST_BLOCK] = [];
    out.hi[REST_BLOCK] = [];
    series.dates.forEach((date, i) => {
      const mean = restOf(others.map((block) => out.mean[block][i]));
      const half = Math.sqrt(others.reduce((total, block) => total + ((out.hi[block][i] ?? 0) - (out.mean[block][i] ?? 0)) ** 2, 0));
      out.mean[REST_BLOCK].push(mean);
      out.lo[REST_BLOCK].push(mean === null ? null : mean - half);
      out.hi[REST_BLOCK].push(mean === null ? null : mean + half);
    });
  }
  return out;
}

/**
 * Polls by block: each poll keeps `date`, `pollster`, `pollster_id`, `name` and `sample_size` (the box of the
 * poll) and gets, per block, the sum of the values of its parties present in the poll; a poll with none of the
 * parties of a block has no key for it. The `REST_BLOCK` is `100 −` the other blocks of the poll, only when
 * the poll has all of them.
 *
 * @param {{polls: object[]}} polls the polls of the page (`chart_polls`)
 * @param {Object<string, string[]>} blocks block → party names (`meta.bmaps.blocks`)
 * @returns {{parties: string[], polls: object[]}} the block polls, with the block names as `parties`
 */
export function blockPolls(polls, blocks) {
  const names = Object.keys(blocks);
  return {
    parties: names,
    polls: (polls.polls || []).map((poll) => {
      const row = {
        date: poll.date, pollster: poll.pollster, pollster_id: poll.pollster_id, name: poll.name,
        sample_size: poll.sample_size,
      };
      for (const block of names.filter((name) => name !== REST_BLOCK)) {
        const values = blocks[block].map((name) => poll[name]).filter(present);
        if (values.length) {
          row[block] = values.reduce((total, value) => total + value, 0);
        }
      }
      if (names.includes(REST_BLOCK)) {
        const rest = restOf(names.filter((name) => name !== REST_BLOCK).map((name) => row[name]));
        if (rest !== null) {
          row[REST_BLOCK] = rest;
        }
      }
      return row;
    }),
  };
}

/**
 * Published runs by block, for the evolution of the publications: each run keeps its fields (`run_id`, `run_at`,
 * `as_of`, `backfill`, `n_polls`...) and gets, in `nowcast.parties`, one row `{name, [field]}` per block with the
 * sum of the `field` of its parties present in the run's headline; a block with none of them has no row. The
 * `REST_BLOCK` is `total −` the other blocks, only when every one of them has a row. The blocks follow the current
 * map: a run from before a party rename (SUMAR → FA) has no row for the new name, so that share falls into the
 * rest block; the histories are republished with the current line-up, which keeps this latent.
 *
 * @param {object[]} runs runs of the history part (`headline` of each, with `nowcast.parties`)
 * @param {Object<string, string[]>} blocks block → party names (`meta.bmaps.blocks`)
 * @param {string} [field='pct'] party field summed (`pct` or `seats`)
 * @param {number} [total=100] what the blocks add up to (100 for shares, the chamber for seats)
 * @returns {object[]} the runs with the block rows as `nowcast.parties`, in the order of `blocks`
 */
export function blockRuns(runs, blocks, field = 'pct', total = 100) {
  const names = Object.keys(blocks);
  const others = names.filter((name) => name !== REST_BLOCK);
  return runs.map((run) => {
    const parties = ((run.nowcast || {}).parties || []);
    const valueOf = (party) => {
      const row = parties.find((item) => item.name === party);
      return row ? row[field] : undefined;
    };
    const rows = [];
    const sums = {};
    for (const block of others) {
      const values = blocks[block].map(valueOf).filter(present);
      if (values.length) {
        sums[block] = values.reduce((sum, value) => sum + value, 0);
        rows.push({name: block, [field]: sums[block]});
      }
    }
    if (names.includes(REST_BLOCK)) {
      const rest = restOf(others.map((name) => sums[name]), total);
      if (rest !== null) {
        rows.push({name: REST_BLOCK, [field]: rest});
      }
    }
    return {...run, nowcast: {...run.nowcast, parties: rows}};
  });
}

/**
 * Projection of the blocks as `renderSeries` reads that of the parties: the published one of the simulator
 * (`groups.blocks`), not recomputed.
 *
 * @param {object|null} projection `data` of the projection part
 * @returns {{dates: string[], groups: {parties: object}}|null} the block projection, null without one
 */
export function blockProjection(projection) {
  const groups = projection && projection.groups && projection.groups.blocks;
  return groups ? {dates: projection.dates, groups: {parties: groups}} : null;
}

/**
 * Official result by block: the sum of the shares of the parties of each block present in `values`; a block
 * with none of them has no key. The `REST_BLOCK` is `100 −` the other blocks, only when all of them have a value.
 *
 * @param {{date: string, values: Object<string, number>}} result the result marked on the series chart
 * @param {Object<string, string[]>} blocks block → party names
 * @returns {{date: string, values: Object<string, number>}} the result with the block names as keys
 */
export function blockResult(result, blocks) {
  const values = {};
  for (const [block, members] of Object.entries(blocks)) {
    const found = members.map((name) => result.values[name]).filter(present);
    if (block !== REST_BLOCK && found.length) {
      values[block] = found.reduce((total, value) => total + value, 0);
    }
  }
  if (REST_BLOCK in blocks) {
    const rest = restOf(Object.keys(blocks).filter((block) => block !== REST_BLOCK).map((block) => values[block]));
    if (rest !== null) {
      values[REST_BLOCK] = rest;
    }
  }
  return {date: result.date, values};
}

/**
 * State of the average controls read from the URL hash (`#ventana=…&vista=…&sel=A,B&mostrar=sondeos`).
 *
 * @param {string} hash `location.hash`, with or without the leading `#`
 * @param {{parties: string[], blocks: string[], windows: string[]}} options names of the parties view, of
 *   the blocks view (empty when the scope has no blocks) and the valid windows
 * @returns {{window: string, view: string, selected: string[]|null, layers: string}} `window` in `windows`
 *   (otherwise `DEFAULT_WINDOW`, `all`), `view` `partidos` or `bloques` (only with blocks), `selected` the names
 *   of the hash in the list of the view, in its order (`[]` for an explicit empty `sel=`; null, the default
 *   selection, when the key is absent or no name is valid), `layers` what is drawn: `sondeos` (the poll points
 *   alone), `promedio` (the average alone) or `ambos` (both, the default and the fallback of any other value)
 */
export function parseSeriesHash(hash, {parties, blocks, windows}) {
  const params = new Map();
  for (const entry of (hash || '').replace(/^#/, '').split('&')) {
    const at = entry.indexOf('=');
    if (at > 0) {
      params.set(entry.slice(0, at), entry.slice(at + 1));
    }
  }
  const choice = params.get(WINDOW_KEY);
  const view = params.get(VIEW_KEY) === BLOCKS_VIEW && blocks.length ? BLOCKS_VIEW : PARTIES_VIEW;
  const raw = params.get(SELECTED_KEY);
  const names = new Set((raw || '').split(',').map((name) => {
    try {
      return decodeURIComponent(name);
    } catch {
      return null;
    }
  }));
  const selected = (view === BLOCKS_VIEW ? blocks : parties).filter((name) => names.has(name));
  return {
    window: windows.includes(choice) ? choice : DEFAULT_WINDOW,
    view,
    selected: selected.length || raw === '' ? selected : null,
    layers: LAYERS.includes(params.get(LAYERS_KEY)) ? params.get(LAYERS_KEY) : DEFAULT_LAYERS,
  };
}

/**
 * URL hash of the state of the average controls, the inverse of `parseSeriesHash`.
 *
 * @param {{window: string, view: string, selected: string[]|null, layers: string}} state the controls state
 * @returns {string} `#ventana=…&vista=…`, plus `&sel=A,B` (each name URI-encoded) when `selected` is a list
 *   (`&sel=` for an empty one) and `&mostrar=sondeos` or `&mostrar=promedio` when one layer alone is shown
 */
export function seriesHash({window, view, selected, layers = DEFAULT_LAYERS}) {
  let hash = `#${WINDOW_KEY}=${encodeURIComponent(window)}&${VIEW_KEY}=${encodeURIComponent(view)}`;
  if (selected) {
    hash += `&${SELECTED_KEY}=${selected.map(encodeURIComponent).join(',')}`;
  }
  if (layers !== DEFAULT_LAYERS) {
    hash += `&${LAYERS_KEY}=${encodeURIComponent(layers)}`;
  }
  return hash;
}
