// Block view of the poll average and the state of its controls in the URL hash.
// Pure functions without DOM nor ECharts, shared by the Promedios page and the parity test.
import {DEFAULT_WINDOW} from './range.js';

const WINDOW_KEY = 'ventana';
const VIEW_KEY = 'vista';
const SELECTED_KEY = 'sel';
const PARTIES_VIEW = 'partidos';
const BLOCKS_VIEW = 'bloques';

/** Whether a value of a part is a number (the parts write missing values as null). */
function present(value) {
  return value !== null && value !== undefined;
}

/**
 * Poll average of every block: the sum of the means of its parties and a band that adds their half-widths in
 * quadrature (independent draws, as the notebook does: the blocks add the variances of their parties).
 *
 * Per date and block, `mean` is the sum of the non-null means of its parties (null when none has one) and
 * `lo`/`hi` are `mean ∓ sqrt(Σ((hi_i − lo_i) / 2)²)` over the parties with a band on that date.
 *
 * @param {{dates: string[], mean: object, lo: object, hi: object}} series `data` of the series part
 * @param {Object<string, string[]>} blocks block → party names (`meta.bmaps.blocks`)
 * @returns {{dates: string[], parties: string[], mean: object, lo: object, hi: object}} the block series, with
 *   the block names as `parties`, ready for `renderSeries`
 */
export function blockSeries(series, blocks) {
  const names = Object.keys(blocks);
  const out = {dates: series.dates, parties: names, mean: {}, lo: {}, hi: {}};
  for (const block of names) {
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
  return out;
}

/**
 * Polls by block: each poll keeps `date`, `pollster` and `pollster_id` and gets, per block, the sum of the
 * values of its parties present in the poll; a poll with none of the parties of a block has no key for it.
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
      const row = {date: poll.date, pollster: poll.pollster, pollster_id: poll.pollster_id};
      for (const block of names) {
        const values = blocks[block].map((name) => poll[name]).filter(present);
        if (values.length) {
          row[block] = values.reduce((total, value) => total + value, 0);
        }
      }
      return row;
    }),
  };
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
 * State of the average controls read from the URL hash (`#ventana=…&vista=…&sel=A,B`).
 *
 * @param {string} hash `location.hash`, with or without the leading `#`
 * @param {{parties: string[], blocks: string[], windows: string[]}} options names of the parties view, of
 *   the blocks view (empty when the scope has no blocks) and the valid windows
 * @returns {{window: string, view: string, selected: string[]|null}} `window` in `windows` (otherwise
 *   `DEFAULT_WINDOW`, `all`), `view` `partidos` or `bloques` (only with blocks), `selected` the names of the
 *   hash in the list of the view, in its order (`[]` for an explicit empty `sel=`; null, the default selection,
 *   when the key is absent or no name is valid)
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
  };
}

/**
 * URL hash of the state of the average controls, the inverse of `parseSeriesHash`.
 *
 * @param {{window: string, view: string, selected: string[]|null}} state the controls state
 * @returns {string} `#ventana=…&vista=…`, plus `&sel=A,B` (each name URI-encoded) when `selected` is a list
 *   (`&sel=` for an empty one)
 */
export function seriesHash({window, view, selected}) {
  let hash = `#${WINDOW_KEY}=${encodeURIComponent(window)}&${VIEW_KEY}=${encodeURIComponent(view)}`;
  if (selected) {
    hash += `&${SELECTED_KEY}=${selected.map(encodeURIComponent).join(',')}`;
  }
  return hash;
}
