// Seat densities for the ridge chart of the seats page: integer counts, Scott's bandwidth, a gaussian kernel on
// a seat grid and one curve per row (a party or a block) scaled to its own peak, each over its own seat range.
// Pure functions without DOM, shared with the parity test, which checks them against `scipy.stats.gaussian_kde`.
import {coalitionSeats, column, summarize} from './seats.js';

/** Bandwidth floor, in seats: a row with (almost) the same seats in every simulation still draws a curve. */
export const MIN_BANDWIDTH = 0.5;

/** Points of the grid of each row. */
export const GRID_POINTS = 200;

/**
 * How many simulations give each number of seats.
 *
 * @param {number[]} values seats per simulation
 * @returns {Map<number, number>} count per distinct value, in ascending order of value
 */
export function counts(values) {
  const out = new Map();
  for (const value of values) {
    out.set(value, (out.get(value) || 0) + 1);
  }
  return new Map([...out.entries()].sort((a, b) => a[0] - b[0]));
}

/**
 * Scott's bandwidth, as `scipy.stats.gaussian_kde` uses it: `n^(-1/5)` times the sample standard deviation
 * (with `n - 1` in the denominator), never below `MIN_BANDWIDTH`.
 *
 * @param {number[]} values seats per simulation (not empty)
 * @returns {number} the bandwidth, in seats
 */
export function scottBandwidth(values) {
  const n = values.length;
  if (n < 2) {
    return MIN_BANDWIDTH;
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1);
  return Math.max(MIN_BANDWIDTH, n ** (-1 / 5) * Math.sqrt(variance));
}

/**
 * Gaussian kernel density of the values on a grid.
 *
 * @param {number[]} values seats per simulation (not empty)
 * @param {number[]} grid points where the density is evaluated
 * @param {number} bw bandwidth, in seats
 * @returns {number[]} the density at each grid point (integrates to 1 over the real line)
 */
export function density(values, grid, bw) {
  const n = values.length;
  const scale = 1 / (n * bw * Math.sqrt(2 * Math.PI));
  const byValue = [...counts(values).entries()];
  return grid.map((x) => {
    let total = 0;
    for (const [value, count] of byValue) {
      const z = (x - value) / bw;
      total += count * Math.exp(-0.5 * z * z);
    }
    return total * scale;
  });
}

/**
 * Evenly spaced grid from `x0` to `x1`, both included.
 *
 * @param {number} x0 first point
 * @param {number} x1 last point (greater than `x0`)
 * @param {number} points number of points (at least 2)
 * @returns {number[]} the grid
 */
export function seatGrid(x0, x1, points = GRID_POINTS) {
  const n = Math.max(2, points);
  const step = (x1 - x0) / (n - 1);
  return Array.from({length: n}, (_, i) => Math.round((x0 + i * step) * 1e6) / 1e6);
}

/**
 * Seat range drawn for a row: from one seat below the smallest simulated value (never below 0) to one above
 * the largest, and at least `minWidth` seats wide so that a narrow distribution still gets some room.
 *
 * @param {{min: number, max: number}} stats the summary of the values
 * @param {number} minWidth smallest width of the range, in seats
 * @returns {number[]} `[x0, x1]`, integers
 */
export function seatRange(stats, minWidth = 6) {
  let x0 = Math.max(0, Math.floor(stats.min) - 1);
  let x1 = Math.ceil(stats.max) + 1;
  if (x1 - x0 < minWidth) {
    x1 = x0 + minWidth;
  }
  return [x0, x1];
}

/**
 * One ridge per series: the density of its values on its own seat range, scaled so that its peak is 1, with
 * the quantiles of `summarize` and the share of simulations at each number of seats (for the tooltip).
 *
 * @param {{name: string, values: number[]}[]} series rows to draw, in order (those without values are skipped)
 * @param {object} opts `majority` (seats of the absolute majority, for `summarize`), `points` (grid points
 *   per row, `GRID_POINTS` by default), `minWidth` (smallest range width, see `seatRange`) and `range`
 *   (`[x0, x1]` shared by every row; without it each row gets its own `seatRange`)
 * @returns {object[]} per row: `name`, `x0`, `x1`, `grid`, `y` (scaled density per grid point), `peak` (the
 *   raw peak density), `bw`, `median`, `lo`, `hi`, `min`, `max`, `n` and `pmf` (`{seats: share}`)
 */
export function ridgeRows(series, {majority = 0, points = GRID_POINTS, minWidth = 6, range = null} = {}) {
  const rows = [];
  for (const {name, values} of series) {
    if (!values || !values.length) {
      continue;
    }
    const stats = summarize(values, majority);
    const [x0, x1] = range ? range : seatRange(stats, minWidth);
    const grid = seatGrid(x0, x1, points);
    const bw = scottBandwidth(values);
    const raw = density(values, grid, bw);
    const peak = Math.max(...raw);
    const pmf = {};
    for (const [value, count] of counts(values)) {
      pmf[value] = count / values.length;
    }
    rows.push({
      name,
      x0,
      x1,
      grid,
      y: raw.map((value) => (peak > 0 ? value / peak : 0)),
      peak,
      bw,
      median: stats.median,
      lo: stats.lo,
      hi: stats.hi,
      min: stats.min,
      max: stats.max,
      n: stats.n,
      pmf,
    });
  }
  return rows;
}

/**
 * Ridges of parties: one series per party present in `dist`, from its column.
 *
 * @param {{parties: string[], seats: number[][]}} dist the dist part
 * @param {string[]} names parties to draw, in row order (those missing from `dist` are skipped)
 * @param {object} opts options of `ridgeRows`
 * @returns {object[]} the rows of `ridgeRows`
 */
export function ridgeData(dist, names, opts = {}) {
  return ridgeRows(names.map((name) => ({name, values: column(dist, name)})), opts);
}

/**
 * Ridges of blocks: one series per block, the sum of the seats of its parties in every simulation.
 *
 * @param {{parties: string[], seats: number[][]}} dist the dist part
 * @param {{name: string, parties: string[]}[]} blocks blocks to draw, in row order, with their parties
 * @param {object} opts options of `ridgeRows`
 * @returns {object[]} the rows of `ridgeRows` (a block without any party in `dist` is skipped)
 */
export function ridgeBlocks(dist, blocks, opts = {}) {
  return ridgeRows(blocks.map(({name, parties}) => ({name, values: coalitionSeats(dist, parties)})), opts);
}
