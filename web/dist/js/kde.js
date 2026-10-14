// Seat densities for the ridge chart of the seats page: integer counts, Scott's bandwidth, a gaussian kernel on
// a seat grid and one curve per party scaled to its own peak. Pure functions without DOM, shared with the parity
// test, which checks them against `scipy.stats.gaussian_kde`.
import {column, summarize} from './seats.js';

/** Bandwidth floor, in seats: a party with (almost) the same seats in every simulation still draws a curve. */
export const MIN_BANDWIDTH = 0.5;

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
 * Evenly spaced grid from 0 to `xMax` (inclusive when it falls on a step).
 *
 * @param {number} xMax last seat of the axis
 * @param {number} step distance between points
 * @returns {number[]} the grid
 */
export function seatGrid(xMax, step) {
  const out = [];
  for (let x = 0; x <= xMax + 1e-9; x += step) {
    out.push(Math.round(x * 1e6) / 1e6);
  }
  return out;
}

/**
 * One ridge per party: its density on the shared grid, scaled so that its peak is 1, with the quantiles of
 * `summarize` and the share of simulations at each number of seats (for the tooltip).
 *
 * @param {{parties: string[], seats: number[][]}} dist the dist part
 * @param {string[]} names parties to draw, in row order (those missing from `dist` are skipped)
 * @param {object} opts `xMax` (last seat of the axis), `step` (grid step, 0.5 by default), `majority` (seats of
 *   the absolute majority, for `summarize`)
 * @returns {{grid: number[], rows: object[]}} the grid and, per party, `name`, `y` (scaled density per grid
 *   point), `peak` (the raw peak density), `bw`, `median`, `lo`, `hi`, `n` and `pmf` (`{seats: share}`)
 */
export function ridgeData(dist, names, {xMax, step = 0.5, majority = 0} = {}) {
  const grid = seatGrid(xMax, step);
  const rows = [];
  for (const name of names) {
    const values = column(dist, name);
    if (!values.length) {
      continue;
    }
    const bw = scottBandwidth(values);
    const raw = density(values, grid, bw);
    const peak = Math.max(...raw);
    const stats = summarize(values, majority);
    const pmf = {};
    for (const [value, count] of counts(values)) {
      pmf[value] = count / values.length;
    }
    rows.push({
      name,
      y: raw.map((value) => (peak > 0 ? value / peak : 0)),
      peak,
      bw,
      median: stats.median,
      lo: stats.lo,
      hi: stats.hi,
      n: stats.n,
      pmf,
    });
  }
  return {grid, rows};
}
