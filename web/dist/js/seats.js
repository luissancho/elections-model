// Seat simulations: columns, coalition sums, quantiles, histogram bins and the coalition hash.
// Pure functions without DOM, shared by the seats page and the parity test.

const COALITION_KEY = 'coalition=';

/**
 * Seats of one party in every simulation.
 *
 * @param {{parties: string[], seats: number[][]}} dist the dist part (one row per simulation)
 * @param {string} name party name
 * @returns {number[]} the column, `[]` when the party is not in `dist.parties`
 */
export function column(dist, name) {
  const index = dist.parties.indexOf(name);
  return index < 0 ? [] : dist.seats.map((row) => row[index]);
}

/**
 * Seats of a coalition in every simulation: the sum of the columns of the parties present in `dist`.
 *
 * @param {{parties: string[], seats: number[][]}} dist the dist part
 * @param {string[]} names party names (unknown ones are ignored)
 * @returns {number[]} the sums, `[]` when none of the parties is present
 */
export function coalitionSeats(dist, names) {
  const indexes = names.map((name) => dist.parties.indexOf(name)).filter((index) => index >= 0);
  if (!indexes.length) {
    return [];
  }
  return dist.seats.map((row) => indexes.reduce((total, index) => total + row[index], 0));
}

/**
 * Quantile `q` of sorted values with the rule of `Stat.quantile` (unit weights): the first value whose
 * cumulative count reaches `q · n`, averaged with the next one when the target falls exactly on it.
 *
 * @param {number[]} sorted values in ascending order (not empty)
 * @param {number} q probability in [0, 1]
 * @returns {number} the quantile
 */
export function quantile(sorted, q) {
  const n = sorted.length;
  const tgt = q * n;
  const i = Math.min(Math.max(Math.ceil(tgt) - 1, 0), n - 1);
  if (Math.abs(tgt - (i + 1)) < 1e-9 && i < n - 1) {
    return (sorted[i] + sorted[i + 1]) / 2;
  }
  return sorted[i];
}

/**
 * Median, central interval, range and probability of majority of seat values.
 *
 * @param {number[]} values seats per simulation
 * @param {number} majority seats of the absolute majority
 * @param {number} alpha the interval leaves out `alpha / 2` on each side
 * @returns {{n: number, median: number, lo: number, hi: number, min: number, max: number,
 *   pMajority: number}|null} the summary, null without values
 */
export function summarize(values, majority, alpha = 0.05) {
  if (!values.length) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const reached = sorted.filter((value) => value >= majority).length;
  return {
    n,
    median: quantile(sorted, 0.5),
    lo: quantile(sorted, alpha / 2),
    hi: quantile(sorted, 1 - alpha / 2),
    min: sorted[0],
    max: sorted[n - 1],
    pMajority: reached / n,
  };
}

/** Smallest and largest value, without spreading large arrays into `Math.min`. */
function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }
  return [min, max];
}

/**
 * Integer bin width that keeps the histogram within `maxBins` bins.
 *
 * @param {number[]} values seats per simulation (not empty)
 * @param {number} maxBins largest number of bins
 * @returns {number} the width, at least 1
 */
export function binWidth(values, maxBins = 40) {
  const [min, max] = extent(values);
  return Math.max(1, Math.ceil((max - min + 1) / maxBins));
}

/**
 * Counts of `values` in the bins `[start, start + width)`, from `floor(min / width) · width` up to the
 * bin holding the largest value, with no gaps (empty bins count 0).
 *
 * @param {number[]} values seats per simulation
 * @param {number} width bin width
 * @returns {{starts: number[], counts: number[]}} bin starts and counts
 */
export function histogram(values, width) {
  if (!values.length) {
    return {starts: [], counts: []};
  }
  const [min, max] = extent(values);
  const first = Math.floor(min / width) * width;
  const size = Math.floor((max - first) / width) + 1;
  const starts = Array.from({length: size}, (_, i) => first + i * width);
  const counts = new Array(size).fill(0);
  for (const value of values) {
    counts[Math.floor((value - first) / width)] += 1;
  }
  return {starts, counts};
}

/**
 * Coalition party names of a URL fragment such as `#coalition=PP,VOX`.
 *
 * @param {string} hash the fragment, with or without the leading `#`
 * @param {string[]} parties known party names
 * @returns {string[]|null} the known names in the order of `parties`, null without `coalition=`
 */
export function parseCoalition(hash, parties) {
  const text = (hash || '').replace(/^#/, '');
  const entry = text.split('&').find((part) => part.startsWith(COALITION_KEY));
  if (entry === undefined) {
    return null;
  }
  const names = new Set(entry.slice(COALITION_KEY.length).split(',').map((name) => {
    try {
      return decodeURIComponent(name);
    } catch {
      return null;
    }
  }));
  return parties.filter((name) => names.has(name));
}

/**
 * URL fragment of a coalition, the inverse of `parseCoalition`.
 *
 * @param {string[]} names party names
 * @returns {string} `#coalition=…`, or `''` without names
 */
export function coalitionHash(names) {
  return names.length ? `#${COALITION_KEY}${names.map(encodeURIComponent).join(',')}` : '';
}
