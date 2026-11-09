// Tooltip boxes of the poll average: one poll (name, date, sample, rating of the house and its values in
// descending order), one
// point of the average or of its projection and the official result of a past election. Pure functions, without
// ECharts or the DOM, testable in node.
import {REST_BLOCK} from './blocks.js';
import {fmtDate, fmtInt, fmtNum, fmtPct, fmtRange} from './format.js';

const DASH = '–';
const HTML_ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};

/**
 * Escape text for the HTML of a tooltip.
 *
 * @param {*} text any value, converted to a string
 * @returns {string} the text with `& < > " '` escaped
 */
export function escapeText(text) {
  return String(text).replace(/[&<>"']/g, (char) => HTML_ESCAPES[char]);
}

/**
 * Round colour marker of a tooltip line, as an inline `<span>` (the one of ECharts needs the library).
 *
 * @param {string} color CSS colour
 * @returns {string} the marker HTML
 */
export function tooltipMarker(color) {
  return '<span style="display:inline-block;margin-right:4px;border-radius:10px;width:10px;height:10px;'
    + `background-color:${escapeText(color)};"></span>`;
}

function present(value) {
  return value !== null && value !== undefined && !Number.isNaN(value);
}

/**
 * Lines of the box of one poll: its name in bold, the date, the sample and, when the record carries the field,
 * the rating of the house (0-100 at one decimal, as the badge of the polls table; a dash for the houses without
 * one, aggregators and online panels; Luis, 2026-10-28), and one line per name of `parties` with a value in the
 * poll, in descending order of value, with its colour marker and `{name} {value}`. The polls of the runs and of
 * the elections always carry `rating`; the live example of the method (`weights.js`) passes none and names the
 * rating factor itself.
 *
 * @param {object} poll poll record (`name` or else `pollster`, `date`, `sample_size`, `rating` and one value per
 *   name)
 * @param {string[]} parties names of the view (parties or blocks); those without a value are left out
 * @param {Object<string, string>} colors colour of each name (the marker of a name without one is grey)
 * @returns {string[]} HTML lines, escaped: `<strong>{name}</strong>`, `{date} · muestra {sample}[ · rating
 *   {rating}]` (dashes when missing) and the values
 */
export function pollTooltipLines(poll, parties, colors) {
  const name = poll.name || poll.pollster || DASH;
  const date = poll.date ? fmtDate(poll.date) : DASH;
  const rating = 'rating' in poll ? ` · rating ${fmtNum(poll.rating, 1)}` : '';
  const facts = `${date} · muestra ${fmtInt(poll.sample_size)}${rating}`;
  const lines = [`<strong>${escapeText(name)}</strong>`, escapeText(facts)];
  const rows = parties
    .filter((party) => present(poll[party]))
    .map((party) => ({party, value: poll[party]}))
    .sort((a, b) => b.value - a.value);
  for (const {party, value} of rows) {
    lines.push(`${tooltipMarker(colors[party] || '#999')}${escapeText(`${party} ${fmtNum(value, 1)}`)}`);
  }
  return lines;
}

/**
 * Box of one point of the average, of its projection or of the range bar at the election:
 * `{name}: {mean} ({lo}–{hi}) · {date}`; without bounds, `{name}: {mean} · {date}` (the history of the series
 * chart, whose band is no longer drawn, 2026-10-26).
 *
 * @param {string} name party or block
 * @param {number|null} mean average of the day
 * @param {number|null} lo lower bound of the interval, or null for none
 * @param {number|null} hi upper bound of the interval, or null for none
 * @param {string} date ISO date of the point
 * @param {string} color colour of the marker
 * @param {boolean|string|null} [header=null] a first line over the box: `true` for «Proyección» (a point of the
 *   projection), a string for a literal heading («Elección» for the range bar), null for none
 * @returns {string} the HTML of the box, escaped
 */
export function meanTooltipLine(name, mean, lo, hi, date, color, header = null) {
  const bounds = lo !== null && lo !== undefined && hi !== null && hi !== undefined;
  const text = `${name}: ${fmtPct(mean)}${bounds ? ` (${fmtRange(lo, hi)})` : ''} · ${fmtDate(date)}`;
  const heading = header === true ? 'Proyección' : header;
  return `${heading ? `${escapeText(heading)}<br>` : ''}${tooltipMarker(color)}${escapeText(text)}`;
}

/**
 * Box of the official result of a party or block at the election: a first line names it with the date, the
 * second the name and its share.
 *
 * @param {string} name party or block
 * @param {number|null} value official share
 * @param {string} date ISO date of the election
 * @param {string} color colour of the marker
 * @returns {string} the HTML of the box, escaped
 */
export function resultTooltipLine(name, value, date, color) {
  return `Resultado oficial · ${escapeText(fmtDate(date))}<br>${tooltipMarker(color)}${escapeText(`${name} ${fmtPct(value)}`)}`;
}

/** Least remainder of a block over its parties shown as «Resto», in points (below it, rounding). */
const REST_MIN = 0.05;

/**
 * Lines of the breakdown of a block on one day of the average, under its box: one per party of the block with a
 * value that day, in descending order, with its colour marker and `{party} {value}`, and, for Otros (the rest up to
 * 100) when it holds more than its parties add up to, a grey «Resto» line with the difference. Izquierda and
 * Derecha never get one: their line, fitted on the sum per poll since 2026-11-09, can differ from the sum of their
 * parties for a few months after a party enters, and that is no missing party.
 *
 * @param {string} block the block
 * @param {number|null} total the value of the block that day
 * @param {number} i index of the day in the series
 * @param {{members: Object<string, string[]>, values: Object<string, Array<number|null>>, colors: Object<string, string>}} breakdown
 *   the parties of each block, the series of each party and its colour
 * @returns {string[]} HTML lines, escaped; none for a name that is not a block of `breakdown`
 */
export function breakdownTooltipLines(block, total, i, breakdown) {
  const members = (breakdown.members || {})[block];
  if (!members) {
    return [];
  }
  const rows = members
    .map((party) => ({party, value: ((breakdown.values || {})[party] || [])[i]}))
    .filter((row) => present(row.value))
    .sort((a, b) => b.value - a.value);
  const lines = rows.map(({party, value}) => `${tooltipMarker((breakdown.colors || {})[party] || '#999')}${escapeText(`${party} ${fmtNum(value, 1)}`)}`);
  const rest = present(total) ? total - rows.reduce((sum, row) => sum + row.value, 0) : 0;
  if (block === REST_BLOCK && rest >= REST_MIN) {
    lines.push(`${tooltipMarker('#999')}${escapeText(`Resto ${fmtNum(rest, 1)}`)}`);
  }
  return lines;
}
