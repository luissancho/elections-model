// Tooltip boxes of the poll average: one poll (name, date, sample and its values in descending order) and one
// point of the average or of its projection. Pure functions, without ECharts or the DOM, testable in node.
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
 * Lines of the box of one poll: its name in bold, the date and the sample, and one line per name of `parties`
 * with a value in the poll, in descending order of value, with its colour marker and `{name} {value}`.
 *
 * @param {object} poll poll record (`name` or else `pollster`, `date`, `sample_size` and one value per name)
 * @param {string[]} parties names of the view (parties or blocks); those without a value are left out
 * @param {Object<string, string>} colors colour of each name (the marker of a name without one is grey)
 * @returns {string[]} HTML lines, escaped: `<strong>{name}</strong>`, `{date} · muestra {sample}` (dashes when
 *   missing) and the values
 */
export function pollTooltipLines(poll, parties, colors) {
  const name = poll.name || poll.pollster || DASH;
  const date = poll.date ? fmtDate(poll.date) : DASH;
  const lines = [`<strong>${escapeText(name)}</strong>`, escapeText(`${date} · muestra ${fmtInt(poll.sample_size)}`)];
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
 * Box of one point of the average (or of its projection): `{name}: {mean} ({lo}–{hi}) · {date}`.
 *
 * @param {string} name party or block
 * @param {number|null} mean average of the day
 * @param {number|null} lo lower bound of the band
 * @param {number|null} hi upper bound of the band
 * @param {string} date ISO date of the point
 * @param {string} color colour of the marker
 * @param {boolean} [projected=false] whether the point belongs to the projection (a first line says so)
 * @returns {string} the HTML of the box, escaped
 */
export function meanTooltipLine(name, mean, lo, hi, date, color, projected = false) {
  const text = `${name}: ${fmtPct(mean)} (${fmtRange(lo, hi)}) · ${fmtDate(date)}`;
  return `${projected ? 'Proyección<br>' : ''}${tooltipMarker(color)}${escapeText(text)}`;
}
