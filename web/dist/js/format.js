// Number and date formatting (es-ES).

const DASH = '–';

function isMissing(x) {
  return x === null || x === undefined || Number.isNaN(x);
}

const numberFormats = new Map();

function numberFormat(digits) {
  if (!numberFormats.has(digits)) {
    numberFormats.set(digits, new Intl.NumberFormat('es-ES', {
      minimumFractionDigits: digits, maximumFractionDigits: digits,
    }));
  }
  return numberFormats.get(digits);
}

export function fmtNum(x, digits = 0) {
  return isMissing(x) ? DASH : numberFormat(digits).format(x);
}

export function fmtInt(x) {
  return fmtNum(x, 0);
}

/**
 * A count with the thousands always marked, also with four digits ('1.000'), as `fmt_num` of `pages/filters.py`
 * writes it; `fmtInt` follows the es-ES rule of leaving four digits ungrouped ('1000'), which keeps years clean.
 */
export function fmtCount(x) {
  if (isMissing(x)) {
    return DASH;
  }
  const digits = String(Math.round(Math.abs(x))).replace(/\B(?=(\d{3})+(?!\d))/g, '.');
  return x < 0 ? `-${digits}` : digits;
}

/** Percentage already in the 0-100 scale, the sign attached: '32,7%'. */
export function fmtPct(x, digits = 1) {
  return isMissing(x) ? DASH : `${numberFormat(digits).format(x)}%`;
}

/**
 * Probability in [0, 1] as a percentage that never rounds a near-certainty away (the mirror of `fmt_prob` of
 * `pages/filters.py`): '0,0%' and '100,0%' only when exact; capped below the top value the decimals can show
 * ('> 99,9%' from 0.9995 with one decimal, '> 99%' from 0.995 with none) and floored above zero ('< 0,1%' up to
 * 0.0005, '< 1%' up to 0.005).
 */
export function fmtProb(p, digits = 1) {
  if (isMissing(p)) {
    return DASH;
  }
  if (p === 0) {
    return fmtPct(0, digits);
  }
  if (p === 1) {
    return fmtPct(100, digits);
  }
  const unit = 10 ** -digits;
  const edge = unit / 2 / 100;
  if (p >= 1 - edge) {
    return `> ${fmtPct(100 - unit, digits)}`;
  }
  if (p > 0 && p <= edge) {
    return `< ${fmtPct(unit, digits)}`;
  }
  return fmtPct(p * 100, digits);
}

export function fmtRange(lo, hi, digits = 1) {
  if (isMissing(lo) || isMissing(hi)) {
    return DASH;
  }
  const f = numberFormat(digits);
  return `${f.format(lo)}${DASH}${f.format(hi)}`;
}

const MONTHS = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sept', 'oct', 'nov', 'dic'];

/** Parse an ISO date (date-only values as local midnight); null when invalid. */
function parseDate(iso) {
  if (!iso) {
    return null;
  }
  const text = /^\d{4}-\d{2}-\d{2}$/.test(iso) ? `${iso}T00:00:00` : iso;
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? null : date;
}

function dayMonth(date) {
  return `${date.getDate()} ${MONTHS[date.getMonth()]}`;
}

/**
 * Label of a tick of a time axis, whatever the span shown: ECharts puts the ticks on round boundaries for the
 * span (years, months or days), and each tick says what it is. 1 January is the year, the 1st of another month
 * its month, any other day its day and month; the first round tick always carries the year, so a short window
 * that does not cross a new year still dates itself. ECharts gives index 0 to the left end of the axis (a date
 * that is not round and is not drawn) and 1 to the first round tick, so both carry it. The year goes in the rich
 * style `year` (bold).
 *
 * @param {number} time timestamp of the tick (local midnight)
 * @param {number} index position of the tick on the axis (0: its left end; 1: the first round tick)
 * @returns {string} the label, with `{year|YYYY}` rich text
 */
export function fmtAxisDate(time, index) {
  const date = new Date(time);
  const year = `{year|${date.getFullYear()}}`;
  if (date.getMonth() === 0 && date.getDate() === 1) {
    return year;
  }
  const text = date.getDate() === 1 ? MONTHS[date.getMonth()] : dayMonth(date);
  return index <= 1 ? `${text} ${year}` : text;
}

/** '2026-10-13' → '13 oct 2026' */
export function fmtDate(iso) {
  const date = parseDate(iso);
  return date ? `${dayMonth(date)} ${date.getFullYear()}` : DASH;
}

/** '2026-10-13' → '13 oct' */
export function fmtDateShort(iso) {
  const date = parseDate(iso);
  return date ? dayMonth(date) : DASH;
}

/** ISO timestamp (run_at) → '8 oct 2026, 20:11' in the viewer's time zone. */
export function fmtDateTime(iso) {
  const date = parseDate(iso);
  if (!date) {
    return DASH;
  }
  const pad = (n) => String(n).padStart(2, '0');
  return `${dayMonth(date)} ${date.getFullYear()}, ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

const HEX_COLOR = /^#[0-9A-Fa-f]{6}$/;

/**
 * Background of a shaded table cell, the mirror of `shade` of `pages/filters.py`: `color` with an opacity that grows
 * with the value from `start` (transparent) to `end` (`alpha`); the range may run downwards.
 *
 * @param {(number|null|undefined)} x value of the cell
 * @param {number} start value of the transparent cell
 * @param {number} end value of the fullest cell
 * @param {string} color `#rrggbb`
 * @param {number} [alpha=0.55] opacity of the fullest cell
 * @returns {string} an `rgba()` colour, or `transparent` when the value is missing, the range is empty or the
 *   colour is not a six-digit hex colour
 */
export function shade(x, start, end, color, alpha = 0.55) {
  if (isMissing(x) || !Number.isFinite(x) || !Number.isFinite(start) || !Number.isFinite(end) || start === end
      || typeof color !== 'string' || !HEX_COLOR.test(color)) {
    return 'transparent';
  }
  const fraction = Math.min(1, Math.max(0, (x - start) / (end - start)));
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${Math.round(alpha * fraction * 1000) / 1000})`;
}

/**
 * Background of a cell shaded both ways from a centre, the mirror of `shade_around` of `pages/filters.py`: `below`
 * grows from `centre` (transparent) down to `centre - span` and `above` from `centre` up to `centre + span`.
 *
 * @param {(number|null|undefined)} x value of the cell
 * @param {number} centre value of the transparent cell
 * @param {number} span distance from the centre of the fullest cell, either way (positive)
 * @param {string} below `#rrggbb` of the values below the centre
 * @param {string} above `#rrggbb` of the values above (or at) the centre
 * @param {number} [alpha=0.55] opacity of the fullest cell
 * @returns {string} an `rgba()` colour, or `transparent` when the value is missing or the span is not positive
 */
export function shadeAround(x, centre, span, below, above, alpha = 0.55) {
  if (isMissing(x) || !Number.isFinite(x) || !Number.isFinite(centre) || !Number.isFinite(span) || span <= 0) {
    return 'transparent';
  }
  return x < centre ? shade(x, centre, centre - span, below, alpha) : shade(x, centre, centre + span, above, alpha);
}

/**
 * Colour of a value on a scale of several stops, the mirror of `gradient` of `pages/filters.py`: the colours are
 * spread evenly from `start` to `end` and the value takes the mix of the two it falls between (clipped to the ends).
 *
 * @param {(number|null|undefined)} x value to colour
 * @param {number} start value of the first colour
 * @param {number} end value of the last colour (the range may run downwards)
 * @param {string[]} colors two or more `#rrggbb` colours, from `start` to `end`
 * @returns {string} a `#rrggbb` colour (each channel rounded half up), or `transparent` when the value is missing,
 *   the range is empty or there are not two valid colours
 */
export function gradient(x, start, end, colors) {
  if (isMissing(x) || !Number.isFinite(x) || !Number.isFinite(start) || !Number.isFinite(end) || start === end
      || !Array.isArray(colors) || colors.length < 2
      || !colors.every((color) => typeof color === 'string' && HEX_COLOR.test(color))) {
    return 'transparent';
  }
  const position = Math.min(1, Math.max(0, (x - start) / (end - start))) * (colors.length - 1);
  const index = Math.min(Math.floor(position), colors.length - 2);
  const t = position - index;
  const channels = (color) => [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
  const [lo, hi] = [channels(colors[index]), channels(colors[index + 1])];
  return `#${lo.map((a, i) => Math.floor(a + (hi[i] - a) * t + 0.5).toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Arrow and magnitude of a difference against the last election, the mirror of `delta_badge` of `pages/filters.py`:
 * '▲ 6', '▼ 1,0', or the dash when the magnitude rounds to zero or the value is missing.
 *
 * @param {(number|null|undefined)} x the difference
 * @param {number} [digits=0] decimals of the magnitude
 * @returns {string} the text
 */
export function fmtDelta(x, digits = 0) {
  if (isMissing(x) || !Number.isFinite(x)) {
    return DASH;
  }
  const magnitude = fmtNum(Math.abs(x), digits);
  if (/^[0,.]*$/.test(magnitude)) {
    return DASH;
  }
  return `${x > 0 ? '▲' : '▼'} ${magnitude}`;
}
