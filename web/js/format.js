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

export function fmtPct(x, digits = 1) {
  return isMissing(x) ? DASH : `${numberFormat(digits).format(x)} %`;
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
