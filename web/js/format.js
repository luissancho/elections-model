// Number and date formatting (es-ES).

const DASH = '–';

function isMissing(x) {
  return x === null || x === undefined || Number.isNaN(x);
}

function numberFormat(digits) {
  return new Intl.NumberFormat('es-ES', {minimumFractionDigits: digits, maximumFractionDigits: digits});
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

function parseDate(iso) {
  return new Date(`${iso}T00:00:00`);
}

function stripDot(text) {
  return text.replace(/\./g, '');
}

/** '2026-10-13' → '13 oct 2026' */
export function fmtDate(iso) {
  if (!iso) {
    return DASH;
  }
  const f = new Intl.DateTimeFormat('es-ES', {day: 'numeric', month: 'short', year: 'numeric'});
  return stripDot(f.format(parseDate(iso))).replace(/ de /g, ' ');
}

/** '2026-10-13' → '13 oct' */
export function fmtDateShort(iso) {
  if (!iso) {
    return DASH;
  }
  const f = new Intl.DateTimeFormat('es-ES', {day: 'numeric', month: 'short'});
  return stripDot(f.format(parseDate(iso))).replace(/ de /g, ' ');
}

/** ISO timestamp (run_at) → '8 oct 2026, 20:11' in the viewer's time zone. */
export function fmtDateTime(iso) {
  if (!iso) {
    return DASH;
  }
  const f = new Intl.DateTimeFormat('es-ES', {
    day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  });
  return stripDot(f.format(new Date(iso))).replace(/ de /g, ' ');
}
