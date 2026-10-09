// Initial zoom windows of the time-axis charts: pure date arithmetic over ISO dates (no DOM, no ECharts).

/** Days shown by each choice of the window control; a choice outside this map shows the whole axis. */
export const WINDOW_DAYS = {'6m': 180};

/** Choice of the window control when nothing says otherwise. */
export const DEFAULT_WINDOW = '6m';

/**
 * Window of the last `days` days: from `days` before `anchor` (the last date when null), clamped to the
 * axis, up to the last date.
 *
 * @param {string[]} dates ascending ISO dates of the axis
 * @param {number} days window length before the anchor
 * @param {string|null} anchor ISO date the window counts back from
 * @returns {{startValue: string, endValue: string}|null} ISO bounds, or null without dates
 */
export function lastWindow(dates, days, anchor = null) {
  if (!dates.length) {
    return null;
  }
  const first = dates[0];
  const last = dates[dates.length - 1];
  const from = new Date(`${anchor || last}T00:00:00Z`);
  from.setUTCDate(from.getUTCDate() - days);
  const start = from.toISOString().slice(0, 10);
  return {startValue: start < first ? first : (start > last ? last : start), endValue: last};
}

/**
 * Window of a choice of the window control: a key of `WINDOW_DAYS` gives the last days before `anchor`;
 * any other choice (`all`) gives the whole axis.
 *
 * @param {string} choice value of the control (`6m`, `all`)
 * @param {string[]} dates ascending ISO dates of the axis
 * @param {string|null} anchor ISO date the window counts back from
 * @returns {{startValue: string, endValue: string}|null} ISO bounds, or null without dates
 */
export function seriesWindow(choice, dates, anchor = null) {
  if (!dates.length) {
    return null;
  }
  if (Object.hasOwn(WINDOW_DAYS, choice)) {
    return lastWindow(dates, WINDOW_DAYS[choice], anchor);
  }
  return {startValue: dates[0], endValue: dates[dates.length - 1]};
}
