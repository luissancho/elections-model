// Pure logic of the pollsters ranking of /casas: which rows the table shows with the "Solo con experiencia en
// generales" filter and the "Cargar más" batch.

/**
 * Which rows of the ranking are shown, filtered out and still to load.
 *
 * @param {number[]} nationalEvents general elections each row's pollster has polled, in the order of the table
 * @param {boolean} onlyNationalEvents true to drop the rows with no general elections
 * @param {number} shown how many unfiltered rows to show (the first ones)
 * @returns {{visible: boolean[], filtered: boolean[], more: boolean}} `filtered[i]`: row excluded by the filter;
 *   `visible[i]`: one of the first `shown` unfiltered rows; `more`: unfiltered rows are left unshown
 */
export function rankingVisibility(nationalEvents, onlyNationalEvents, shown) {
  const filtered = nationalEvents.map((count) => onlyNationalEvents && count === 0);
  let taken = 0;
  const visible = filtered.map((out) => {
    if (out || taken >= shown) {
      return false;
    }
    taken += 1;
    return true;
  });
  const more = filtered.some((out, i) => !out && !visible[i]);
  return {visible, filtered, more};
}
