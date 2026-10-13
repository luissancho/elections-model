// Progressive loading of the polls table of Promedios: which rows come next and what the caption says.

/** Rows added by each click of "Cargar más". */
export const TABLE_STEP = 40;

/**
 * Rows to append after the ones already shown.
 *
 * @param {Array} all every row of the table, in display order
 * @param {number} shown rows already in the table
 * @param {number} [step=TABLE_STEP] rows added per click
 * @returns {Array} the next `step` rows (fewer at the end; none when every row is shown)
 */
export function nextRows(all, shown, step = TABLE_STEP) {
  return all.slice(shown, shown + step);
}

/**
 * Caption of the table, as the server writes it.
 *
 * @param {number} shown rows in the table
 * @param {number} total polls of the cycle
 * @returns {string} `Últimos {shown} de {total} sondeos`, or `Todos los sondeos del ciclo ({total})` once all
 *   of them are shown
 */
export function captionText(shown, total) {
  return shown < total ? `Últimos ${shown} de ${total} sondeos` : `Todos los sondeos del ciclo (${total})`;
}
