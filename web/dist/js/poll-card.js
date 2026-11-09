/** La lógica pura de la tarjeta de un sondeo: filas del voto, escaños de la casa y anillo sobre sus series. */
import {OTHERS_COLOR} from './catalog.js';
import {dayTime} from './charts/series.js';

/** Un valor numérico finito (los huecos llegan como null o undefined). */
const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);

/**
 * Las filas del voto de un sondeo: los partidos con valor, de mayor a menor (los empates conservan el orden de
 * `parties`), con su cambio frente a la entrega anterior de la casa y medio. Sin «Otros» (Luis, 2026-11-09): el resto
 * del voto no se dibuja; en los escaños, el arco «Otros» del hemiciclo sí.
 * @param {Object} poll El sondeo, con el valor de cada partido por nombre.
 * @param {string[]} parties Los partidos, en su orden.
 * @param {Object|null} [previous] La entrega anterior (voto por partido), si la hay.
 * @returns {{name: string, value: number, change: number|null}[]} Las filas; `change` es la diferencia con la entrega
 *   anterior a un decimal (null sin cifra en las dos entregas).
 */
export function voteRows(poll, parties, previous = null) {
    const rows = parties
        .filter((name) => isNumber(poll[name]))
        .map((name) => ({
            name,
            value: poll[name],
            change: previous && isNumber(previous[name]) ? Math.round((poll[name] - previous[name]) * 10) / 10 : null,
        }));
    rows.sort((a, b) => b.value - a.value);
    return rows;
}

/**
 * Los escaños de una casa para el hemiciclo: el punto medio de cada horquilla, su rótulo y el resto hasta la cámara
 * (el arco «Otros»: los escaños que la casa da a partidos sin cifra propia en la tabla y el redondeo de las
 * horquillas, no escaños sin asignar).
 * @param {Object|null} seats Las horquillas `[lo, hi]` por partido.
 * @param {number} nSeats Los escaños de la cámara (el hemiciclo necesita los mismos `nSeats` para dibujar el resto).
 * @returns {{seats: Object, labels: Object, rest: number|null}} Puntos medios, rótulos y resto (null si no queda).
 */
export function pollSeats(seats, nSeats) {
    const out = {seats: {}, labels: {}, rest: null};
    if (!seats) {
        return out;
    }
    let total = 0;
    for (const [name, range] of Object.entries(seats)) {
        const [lo, hi] = range;
        out.seats[name] = Math.round((lo + hi) / 2);
        out.labels[name] = lo === hi ? String(lo) : `${lo}–${hi}`;
        total += out.seats[name];
    }
    if (Object.keys(out.seats).length && nSeats - total > 0) {
        out.rest = nSeats - total;
    }

    return out;
}

/**
 * El cambio de los escaños de cada partido frente a la entrega anterior: la diferencia entre los puntos medios de sus
 * dos horquillas (los mismos que dibuja el hemiciclo), sólo para los partidos con horquilla en las dos.
 * @param {Object|null} seats Las horquillas `[lo, hi]` de este sondeo por partido.
 * @param {Object|null} previous Las de la entrega anterior.
 * @returns {Object} `{partido: diferencia}` (vacío sin escaños en alguna de las dos).
 */
export function seatChanges(seats, previous) {
    const mid = (range) => (Array.isArray(range) && range.length === 2 && range.every(isNumber)
        ? Math.round((range[0] + range[1]) / 2) : null);
    const out = {};
    for (const [name, range] of Object.entries(seats || {})) {
        const now = mid(range);
        const before = mid((previous || {})[name]);
        if (now !== null && before !== null) {
            out[name] = now - before;
        }
    }
    return out;
}

/**
 * Los anillos que marcan este sondeo sobre las series de la casa: un punto hueco por partido con valor, con el nombre
 * de la línea para que la leyenda lo oculte con ella.
 * @param {Object} poll El sondeo (con `date`).
 * @param {string[]} parties Los partidos.
 * @param {Object} colors El color por partido.
 * @returns {Object[]} Las series de ECharts.
 */
export function highlightSeries(poll, parties, colors) {
    return parties.filter((name) => isNumber(poll[name])).map((name) => ({
        type: 'scatter',
        name,
        id: `highlight:${name}`,
        silent: true,
        z: 6,
        symbol: 'circle',
        symbolSize: 11,
        itemStyle: {color: 'transparent', borderColor: colors[name] || OTHERS_COLOR, borderWidth: 2},
        tooltip: {show: false},
        data: [[dayTime(poll.date), poll[name]]],
    }));
}
