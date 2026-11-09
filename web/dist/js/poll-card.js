/** La lógica pura de la tarjeta de un sondeo: filas del voto, escaños de la casa y anillo sobre sus series. */
import {OTHERS_COLOR} from './catalog.js';
import {dayTime} from './charts/series.js';

/** Un valor numérico finito (los huecos llegan como null o undefined). */
const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);

/**
 * Las filas del voto de un sondeo: los partidos con valor, de mayor a menor (los empates conservan el orden de
 * `parties`), con su promedio, y al final «Otros» si el resto hasta 100 llega a 0,5 puntos.
 * @param {Object} poll El sondeo, con el valor de cada partido por nombre.
 * @param {string[]} parties Los partidos, en su orden.
 * @param {Object} [average] El promedio por partido.
 * @param {Object|null} [previous] La entrega anterior de la casa y medio (voto por partido), si la hay.
 * @returns {{name: string, value: number, avg: number|null, change: number|null}[]} Las filas; `change` es la
 *   diferencia con la entrega anterior a un decimal (null sin cifra en las dos entregas y en «Otros»).
 */
export function voteRows(poll, parties, average = {}, previous = null) {
    const rows = parties
        .filter((name) => isNumber(poll[name]))
        .map((name) => ({
            name,
            value: poll[name],
            avg: isNumber(average[name]) ? average[name] : null,
            change: previous && isNumber(previous[name]) ? Math.round((poll[name] - previous[name]) * 10) / 10 : null,
        }));
    rows.sort((a, b) => b.value - a.value);
    const total = rows.reduce((sum, row) => sum + row.value, 0);
    const rest = Math.round((100 - total) * 10) / 10;
    if (rest >= 0.5) {
        rows.push({name: 'Otros', value: rest, avg: null, change: null});
    }

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
        symbolSize: 16,
        itemStyle: {color: 'transparent', borderColor: colors[name] || OTHERS_COLOR, borderWidth: 3},
        tooltip: {show: false},
        data: [[dayTime(poll.date), poll[name]]],
    }));
}
