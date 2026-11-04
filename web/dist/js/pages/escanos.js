// Seats page (`/{scope}/escanos`, until it redirects to the event page): the stacked bar of «Reparto de escaños»
// with its Partidos/Bloques switch and its download (the bar with the table of its view under it, as a signed image,
// `share.js`) and the evolution of the seats; the calculator, the distributions, the territory and the fan are those
// of the event page (`seats-wire.js`), drawn from the simulations this page still embeds.
import {fmtInt} from '../format.js';
import {Catalog} from '../catalog.js';
import {renderStacked, stackedOption} from '../charts/stacked.js';
import {wireEvolution} from './evolution-wire.js';
import {mapNames, markSegments, mountWhenReady, readInitial, showError, tryDraw, wireControls} from './common.js';
import {BAR_IMAGE_HEIGHT, partyBar, wireSeatsPage} from './seats-wire.js';
import {tableImage, wireDownload} from '../share.js';

/**
 * Wire the stacked bar of the seats card and its Partidos/Bloques switch, which also shows the table and the
 * distributions of the view (`seats.redraw`). The download button draws the bar of the view shown with its table
 * under it.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} seats what `wireSeatsPage` returns
 * @param {function} setView records the view shown
 */
function wireBar(initial, catalog, seats, setView) {
  const {meta, summary, dist} = initial;
  const majority = meta.majority ?? summary.majority;
  const nSeats = meta.n_seats ?? summary.n_seats;
  const colors = mapNames([...new Set([...dist.parties, ...(summary.parties || []).map((row) => row.name)])],
    (name) => catalog.color(name));
  const parties = partyBar(initial, dist, catalog, colors);
  const blockRows = (summary.blocks || []).filter((row) => row.seats !== null && row.seats !== undefined);
  const deltas = initial.deltas || {parties: {}, blocks: {}, year: null};
  const views = {
    partidos: {container: 'seats-parties', bar: parties.rows, colors, deltas: parties.deltas, fullnames: parties.fullnames},
    bloques: {
      container: 'seats-blocks',
      bar: catalog.orderBlocks(blockRows.map((row) => row.name))
        .map((name) => ({name, seats: blockRows.find((row) => row.name === name).seats})),
      colors: mapNames(blockRows.map((row) => row.name), (name) => catalog.blockColor(name)),
      deltas: deltas.blocks,
      fullnames: {},
    },
  };
  const barEl = document.getElementById('seats-bar');
  let shown = views.partidos;
  const barOf = (view) => ({
    colors: view.colors, fullnames: view.fullnames, majority, total: nSeats, deltas: view.deltas, year: deltas.year,
  });
  const draw = (view) => {
    shown = views[view];
    setView(view);
    seats.redraw(view);
    if (barEl) {
      tryDraw(() => renderStacked(barEl, shown.bar, barOf(shown)), `the ${view} bar`);
    }
  };

  draw('partidos');
  wireDownload(document.getElementById('seats-download'), () => stackedOption(shown.bar, barOf(shown)), showError, {
    chartHeight: BAR_IMAGE_HEIGHT, bottom: () => tableImage(document.querySelector(`#${shown.container} table`)),
  });
  const fieldset = document.getElementById('seats-view');
  if (fieldset) {
    fieldset.hidden = false;
    fieldset.addEventListener('change', (event) => {
      if (event.target.matches('input[name=sview]') && views[event.target.value]) {
        draw(event.target.value);
        markSegments(fieldset);
      }
    });
  }
}

/** Draw the seats of the page, its bar and the evolution of the seats. */
function paint() {
  const initial = readInitial();
  const catalog = new Catalog(initial.meta);
  const nSeats = initial.meta.n_seats ?? initial.summary.n_seats;
  let view = 'partidos';
  const seats = wireSeatsPage(initial, catalog, () => view);
  tryDraw(() => wireBar(initial, catalog, seats, (name) => {
    view = name;
  }), 'the seats card');
  // the parties of the summary, in its order: the same list and order as the average of the event page
  tryDraw(() => wireEvolution(initial, catalog, {
    parties: (initial.summary.parties || []).map((row) => row.name),
    metrics: {seats: {field: 'seats', formatter: fmtInt, total: nSeats}},
  }), 'the evolution');
}

wireControls();
mountWhenReady(paint);
