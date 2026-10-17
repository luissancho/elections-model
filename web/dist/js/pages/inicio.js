// Home page: the small hemicycle of the featured election, drawn from the data embedded by the server.
import {renderHemicycle} from '../charts/hemicycle.js';
import {mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

/** Draw the hemicycle of the hero from the embedded data. */
function paint() {
  const {hero} = readInitial();
  const el = document.getElementById('hemicycle-mini');
  if (!hero || !el) {
    return;
  }
  tryDraw(() => renderHemicycle(el, hero.seats, {
    colors: hero.colors,
    order: (names) => hero.order.filter((name) => names.includes(name)),
    nSeats: hero.n_seats,
    majority: hero.majority,
    fullnames: hero.fullnames,
    deltas: hero.deltas,  // seat differences against the last election, after the seats of each arc
    year: hero.results_year,
  }), 'the hemicycle of the home');
}

wireControls();
mountWhenReady(paint);
