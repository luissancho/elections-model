// Poll average page: the series chart with polls and projection, drawn from the data embedded by the server,
// and the window control ("Últimos 6 meses" / "Todo el ciclo") that zooms it without reloading.
import {Catalog} from '../catalog.js';
import {renderSeries, setWindow} from '../charts/series.js';
import {mapNames, mountWhenReady, readInitial, wireControls} from './common.js';

/** Value of the checked option of the window control (`6m` when the control is missing). */
function windowChoice() {
  const checked = document.querySelector('#series-window input[name=window]:checked');
  return checked ? checked.value : '6m';
}

/** Draw the series chart from the embedded data and zoom it when the window control changes. */
function paint() {
  const {meta, series, polls, projection, vote} = readInitial();

  const catalog = new Catalog(meta);
  const parties = series.parties;
  const main = (catalog.bmaps.main || []).filter((name) => parties.includes(name));
  const asOf = meta.as_of;
  const anchor = meta.date_last || asOf;

  const chart = renderSeries(document.getElementById('series-chart'), {
    series,
    polls,
    projection,
    parties,
    selected: main.length ? main : null,
    colors: mapNames(parties, (name) => catalog.color(name)),
    fullnames: mapNames(parties, (name) => catalog.fullname(name)),
    asOf,
    when: vote.when,
    window: windowChoice(),
    anchor,
  });

  const control = document.getElementById('series-window');
  if (control) {
    control.addEventListener('change', (event) => {
      if (event.target.matches('input[name=window]')) {
        setWindow(chart, event.target.value, {series, projection, anchor});
      }
    });
  }
}

wireControls();
mountWhenReady(paint);
