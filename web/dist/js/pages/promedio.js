// Poll average page: the series chart with polls and projection, drawn from the data embedded by the server.
import {Catalog} from '../catalog.js';
import {renderSeries} from '../charts/series.js';
import {mapNames, mountWhenReady, readInitial, wireControls} from './common.js';

/** Draw the series chart from the embedded data. */
function paint() {
  const {meta, series, polls, projection, vote} = readInitial();

  const catalog = new Catalog(meta);
  const parties = series.parties;
  const main = (catalog.bmaps.main || []).filter((name) => parties.includes(name));
  const asOf = meta.as_of;

  renderSeries(document.getElementById('series-chart'), {
    series,
    polls,
    projection,
    parties,
    selected: main.length ? main : null,
    colors: mapNames(parties, (name) => catalog.color(name)),
    fullnames: mapNames(parties, (name) => catalog.fullname(name)),
    asOf,
    when: vote.when,
  });
}

wireControls();
mountWhenReady(paint);
