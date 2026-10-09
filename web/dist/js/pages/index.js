// Home page: vote and seat bars, hemicycle and evolution, drawn from the data embedded by the server.
import {fmtInt, fmtPct} from '../format.js';
import {Catalog} from '../catalog.js';
import {renderBars} from '../charts/bars.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {renderEvolution} from '../charts/evolution.js';
import {mapNames, mountWhenReady, readInitial, wireControls} from './common.js';

function paint() {
  const initial = readInitial();
  const {meta, headline, vote, summary, runs} = initial;
  const mode = initial.state.mode;

  const catalog = new Catalog(meta);
  const current = headline[mode];
  const names = [...new Set([
    ...catalog.parties.map((party) => party.name),
    ...current.parties.map((party) => party.name),
    ...vote.rows.map((row) => row.name),
    ...summary.parties.map((row) => row.name),
  ])];
  const colors = mapNames(names, (name) => catalog.color(name));
  const fullnames = mapNames(names, (name) => catalog.fullname(name));
  const nSeats = meta.n_seats ?? summary.n_seats;
  const majority = meta.majority ?? summary.majority;

  const voteRows = [...vote.rows].sort((a, b) => b.pct - a.pct);
  renderBars(document.getElementById('vote-chart'), voteRows, {
    value: 'pct', lo: 'lo', hi: 'hi', colors, fullnames, formatter: (x) => fmtPct(x),
  });

  const seatRows = summary.parties
    .filter((row) => row.seats > 0 || row.seats_hi > 0)
    .sort((a, b) => b.seats - a.seats);
  renderBars(document.getElementById('seats-chart'), seatRows, {
    value: 'seats', lo: 'seats_lo', hi: 'seats_hi', colors, fullnames, formatter: fmtInt, majority,
  });

  const seats = summary.totals
    ?? Object.fromEntries(current.parties.map((party) => [party.name, party.seats]));
  renderHemicycle(document.getElementById('hemicycle'), seats, {
    colors, fullnames, order: (list) => catalog.order(list), nSeats, majority,
  });

  renderEvolution(document.getElementById('evolution'), runs.runs, mode, {
    colors, fullnames, parties: current.parties.map((party) => party.name),
  });
}

wireControls();
mountWhenReady(paint);
