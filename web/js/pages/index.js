// Home page: headline table, vote and seat bars, hemicycle, majorities and evolution.
import {ApiError, getForecast, getManifest, getScopes} from '../api.js';
import {readState, writeState, onStateChange} from '../state.js';
import {fmtDate, fmtInt, fmtPct, fmtRange} from '../format.js';
import {Catalog} from '../catalog.js';
import {renderHeader, renderFooter, renderFreeze, renderError} from '../layout.js';
import {renderBars} from '../charts/bars.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {renderEvolution} from '../charts/evolution.js';

const PAGES = [{href: '/', label: 'Portada'}, {href: '/promedio', label: 'Promedio'}];

const dom = {
  header: document.getElementById('site-header'),
  footer: document.getElementById('site-footer'),
  banner: document.getElementById('freeze-banner'),
  status: document.getElementById('status'),
  content: document.getElementById('content'),
  headlineTable: document.getElementById('headline-table'),
  averageLink: document.getElementById('average-link'),
  vote: document.getElementById('vote-chart'),
  seats: document.getElementById('seats-chart'),
  hemicycle: document.getElementById('hemicycle'),
  majority: document.getElementById('majority'),
  evolution: document.getElementById('evolution'),
};

// Fetched once per page view.
let manifest = null;
let scopes = [];
// Increases with every load, so a slow response of an older state never paints over a newer one.
let loadId = 0;

function el(tag, text = null, className = null) {
  const node = document.createElement(tag);
  if (text !== null) {
    node.textContent = text;
  }
  if (className) {
    node.className = className;
  }
  return node;
}

function showStatus(text) {
  dom.status.classList.remove('error');
  dom.status.textContent = text;
  dom.status.hidden = false;
}

/** `{name: value}` for every name, from a catalogue method. */
function mapNames(names, fn) {
  return Object.fromEntries(names.map((name) => [name, fn(name)]));
}

/** Section titles with the mode: event date (forecast) or estimate date (nowcast). */
function renderTitles(meta, mode) {
  const title = mode === 'nowcast'
    ? `Estimación a ${fmtDate(meta.as_of)}`
    : `Pronóstico para el ${fmtDate(meta.event_date)}`;
  for (const node of document.querySelectorAll('.mode-title')) {
    node.textContent = title;
  }
  const subtitle = `· ${title.charAt(0).toLowerCase()}${title.slice(1)}`;
  for (const node of document.querySelectorAll('.mode-subtitle')) {
    node.textContent = subtitle;
  }
  const evolution = mode === 'nowcast' ? '· estimación de cada publicación' : '· pronóstico de cada publicación';
  for (const node of document.querySelectorAll('.evolution-subtitle')) {
    node.textContent = evolution;
  }
}

/** Table of the headline parties, in arrival order. */
function renderHeadlineTable(root, parties, catalog) {
  const table = el('table');
  const head = el('tr');
  for (const label of ['Partido', 'Voto', 'Escaños', 'Primero']) {
    head.append(el('th', label));
  }
  table.append(el('thead'));
  table.tHead.append(head);
  const body = el('tbody');
  for (const party of parties) {
    const row = el('tr');
    const name = el('td');
    name.title = party.name;
    const dot = el('span', null, 'dot');
    dot.style.backgroundColor = catalog.color(party.name);
    name.append(dot, catalog.fullname(party.name));
    const vote = el('td');
    vote.append(`${fmtPct(party.pct)} `, el('span', fmtRange(party.lo, party.hi), 'muted'));
    const seats = el('td');
    seats.append(`${fmtInt(party.seats)} `, el('span', fmtRange(party.seats_lo, party.seats_hi, 0), 'muted'));
    const first = el('td', fmtPct(party.p_first * 100, 0));
    row.append(name, vote, seats, first);
    body.append(row);
  }
  table.append(body);
  root.replaceChildren(table);
}

/** One row per block of `pMajority` (object order): name, 0-100 % bar and the probability text. */
function renderMajority(root, pMajority, catalog) {
  const rows = Object.entries(pMajority || {}).map(([block, p]) => {
    const row = el('div', null, 'majority-row');
    const track = el('div', null, 'majority-track');
    const bar = el('div', null, 'majority-bar');
    bar.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
    bar.style.backgroundColor = catalog.blockColor(block);
    track.append(bar);
    row.append(
      el('strong', block),
      track,
      el('span', `${fmtPct(p * 100, 0)} de probabilidad de mayoría absoluta`, 'majority-text'),
    );
    return row;
  });
  root.replaceChildren(...rows);
}

/** First published scope, in catalogue order (manifest order as fallback). */
function firstPublished() {
  const published = Object.keys(manifest.data.scopes || {});
  const row = scopes.find((scope) => scope.simulable && published.includes(scope.code));
  return row ? row.code : published[0] || null;
}

/** Load everything that depends on scope/mode/run and paint the page; errors of stale loads are dropped. */
async function load() {
  const id = ++loadId;
  try {
    await paint(id);
  } catch (error) {
    if (id === loadId) {
      fail(error);
    }
  }
}

async function paint(id) {
  const state = readState();
  renderHeader(dom.header, {pages: PAGES, active: '/', scopes, state});
  dom.averageLink.href = `/promedio${location.search}`;
  showStatus('Cargando…');

  const {scope, mode, run} = state;
  const [meta, headline, vote, summary, history] = await Promise.all([
    getForecast(scope, 'meta', {run}),
    getForecast(scope, 'headline', {run}),
    getForecast(scope, 'vote', {mode, run}),
    getForecast(scope, 'summary', {mode, run}),
    getForecast(scope, 'runs'),
  ]);
  if (id !== loadId) {
    return;
  }

  const catalog = new Catalog(meta.data);
  const current = headline.data[mode];
  const names = [...new Set([
    ...catalog.parties.map((party) => party.name),
    ...current.parties.map((party) => party.name),
    ...vote.data.rows.map((row) => row.name),
    ...summary.data.parties.map((row) => row.name),
  ])];
  const colors = mapNames(names, (name) => catalog.color(name));
  const fullnames = mapNames(names, (name) => catalog.fullname(name));
  const nSeats = meta.data.n_seats ?? summary.data.n_seats;
  const majority = meta.data.majority ?? summary.data.majority;

  // Charts need a laid-out container: show the content before mounting them.
  dom.content.hidden = false;
  renderTitles(meta.data, mode);
  renderHeadlineTable(dom.headlineTable, current.parties, catalog);

  const voteRows = [...vote.data.rows].sort((a, b) => b.pct - a.pct);
  renderBars(dom.vote, voteRows, {value: 'pct', lo: 'lo', hi: 'hi', colors, fullnames, formatter: (x) => fmtPct(x)});

  const seatRows = summary.data.parties
    .filter((row) => row.seats > 0 || row.seats_hi > 0)
    .sort((a, b) => b.seats - a.seats);
  renderBars(dom.seats, seatRows, {
    value: 'seats', lo: 'seats_lo', hi: 'seats_hi', colors, fullnames, formatter: fmtInt, majority,
  });

  const seats = summary.data.totals
    ?? Object.fromEntries(current.parties.map((party) => [party.name, party.seats]));
  renderHemicycle(dom.hemicycle, seats, {
    colors, fullnames, order: (list) => catalog.order(list), nSeats, majority,
  });

  renderMajority(dom.majority, current.p_majority, catalog);
  renderEvolution(dom.evolution, history.data.runs, mode, {
    colors, fullnames, parties: current.parties.map((party) => party.name),
  });

  renderFooter(dom.footer, {meta, manifest});
  renderFreeze(dom.banner, manifest);
  dom.status.hidden = true;
}

function fail(error) {
  console.error(error);
  dom.content.hidden = true;
  if (error instanceof ApiError) {
    renderError(dom.status, error);
  } else {
    renderError(dom.status, {status: 0, message: 'No se pudo mostrar la página.'});
  }
}

async function init() {
  renderHeader(dom.header, {pages: PAGES, active: '/', scopes: [], state: readState()});
  try {
    const [manifestEnvelope, scopesEnvelope] = await Promise.all([getManifest(), getScopes()]);
    manifest = manifestEnvelope;
    scopes = scopesEnvelope.scopes ?? scopesEnvelope.data?.scopes ?? [];
    renderFreeze(dom.banner, manifest);

    const state = readState();
    if (!(state.scope in (manifest.data.scopes || {}))) {
      const scope = firstPublished();
      if (!scope) {
        throw new ApiError(503, 'Sin pronósticos publicados');
      }
      // Before the listener exists, so this does not trigger a second load.
      writeState({scope, run: null});
    }
    onStateChange(load);
    await load();
  } catch (error) {
    fail(error);
  }
}

init();
