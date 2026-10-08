// Poll average page: the series chart with polls and projection, and the table of the latest polls.
import {ApiError, apiBase, forecastUrl, getForecast, getManifest, getScopes} from '../api.js';
import {readState, writeState, onStateChange} from '../state.js';
import {fmtDate, fmtInt, fmtNum} from '../format.js';
import {Catalog} from '../catalog.js';
import {renderHeader, updateHeader, renderFooter, renderFreeze, renderError} from '../layout.js';
import {renderSeries} from '../charts/series.js';

const PAGES = [{href: '/', label: 'Portada'}, {href: '/promedio', label: 'Promedio'}];
const ACTIVE = '/promedio';
const TABLE_ROWS = 40;
const DASH = '–';

const dom = {
  header: document.getElementById('site-header'),
  footer: document.getElementById('site-footer'),
  banner: document.getElementById('freeze-banner'),
  status: document.getElementById('status'),
  content: document.getElementById('content'),
  homeLink: document.getElementById('home-link'),
  seriesSubtitle: document.getElementById('series-subtitle'),
  seriesChart: document.getElementById('series-chart'),
  pollsTable: document.getElementById('polls-table'),
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

/**
 * Cells of the polls table: the latest `n` polls, newest first, as text (date, pollster, sponsor, sample
 * and one value per party; missing values as a dash).
 *
 * @param {object[]} polls `polls.data.polls` (ascending by date)
 * @param {string[]} parties party columns
 * @param {number} n maximum number of rows
 * @returns {string[][]} one array of cell texts per row
 */
export function tableRows(polls, parties, n = TABLE_ROWS) {
  // Reverse first so polls of the same date keep newest-first order under the stable sort.
  const newest = [...polls].reverse().sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  return newest.slice(0, n).map((poll) => [
    fmtDate(poll.date),
    poll.pollster ?? DASH,
    poll.sponsor ?? DASH,
    fmtInt(poll.sample_size),
    ...parties.map((name) => fmtNum(poll[name], 1)),
  ]);
}

/** Subtitle of the series section: the projected span (forecast) or the estimate date (nowcast). */
function seriesSubtitle(mode, asOf, when) {
  return mode === 'nowcast' ? `a ${fmtDate(asOf)}` : `${fmtDate(asOf)} → ${fmtDate(when)}`;
}

/** Table of the latest polls with a caption carrying the poll count and the CSV download link. */
function renderPollsTable(root, polls, {parties, catalog, nPolls, csvHref}) {
  const table = el('table');
  const caption = el('caption', `${fmtInt(nPolls)} sondeos en el ciclo · `);
  const download = el('a', 'Descargar CSV');
  download.href = csvHref;
  caption.append(download);
  table.append(caption);

  const head = el('tr');
  for (const label of ['Fecha', 'Casa', 'Patrocinador', 'Muestra']) {
    const th = el('th', label);
    th.scope = 'col';
    head.append(th);
  }
  for (const name of parties) {
    const th = el('th', name);
    th.scope = 'col';
    th.title = catalog.fullname(name);
    head.append(th);
  }
  const thead = el('thead');
  thead.append(head);
  table.append(thead);

  const body = el('tbody');
  for (const cells of tableRows(polls, parties)) {
    const row = el('tr');
    row.append(...cells.map((text) => el('td', text)));
    body.append(row);
  }
  table.append(body);
  root.replaceChildren(table);
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
  updateHeader(dom.header, {state});
  dom.homeLink.href = `/${location.search}`;
  showStatus('Cargando…');

  // Pin the run for the whole page load, so the parts of one paint never mix runs; the URL keeps `run`
  // only when the user pinned it.
  const {scope, mode} = state;
  const run = state.run || manifest.data.scopes?.[scope]?.latest || null;
  const [meta, series, polls, projection, vote] = await Promise.all([
    getForecast(scope, 'meta', {run}),
    getForecast(scope, 'series', {run}),
    getForecast(scope, 'polls', {run}),
    getForecast(scope, 'projection', {mode, run}),
    getForecast(scope, 'vote', {mode, run}),
  ]);
  if (id !== loadId) {
    return;
  }

  const catalog = new Catalog(meta.data);
  const parties = series.data.parties;
  const main = (catalog.bmaps.main || []).filter((name) => parties.includes(name));
  const tableParties = main.length ? main : parties;
  const asOf = meta.data.as_of;
  const when = vote.data.when;

  // Charts need a laid-out container: show the content before mounting them.
  dom.content.hidden = false;
  dom.seriesSubtitle.textContent = seriesSubtitle(mode, asOf, when);
  renderSeries(dom.seriesChart, {
    series: series.data,
    polls: polls.data,
    projection: projection.data,
    parties,
    selected: main.length ? main : null,
    colors: mapNames(parties, (name) => catalog.color(name)),
    fullnames: mapNames(parties, (name) => catalog.fullname(name)),
    asOf,
    when,
    anchor: meta.data.date_last || asOf,
  });
  renderPollsTable(dom.pollsTable, polls.data.polls || [], {
    parties: tableParties,
    catalog,
    nPolls: meta.data.n_polls,
    csvHref: apiBase() + forecastUrl(scope, 'polls', {format: 'csv', run}),
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
  renderHeader(dom.header, {pages: PAGES, active: ACTIVE, scopes: [], state: readState()});
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
    updateHeader(dom.header, {scopes, state: readState()});
    onStateChange(load);
    await load();
  } catch (error) {
    fail(error);
  }
}

// Only on the page itself: `tableRows` is also imported by checks with a stub document.
if (dom.seriesChart) {
  init();
}
