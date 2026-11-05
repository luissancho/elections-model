// Wiring of the seats of an event page: the majority calculator (its chip, bar and histogram), the distributions of
// the seats of each party or block (the ridges, open on demand), «Resultados por territorio» with the detail of one
// district and the vote fan. The simulations (`dist`) are not embedded in the page (too heavy: Luis, 2026-11-04):
// what does not need them is drawn at once, and the calculator and the distributions wait for `initial.dist_href`.
// The calculator downloads as a signed image (`share.js`) and its selection lives in the URL fragment, merged with
// the rest of the state of the page (`url-state.js`).
import {fmtInt} from '../format.js';
import {THEME} from '../charts/base.js';
import {renderFan} from '../charts/fan.js';
import {histogramOption, renderHistogram} from '../charts/histogram.js';
import {renderRidge, ridgeRowHeight} from '../charts/ridge.js';
import {ridgeBlocks, ridgeData} from '../kde.js';
import {renderStacked, stackedOption} from '../charts/stacked.js';
import {canGroup, regionRows, sortTerritory, territoryRows} from '../territory.js';
import {renderBody, renderHead, renderRegion} from './territory-dom.js';
import {
  coalitionCaption, coalitionLocationHash, coalitionSeats, coalitionText, loadDist, parseCoalition, summarize,
} from '../seats.js';
import {mapNames, markPills, showError, tryDraw} from './common.js';
import {fileSlug, wireDownload} from '../share.js';

const EMPTY_COALITION = 'Marca partidos para calcular su mayoría.';
const LOADING = 'Cargando las simulaciones…';
const LOAD_FAILED = 'No se han podido cargar las simulaciones.';
/** Heights of the charts in the image of the calculator: the bar as on the page, the histogram a little shorter. */
const BAR_IMAGE_HEIGHT = 120;
const HISTOGRAM_IMAGE_HEIGHT = 300;
/** Shared seat axis of the ridges, as a share of the chamber: parties up to 60%, blocks up to 80% (Luis, 2026-10-14). */
const PARTY_AXIS_SHARE = 0.6;
const BLOCK_AXIS_SHARE = 0.8;
/** The containers of the results tables of each view, shown one at a time. */
const VIEW_CONTAINERS = {partidos: 'seats-parties', bloques: 'seats-blocks'};

/** Order of the party rows of the ridges and the bars: by headline seats, then by mean seats. */
const bySeats = (a, b) => ((b.seats ?? 0) - (a.seats ?? 0)) || ((b.seats_mean ?? 0) - (a.seats_mean ?? 0));

/**
 * Data of the party bar of the central allocation (the headline seats), drawn by the majority calculator: the
 * parties with seats and a distribution, in the catalogue order, with their colours, full names and seat
 * differences against the last published election.
 *
 * @param {object} initial the page's initial data (`summary`, `deltas`)
 * @param {object} dist the dist part
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} colors `{party: colour}`
 * @returns {object} `names` (by seats), `rows` (`{name, seats}` in the catalogue order), `colors`, `fullnames`,
 *   `deltas` and `year`
 */
function partyBar(initial, dist, catalog, colors) {
  const partyRows = (initial.summary.parties || []).filter((row) => row.seats !== null && row.seats !== undefined);
  const names = [...partyRows].sort(bySeats).map((row) => row.name).filter((name) => dist.parties.includes(name));
  const deltas = initial.deltas || {parties: {}, year: null};
  return {
    names,
    rows: catalog.order(names).map((name) => ({name, seats: partyRows.find((row) => row.name === name).seats})),
    colors,
    fullnames: mapNames(names, (name) => catalog.fullname(name)),
    deltas: deltas.parties,
    year: deltas.year,
  };
}

/**
 * Wire the majority calculator: restore the selection of the URL fragment, then recompute the text
 * (`coalitionText`), the bar of the central allocation (the marked parties first, in colour, and the rest after
 * them in grey) and the histogram of the marked parties, marked at their central seats, with its caption
 * (`coalitionCaption`) on load and on every change. Only a change rewrites the fragment, and only its own key and,
 * with parties marked, the section token `escanos` (`coalitionLocationHash`), so a shared link opens on the
 * calculator and the state of the average survives. The download button (disabled without
 * parties marked) draws the chip, the bar and the histogram of the moment, the parties in the subtitle and in the
 * file name.
 *
 * @param {object} dist the dist part
 * @param {number} majority seats of the absolute majority
 * @param {object} bar `partyBar` of the page
 * @param {number} nSeats seats of the chamber
 */
function wireCoalition(dist, majority, bar, nSeats) {
  const fieldset = document.getElementById('coalition-parties');
  const result = document.getElementById('coalition-result');
  const chart = document.getElementById('coalition-chart');
  const barEl = document.getElementById('coalition-bar');
  const download = document.getElementById('coalition-download');
  if (!fieldset || !result || !chart) {
    return;
  }
  fieldset.disabled = false;
  const boxes = [...fieldset.querySelectorAll('input[type=checkbox][name=coalition]')];
  const fromHash = parseCoalition(window.location.hash, dist.parties);
  if (fromHash !== null) {
    for (const box of boxes) {
      box.checked = fromHash.includes(box.value);
    }
  }
  let current = {names: [], values: [], stats: null, central: 0};  // the selection drawn, the one the button exports

  /** Rows and options of the bar with `names` first, in colour, and the rest after them in grey. */
  const barOf = (names) => {
    const muted = bar.rows.map((row) => row.name).filter((name) => !names.includes(name));
    return [
      [...bar.rows.filter((row) => names.includes(row.name)), ...bar.rows.filter((row) => muted.includes(row.name))],
      {colors: bar.colors, fullnames: bar.fullnames, majority, total: nSeats, deltas: bar.deltas, year: bar.year, muted},
    ];
  };
  /** Options of the histogram of the sums marked at `central` with its caption. */
  const histogramOf = ({stats, central}) => ({
    majority, median: central, markLabel: `Reparto central: ${fmtInt(central)}`, lo: stats.lo, hi: stats.hi,
    color: THEME.barColor, bounds: true, markLabels: false, caption: coalitionCaption(stats, central),
  });

  function update() {
    markPills(fieldset);
    const names = boxes.filter((box) => box.checked).map((box) => box.value);
    const values = coalitionSeats(dist, names);
    const stats = summarize(values, majority);
    const central = bar.rows.filter((row) => names.includes(row.name)).reduce((sum, row) => sum + row.seats, 0);
    current = {names, values, stats, central};
    if (barEl) {
      tryDraw(() => renderStacked(barEl, ...barOf(names)), 'the majority bar');
    }
    result.classList.toggle('ok', Boolean(stats) && stats.pMajority >= 0.5);
    if (!stats) {
      result.textContent = EMPTY_COALITION;
      if (chart.__chart) {
        chart.__chart.clear();
      }
    } else {
      result.textContent = coalitionText(names.length, central, stats);
      renderHistogram(chart, values, histogramOf(current));
    }
    if (download) {
      download.disabled = !stats;
    }
    return names;
  }

  fieldset.addEventListener('change', () => {
    const names = update();
    const {pathname, search} = window.location;
    window.history.replaceState(null, '', pathname + search + coalitionLocationHash(window.location.hash, names));
  });
  update();
  wireDownload(download, null, showError, {
    charts: () => [
      {height: BAR_IMAGE_HEIGHT, option: () => stackedOption(...barOf(current.names))},
      {height: HISTOGRAM_IMAGE_HEIGHT, option: () => histogramOption(current.values, histogramOf(current))},
    ],
    note: () => ({text: result.textContent, ok: result.classList.contains('ok')}),
    texts: ({subtitle, file}) => {
      const {names} = current;
      if (!names.length) {
        return {};
      }
      // the parties marked after the name of the card: «Calculadora de mayorías · PP + VOX · Pronóstico para …»
      const [head, ...rest] = subtitle.split(' · ');
      return {subtitle: [head, names.join(' + '), ...rest].join(' · '), file: file.replace(/\.png$/, `-${fileSlug(names.join(' '))}.png`)};
    },
  });
}

/**
 * Set the calculator waiting for the simulations, or saying they could not be loaded: its pills disabled and the
 * text of its chip.
 *
 * @param {string} text what the chip says
 */
function holdCoalition(text) {
  const fieldset = document.getElementById('coalition-parties');
  const result = document.getElementById('coalition-result');
  if (fieldset) {
    fieldset.disabled = true;
  }
  if (result) {
    result.classList.remove('ok');
    result.textContent = text;
  }
}

/**
 * Wire the distributions of the seats card: the button «Ver las distribuciones» (revealed here, once the
 * simulations are there) opens and closes the ridges of the view shown. Parties are ordered by headline seats (then
 * by mean, like `Simulator.plot_dist_kde`); blocks the same way, their simulations being the sum of their parties.
 *
 * @param {object} initial the page's initial data (`meta`, `summary`)
 * @param {object} dist the dist part
 * @param {Catalog} catalog party catalogue of the run
 * @param {object} colors `{party: colour}`
 * @param {function} view the view shown: `() => 'partidos'|'bloques'`
 * @returns {{redraw: function}} `redraw(view)` draws the ridges of `view` when they are open
 */
function wireRidges(initial, dist, catalog, colors, view) {
  const {meta, summary} = initial;
  const majority = meta.majority ?? summary.majority;
  const nSeats = meta.n_seats ?? summary.n_seats;
  const parties = partyBar(initial, dist, catalog, colors);
  const blockRows = (summary.blocks || []).filter((row) => row.seats !== null && row.seats !== undefined);
  const blockParties = catalog.bmaps.blocks && !Array.isArray(catalog.bmaps.blocks) ? catalog.bmaps.blocks : {};
  const views = {
    partidos: {
      colors,
      fullnames: parties.fullnames,
      ridges: () => ridgeData(dist, parties.names, {
        majority,
        range: [0, Math.round(nSeats * PARTY_AXIS_SHARE)],
        central: Object.fromEntries(parties.rows.map((row) => [row.name, row.seats])),
      }),
    },
    bloques: {
      colors: mapNames(blockRows.map((row) => row.name), (name) => catalog.blockColor(name)),
      fullnames: {},
      ridges: () => ridgeBlocks(dist, [...blockRows].sort(bySeats)
        .map((row) => ({name: row.name, parties: blockParties[row.name] || []})),
      {
        majority,
        range: [0, Math.round(nSeats * BLOCK_AXIS_SHARE)],
        central: Object.fromEntries(blockRows.map((row) => [row.name, row.seats])),
      }),
    },
  };
  const ridgeEl = document.getElementById('ridge-chart');
  const toggle = document.getElementById('ridge-toggle');
  const cache = {};
  let open = false;  // the distributions open on demand (Luis, 2026-10-30)

  /** Draw the ridges of `name` (computed once per view). */
  function draw(name) {
    const chosen = views[name];
    if (!ridgeEl || !open || !chosen) {
      return;
    }
    cache[name] = cache[name] || chosen.ridges();
    if (cache[name].length) {
      tryDraw(() => renderRidge(ridgeEl, cache[name], {
        colors: chosen.colors, majority, fullnames: chosen.fullnames, rowHeight: ridgeRowHeight(window.innerWidth),
      }), `the ${name} distributions`);
    }
  }

  if (toggle && ridgeEl) {
    toggle.hidden = false;
    toggle.addEventListener('click', () => {
      open = !open;
      ridgeEl.hidden = !open;
      toggle.setAttribute('aria-expanded', String(open));
      toggle.textContent = open ? 'Ocultar las distribuciones' : 'Ver las distribuciones';
      draw(view());
    });
  }
  return {redraw: draw};
}

const CAPTION = {
  esc: 'Reparto central: D\'Hondt en cada circunscripción sobre el voto central; cada fila suma sus escaños. '
    + 'Pasa el ratón para ver el rango del 95%.',
  ccaa: 'Reparto central: D\'Hondt en cada circunscripción sobre el voto central; cada fila suma los escaños de '
    + 'las provincias de la comunidad.',
  pct: 'Voto estimado en cada territorio; en CCAA, la media ponderada por los votos de la elección anterior.',
};

/**
 * Link of a district of the territory table, as the template writes it: the event page on the territory section
 * with that district (what the form sends without JavaScript; with it, the click is caught by `wireRegion`). The
 * `#territorio` fragment lands a link opened apart (a new window, a modified click) on the section.
 *
 * @param {object} state the `state` of the initial data (`scope`, `mode`, `run`, `pinned`)
 * @param {number} id district id
 * @returns {string} the URL
 */
function regionHref({scope, mode, run, pinned}, id) {
  return `/${scope}?mode=${mode}${pinned ? `&run=${run}` : ''}&region=${id}&seccion=territorio#territorio`;
}

/**
 * Wire "Resultados por territorio": rebuild the table of the central allocation from the embedded `districts`,
 * `scenario` and `summary` for the chosen level (CCAA or province) and mode (seats or vote), sortable from its
 * headers. The server draws the province-by-seats table; this replaces it and shows the controls.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function wireTerritory(initial, catalog) {
  const controls = document.getElementById('territory-controls');
  const table = document.getElementById('district-table');
  if (!controls || !table || !initial.districts || !initial.scenario) {
    return;
  }
  const levels = document.getElementById('territory-level');
  const state = {sort: {key: 'name', asc: true}};
  const checked = (name, fallback) => {
    const input = document.querySelector(`#territory-controls input[name="${name}"]:checked`);
    return input ? input.value : fallback;
  };
  const hrefOf = (row) => (state.level === 'prov' && row.id !== null ? regionHref(initial.state, row.id) : null);

  const render = (focusKey = null) => {
    state.level = levels && canGroup(initial.districts.regions || []) ? checked('level', 'ccaa') : 'prov';
    const mode = checked('tmode', 'esc');
    const data = territoryRows(initial.districts, initial.scenario, initial.summary,
      {level: state.level, mode});
    if (state.sort.key !== 'name' && state.sort.key !== 'total' && !data.parties.includes(state.sort.key)) {
      state.sort = {key: 'name', asc: true};
    }
    const caption = table.querySelector('caption');
    if (caption) {
      caption.textContent = mode === 'esc' && state.level === 'ccaa' ? CAPTION.ccaa : CAPTION[mode];
    }
    const sorted = {...data, rows: sortTerritory(data.rows, state.sort.key, state.sort.asc)};
    renderHead(table.tHead, data.parties, catalog, state.sort, state.level === 'ccaa' ? 'CCAA' : 'Provincia');
    renderBody(table.tBodies[0], sorted, mode, catalog, hrefOf);
    if (focusKey) {
      const button = table.querySelector(`th[data-key="${CSS.escape(focusKey)}"] .sort`);
      if (button) {
        button.focus();
      }
    }
  };

  controls.addEventListener('change', () => render());
  table.tHead.addEventListener('click', (event) => {
    const th = event.target.closest('th[data-key]');
    if (!th) {
      return;
    }
    const key = th.dataset.key;
    state.sort = key === state.sort.key ? {key, asc: !state.sort.asc} : {key, asc: key === 'name'};
    render(key);
  });
  controls.hidden = false;
  render();
}

/**
 * Wire the detail of one district without reloading the page: changing the select of `#district-form`, or
 * following the link of a province of the territory table, draws its detail in `#region-table` from the embedded
 * `districts` (`regionRows`, `renderRegion`), writes `region=N` in the query (keeping the fragment) and scrolls to
 * the detail. Without JavaScript the form and the links load the page on that district.
 *
 * @param {object} initial the page's initial data (`districts`)
 * @param {Catalog} catalog party catalogue of the run
 */
function wireRegion(initial, catalog) {
  const form = document.getElementById('district-form');
  const select = document.getElementById('region');
  const table = document.getElementById('region-table');
  const regions = (initial.districts || {}).regions || [];
  if (!form || !select || !table || !regions.length) {
    return;
  }

  /** Draw the detail of district `id`; false when there is no such district. */
  function show(id) {
    const region = regions.find((item) => item.id === id);
    if (!region) {
      return false;
    }
    renderRegion(table, regionRows(initial.districts, id), region, catalog);
    select.value = String(id);
    const params = new URLSearchParams(window.location.search);
    params.set('region', String(id));
    const {pathname, hash} = window.location;
    window.history.replaceState(null, '', `${pathname}?${params}${hash}`);
    table.scrollIntoView({block: 'nearest', behavior: 'smooth'});
    return true;
  }

  form.classList.add('js');  // hides the submit button (`form.js .nojs-only`)
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    show(Number(select.value));
  });
  select.addEventListener('change', () => show(Number(select.value)));
  const territory = document.getElementById('district-table');
  if (territory) {
    territory.addEventListener('click', (event) => {
      const link = event.target.closest('a[href]');
      if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }
      const id = Number(new URL(link.href, window.location.href).searchParams.get('region'));
      if (id && show(id)) {
        event.preventDefault();
      }
    });
  }
}

/**
 * Draw the seats of an event page: «Resultados por territorio» with the detail of a district and the vote fan at
 * once; then, once the simulations arrive (one request to `initial.dist_href`), the majority calculator and the
 * button of the distributions. Meanwhile the calculator says «Cargando las simulaciones…»; if the request fails, or
 * the calculator cannot be wired with what came back, it says they could not be loaded, and the rest of the page
 * keeps working.
 *
 * @param {object} initial the page's initial data (`state`, `meta`, `summary`, `deltas`, `fan`, `districts`,
 *   `scenario` and `dist_href`)
 * @param {Catalog} catalog party catalogue of the run
 * @param {function} [view] the view of the results shown: `() => 'partidos'|'bloques'`
 * @returns {{redraw: function}} `redraw(view)`, for the Partidos/Bloques switch: shows the results table of `view`
 *   (`#seats-parties` or `#seats-blocks`) and redraws its distributions when they are open
 */
export function wireSeatsPage(initial, catalog, view = () => 'partidos') {
  const {state, meta, summary} = initial;
  const fan = initial.fan || {};
  const majority = meta.majority ?? summary.majority;
  const nSeats = meta.n_seats ?? summary.n_seats;
  const fanNames = catalog.order([...new Set((fan.rows || []).map((row) => row.name).filter(Boolean))]);
  const colorsOf = (names) => mapNames(names, (name) => catalog.color(name));

  /** Show the results table of `name` and hide the other one. */
  const showTable = (name) => {
    for (const [key, id] of Object.entries(VIEW_CONTAINERS)) {
      const container = document.getElementById(id);
      if (container) {
        container.hidden = key !== name;
      }
    }
  };

  tryDraw(() => wireTerritory(initial, catalog), 'the territory table');
  tryDraw(() => wireRegion(initial, catalog), 'the district detail');
  const fanEl = document.getElementById('fan-chart');
  if (fanEl) {
    const main = (catalog.bmaps.main || []).filter((name) => fanNames.includes(name));
    tryDraw(() => renderFan(fanEl, fan, {
      parties: fanNames,
      selected: main.length ? main : null,
      colors: colorsOf(fanNames),
      markAt: state.mode === 'nowcast' ? 0 : meta.horizon_max,
      markLabel: state.mode === 'nowcast' ? 'Hoy' : 'Elección',
    }), 'the fan');
  }
  showTable(view());

  let ridges = null;
  let failed = false;
  /** Say the simulations could not be loaded (once): the request failed, or the calculator could not use them. */
  const fail = (error) => {
    if (failed) {
      return;
    }
    failed = true;
    console.error('Could not load the simulations:', error);
    holdCoalition(LOAD_FAILED);
  };
  holdCoalition(LOADING);
  // asked for once per page load; an exception while wiring the calculator ends in the same notice as a failed
  // request, not in a chip stuck on «Cargando…»
  loadDist(initial.dist_href).then((dist) => {
    const names = [...new Set([...dist.parties, ...(summary.parties || []).map((row) => row.name)])];
    const colors = colorsOf(names);
    tryDraw(() => {
      ridges = wireRidges(initial, dist, catalog, colors, view);
    }, 'the distributions');
    const bar = partyBar(initial, dist, catalog, colors);
    if (!tryDraw(() => wireCoalition(dist, majority, bar, nSeats), 'the majority calculator')) {
      fail(new Error('the majority calculator could not be wired'));
    }
  }).catch(fail);

  return {
    redraw(name) {
      showTable(name);
      if (ridges) {
        ridges.redraw(name);
      }
    },
  };
}
