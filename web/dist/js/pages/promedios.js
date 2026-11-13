// The event page (`/{scope}`, «Encuestas»): the poll average with its window, party and block views (wired by
// `series-wire.js`, shared with the detail of a past election), the hemicycle with the one Partidos/Bloques switch of
// the results, the seats (`seats-wire.js`: calculator, distributions, territory and fan), the card «Escaños que
// bailan» (`seat-odds-wire.js`: the name of its image, the dropdown of the phones and the scroll to the block of a
// district), the evolution of the updates in vote or seats, one house against the average and the filterable, sortable
// table of all the polls, drawn from the data embedded by the server. The state of the controls lives in the URL
// fragment, each control rewriting only its own keys (`url-state.js`); the index of the sections scrolls in place and
// a `seccion` of the query (the old seats URLs) scrolls to its section on load.
import {Catalog} from '../catalog.js';
import {parseSeriesHash} from '../blocks.js';
import {hemicycleOption, renderHemicycle} from '../charts/hemicycle.js';
import {chartZoom, renderSeries, seriesImageOption, setWindow} from '../charts/series.js';
import {fmtInt, fmtPct} from '../format.js';
import {sparklineSvg} from '../sparkline.js';
import {mapNames, markSegments, mountWhenReady, readInitial, showError, tryDraw, wireControls} from './common.js';
import {markLegend} from './legend-dom.js';
import {wireEvolution} from './evolution-wire.js';
import {wireHouseChanges} from './house-changes-wire.js';
import {wirePollsTable} from './polls-wire.js';
import {wireSeatsPage} from './seats-wire.js';
import {wireSeatOdds} from './seat-odds-wire.js';
import {WINDOWS, chosenNames, fillLegend, seriesViews, viewCaption, wireAverage} from './series-wire.js';
import {fileSlug, tableImage, wireDownload} from '../share.js';
import {dropParam, sectionTarget, withSection} from '../url-state.js';

/** Size of the hemicycle image: landscape, the hemicycle at the left and its results table at the right. */
const HEMICYCLE_IMAGE = {width: 1200, chartHeight: 360};

/**
 * Draw the hemicycle and wire the one Partidos/Bloques switch of the results (`#results-view`): it draws the
 * hemicycle of the view and tells `onView`, which shows its results table and redraws its distributions when they
 * are open (`wireSeatsPage`, wired apart by `paint` so that a failure here does not stop the seats). The download
 * button draws the hemicycle of the view shown beside its results table, as a signed image, with the parties of each
 * block as its caption in the block view (on the page they head the table of blocks, server-rendered).
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 * @param {function} onView called with the view chosen in the switch: `(name) => void`
 * @returns {{view: function}} `view()`, the view shown: `'partidos'|'bloques'`
 */
function paintResults(initial, catalog, onView) {
  const {meta, headline, summary} = initial;
  const mode = initial.state.mode;
  const current = headline[mode];
  const names = [...new Set([
    ...catalog.parties.map((party) => party.name),
    ...current.parties.map((party) => party.name),
    ...summary.parties.map((row) => row.name),
  ])];
  const colors = mapNames(names, (name) => catalog.color(name));
  const nSeats = meta.n_seats ?? summary.n_seats;
  const majority = meta.majority ?? summary.majority;
  const blocks = (summary.blocks || []).filter((row) => row.seats !== null && row.seats !== undefined);
  const members = catalog.bmaps.blocks && !Array.isArray(catalog.bmaps.blocks) ? catalog.bmaps.blocks : null;
  // the seat differences against the last published election, written after the seats of each arc
  const deltas = initial.deltas || {parties: {}, blocks: {}, year: null};
  const views = {
    partidos: {
      seats: summary.totals ?? Object.fromEntries(current.parties.map((party) => [party.name, party.seats])),
      colors,
      deltas: deltas.parties,
      order: (list) => catalog.order(list),
      label: 'Hemiciclo con los escaños previstos por partido',
      table: '#seats-parties table',
    },
    bloques: {
      seats: Object.fromEntries(blocks.map((row) => [row.name, row.seats])),
      colors: mapNames(blocks.map((row) => row.name), (name) => catalog.blockColor(name)),
      deltas: deltas.blocks,
      order: (list) => catalog.orderBlocks(list),
      label: 'Hemiciclo con los escaños previstos por bloque',
      table: '#seats-blocks table',
      members,
    },
  };
  const el = document.getElementById('hemicycle');
  let view = 'partidos';
  let shown = views.partidos;  // the view drawn, the one the download button exports

  /** Draw the hemicycle of `name`. */
  function draw(name) {
    view = name;
    shown = views[name];
    el.setAttribute('aria-label', shown.label);
    tryDraw(() => renderHemicycle(el, shown.seats, {
      colors: shown.colors, order: shown.order, nSeats, majority, deltas: shown.deltas, year: deltas.year,
    }), 'the hemicycle');
  }

  draw('partidos');
  wireDownload(document.getElementById('hemicycle-download'), (box) => hemicycleOption(shown.seats, {
    colors: shown.colors, order: shown.order, nSeats, majority, deltas: shown.deltas, year: deltas.year,
    width: box.width, height: box.height,
  }), showError, {
    ...HEMICYCLE_IMAGE, side: () => tableImage(document.querySelector(shown.table)), caption: () => viewCaption(shown),
  });
  const fieldset = document.getElementById('results-view');
  if (fieldset) {
    fieldset.hidden = false;
    fieldset.addEventListener('change', (event) => {
      if (event.target.matches('input[name=rview]') && views[event.target.value]) {
        draw(event.target.value);
        onView(event.target.value);
        markSegments(fieldset);
      }
    });
  }
  return {view: () => view};
}

/**
 * Draw the evolution of the updates with its % Voto/Escaños switch: the vote by default, the seats of the central
 * allocation on the other side.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function paintEvolution(initial, catalog) {
  const nSeats = initial.meta.n_seats ?? initial.summary.n_seats;
  // the same parties, in the same order, as the legend of the average (Luis, 2026-10-30)
  wireEvolution(initial, catalog, {
    parties: initial.series.parties,
    metrics: {
      pct: {field: 'pct', formatter: fmtPct, total: 100},
      seats: {field: 'seats', formatter: fmtInt, total: nSeats},
    },
  });
}

/**
 * Draw the house chart and wire its own controls (view, window, house selector and legend) and its download
 * button (the chart as shown, its subtitle and file named after the house chosen). Its state starts as the
 * average's at load and never touches the URL hash.
 *
 * @param {object} initial the page's initial data
 * @param {object} views `seriesViews` of the page
 * @param {object} start `{view, window, selected, layers}` of the average at load (its layers are not kept: the
 *   house chart always draws its polls over the average)
 * @param {function} render draws a chart: `(el, polls, chartState) => chart`
 * @param {function} imageOf the option of its image: `(polls, chartState, zoom) => option`
 */
function wireHouse(initial, views, start, render, imageOf) {
  const {meta, polls} = initial;
  const anchor = meta.date_last || meta.as_of;
  const houseEl = document.getElementById('pollster-chart');
  const legendEl = document.getElementById('pollster-legend');
  const select = document.getElementById('pollster-select');
  const controls = document.getElementById('pollster-controls');
  const windowSet = document.getElementById('pollster-window');
  const viewSet = document.getElementById('pollster-view');
  const state = {...start, layers: 'ambos'};
  let chart = null;

  /** Polls of the chosen house, in the current view. */
  function housePolls() {
    const house = select ? select.value : '';
    const chosen = (polls.polls || []).filter((poll) => poll.pollster === house);
    return views[state.view].toPolls({parties: polls.parties, polls: chosen});
  }

  /** Draw the chart, fill its pill legend and, the first time, listen to the legend selection. */
  function draw() {
    const drawn = render(houseEl, housePolls(), state);
    if (!chart) {
      chart = drawn;
      chart.on('legendselectchanged', (event) => {
        state.selected = chosenNames(views[state.view], event.selected);
        if (legendEl) {
          markLegend(legendEl, event.selected);
        }
      });
    }
    fillLegend(legendEl, views[state.view], state.selected, chart, initial.inks || {});
  }

  for (const input of document.querySelectorAll('#pollster-window input[name=pwindow]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#pollster-view input[name=pview]')) {
    input.checked = input.value === state.view;
  }
  markSegments();
  tryDraw(draw, 'the house against the average');
  wireDownload(document.getElementById('pollster-download'), () => imageOf(housePolls(), state, chartZoom(chart)), showError, {
    texts: ({subtitle, file}) => {
      const house = select ? select.value : '';
      return house ? {subtitle: `${house} ${subtitle}`, file: file.replace(/\.png$/, `-${fileSlug(house)}.png`)} : {};
    },
    caption: () => viewCaption(views[state.view]),
  });

  if (controls) {
    controls.hidden = false;
  }
  if (windowSet) {
    windowSet.hidden = false;
    windowSet.addEventListener('click', (event) => {
      if (event.target.matches('input[name=pwindow]') && chart) {
        state.window = event.target.value;
        const current = views[state.view];
        setWindow(chart, state.window, {series: current.series, projection: current.projection, anchor});
      }
    });
  }
  if (viewSet) {
    viewSet.hidden = false;
    viewSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=pview]') && views[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        tryDraw(draw, 'the house against the average');
      }
    });
  }
  if (select) {
    // `renderSeries` redraws the same ECharts instance; the window, view and legend choices are kept.
    select.addEventListener('change', () => tryDraw(draw, 'the house against the average'));
  }
}

/**
 * Wire the poll average and, when the page has one, the house chart.
 *
 * @param {object} initial the page's initial data
 * @param {Catalog} catalog party catalogue of the run
 */
function wireSeries(initial, catalog) {
  const {meta, vote} = initial;
  const views = seriesViews(initial, catalog);
  const anchor = meta.date_last || meta.as_of;
  const state = parseSeriesHash(window.location.hash, {
    parties: views.partidos.parties,
    blocks: views.bloques ? views.bloques.parties : [],
    windows: WINDOWS,
  });
  const start = {...state};

  /** Data of a chart (`seriesOption`) with the polls and the `{view, window, selected, layers}` of a chart; the
   * ECharts legend stays hidden and selectable, driven by the pill legend. */
  function dataOf(pollsOfChart, chartState) {
    const current = views[chartState.view];
    return {
      series: current.series,
      polls: pollsOfChart,
      projection: current.projection,
      parties: current.parties,
      selected: chartState.selected ?? current.defaults,
      colors: current.colors,
      asOf: meta.as_of,
      when: vote.when,
      window: chartState.window,
      anchor,
      legend: 'hidden',
      layers: chartState.layers,
      breakdown: current.breakdown || null,
      marks: current.marks || [],
    };
  }

  /** Draw the chart `el` with the polls and the state of a chart. */
  function render(el, pollsOfChart, chartState) {
    return renderSeries(el, dataOf(pollsOfChart, chartState));
  }

  /** Option of the image of a chart (`seriesImageOption`) with its polls, state and zoom on screen. */
  const imageOf = (pollsOfChart, chartState, zoom) => seriesImageOption(dataOf(pollsOfChart, chartState), zoom);

  wireAverage(views, state, render, {anchor, inks: initial.inks || {}, imageOf});
  if (document.getElementById('pollster-chart')) {
    wireHouse(initial, views, start, render, imageOf);
  }
}

/** Days of the average the sparkline of a party card covers, and the sampling step. */
const SPARK_DAYS = 365;
const SPARK_STEP = 7;

/**
 * Draw the sparkline of every party card (`article.pc[data-name]`): the mean of the last year of the average,
 * one point a week.
 *
 * @param {object} initial the page's initial data (`series.mean`)
 */
function paintCards(initial) {
  const mean = (initial.series && initial.series.mean) || {};
  for (const card of document.querySelectorAll('article.pc[data-name]')) {
    const svg = card.querySelector('svg.spark');
    const values = mean[card.dataset.name];
    if (!svg || !values) {
      continue;
    }
    const recent = values.slice(-SPARK_DAYS);
    const sampled = recent.filter((_, i) => i % SPARK_STEP === 0 || i === recent.length - 1);
    sparklineSvg(svg, sampled);
  }
}

/**
 * Wire the arrows of the party carousel (`#pcards`): shown when the strip overflows, each scrolls one view, the one
 * at an end disabled. Touch and trackpads scroll the strip on their own.
 */
function wireCardsCarousel() {
  const strip = document.getElementById('pcards');
  const prev = document.querySelector('.pcarousel .pc-arrow.prev');
  const next = document.querySelector('.pcarousel .pc-arrow.next');
  if (!strip || !prev || !next) {
    return;
  }
  const update = () => {
    const overflow = strip.scrollWidth > strip.clientWidth + 1;
    prev.hidden = !overflow;
    next.hidden = !overflow;
    prev.disabled = strip.scrollLeft <= 0;
    next.disabled = strip.scrollLeft + strip.clientWidth >= strip.scrollWidth - 1;
  };
  prev.addEventListener('click', () => strip.scrollBy({left: -strip.clientWidth, behavior: 'smooth'}));
  next.addEventListener('click', () => strip.scrollBy({left: strip.clientWidth, behavior: 'smooth'}));
  strip.addEventListener('scroll', update, {passive: true});
  new ResizeObserver(update).observe(strip);
  update();
}

/** Ids of the sections of the page (`EVENT_SECTIONS` that it renders). */
const sectionIds = () => [...document.querySelectorAll('section.event-section[id]')].map((section) => section.id);

/**
 * Ids the URL can ask the page to land on: its sections and, when the page has it, the card «Escaños que bailan»,
 * where the addresses of the page it replaced arrive (`#seat-odds`).
 */
const landingIds = () => sectionIds().concat(document.getElementById('seat-odds') ? ['seat-odds'] : []);

/**
 * Scroll to the section the URL asks for (`seccion` of the query, which the old seats URLs bring, or a loose token
 * of the fragment such as `#resultados&coalition=PP,VOX`, which the browser does not scroll to by itself), or to the
 * card «Escaños que bailan» when the fragment asks for it (`landingIds`: the territory table above it has just been
 * redrawn; Chrome holds the anchor through that by itself, and this covers a browser that does not), and drop
 * `seccion` from the query, keeping the fragment.
 */
function scrollToSection() {
  const {pathname, search, hash} = window.location;
  const target = sectionTarget(search, hash, landingIds());
  if (target) {
    document.getElementById(target).scrollIntoView();
  }
  if (new URLSearchParams(search).has('seccion')) {
    window.history.replaceState(null, '', pathname + dropParam(search, 'seccion') + hash);
  }
}

/**
 * Catch the links of the index of the sections (`.event-index`): scroll to the section and write its anchor in the
 * fragment without losing the state of the controls (`withSection`). Without JavaScript they are plain anchors.
 */
function wireSectionIndex() {
  const ids = sectionIds();
  for (const link of document.querySelectorAll('.event-index a[href^="#"]')) {
    const id = link.getAttribute('href').slice(1);
    if (!ids.includes(id)) {
      continue;
    }
    link.addEventListener('click', (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }
      event.preventDefault();
      document.getElementById(id).scrollIntoView({behavior: 'smooth'});
      const {pathname, search, hash} = window.location;
      window.history.replaceState(null, '', pathname + search + withSection(hash, id));
    });
  }
}

/** Draw the page from the embedded data. */
function paint() {
  const initial = readInitial();
  // the catalogue the server builds for the page (the run's parties, those of the polls table and those of the
  // last election, named and coloured by the published catalogue), with the bmaps of the run
  const catalog = new Catalog(initial.catalog || initial.meta);
  tryDraw(() => paintCards(initial), 'the party cards');
  tryDraw(wireCardsCarousel, 'the party carousel');
  tryDraw(() => wireSeries(initial, catalog), 'the poll average and its controls');
  tryDraw(() => wireHouseChanges(document, showError), 'the changes of each house');
  // the hemicycle and the seats in their own calls: a failure in one does not stop the other, and the seats read the
  // view of the switch (Partidos until the hemicycle has been drawn)
  let results = null;
  let seats = null;
  tryDraw(() => {
    results = paintResults(initial, catalog, (name) => {
      if (seats) {
        seats.redraw(name);
      }
    });
  }, 'the results');
  tryDraw(() => {
    seats = wireSeatsPage(initial, catalog, () => (results ? results.view() : 'partidos'));
  }, 'the seats');
  tryDraw(() => paintEvolution(initial, catalog), 'the evolution');
  tryDraw(() => wirePollsTable(initial.polls.polls, catalog), 'the polls table');
  tryDraw(wireSectionIndex, 'the index of the sections');
  // the card of the seats that dance before the scroll: under 720 px it reveals its dropdown of districts, which would
  // push the sections below it down once the page had landed on one of them
  tryDraw(wireSeatOdds, 'the seats that dance');
  tryDraw(scrollToSection, 'the section of the URL');
}

wireControls();
mountWhenReady(paint);
