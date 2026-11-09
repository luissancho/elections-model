// Page of a poll (`/{scope}/sondeos/{date}/{pollster}[/{sponsor}]`): the card to share, with the bars of the vote of
// the poll and the mark of the average of the day on each one and, when the house published seats, the hemicycle of its
// seats (the midpoint of each range, its arc labelled with the range); the chart of the house in the cycle against the
// average, the house fixed, with this poll ringed, its own window and Partidos/Bloques view and its pill legend (as the
// house chart of the event page, `promedios.js`, but its state never touches the URL); and the downloads of the chart
// (`share.js`) and of the card (`card-image.js`). Everything is drawn from the data embedded by the server; the pure
// logic of the card lives in `poll-card.js`.
import {Catalog, OTHERS_COLOR} from '../catalog.js';
import {renderBars} from '../charts/bars.js';
import {mountChart} from '../charts/base.js';
import {renderHemicycle} from '../charts/hemicycle.js';
import {chartZoom, seriesImageOption, seriesOption, setWindow} from '../charts/series.js';
import {wireCardImages} from '../card-image.js';
import {fmtPct} from '../format.js';
import {highlightSeries, pollSeats, voteRows} from '../poll-card.js';
import {wireDownload} from '../share.js';
import {mapNames, markSegments, mountWhenReady, readInitial, showError, tryDraw, wireControls} from './common.js';
import {markLegend} from './legend-dom.js';
import {chosenNames, fillLegend, seriesViews, viewCaption} from './series-wire.js';

/** Name of the row of the rest of the vote in `voteRows`, painted grey. */
const OTHERS_ROW = 'Otros';

/**
 * Draw the bars of the vote of the poll (`#poll-bars`): one per party with a value, from the largest, the share
 * written after the bar, the average of the day as a capped zero-length interval on it (none without an average),
 * named «Promedio del día» in the tooltip, and a grey «Otros» row with the rest up to 100 (`voteRows`).
 *
 * @param {object} initial the page's initial data (`poll`, `polls.parties`, `average`)
 * @param {Catalog} catalog party catalogue of the page
 */
function paintBars(initial, catalog) {
  const el = document.getElementById('poll-bars');
  if (!el) {
    return;
  }
  const rows = voteRows(initial.poll, initial.polls.parties || [], initial.average || {});
  const names = rows.map((row) => row.name);
  renderBars(el, rows, {
    value: 'value',
    lo: 'avg',
    hi: 'avg',
    colors: mapNames(names, (name) => (name === OTHERS_ROW ? OTHERS_COLOR : catalog.color(name))),
    fullnames: mapNames(names, (name) => catalog.fullname(name)),
    formatter: (x) => fmtPct(x),
    axisFormatter: (x) => fmtPct(x, 0),
    // the interval is the zero-length mark of the average: the tooltip names it as such
    intervalText: (row) => (row.lo === null ? null : `Promedio del día: ${fmtPct(row.lo)}`),
  });
}

/**
 * Draw the hemicycle of the seats the house published with the poll (`#poll-hemi`, only rendered when it has them):
 * each party with the midpoint of its range, its arc labelled with the range, and a grey «Sin asignar» arc with the
 * seats the midpoints leave out of the chamber (`pollSeats`). The majority and the total are those of the run.
 *
 * @param {object} initial the page's initial data (`seats`, `n_seats`, `majority`)
 * @param {Catalog} catalog party catalogue of the page
 */
function paintSeats(initial, catalog) {
  const el = document.getElementById('poll-hemi');
  if (!el || !initial.seats) {
    return;
  }
  const {seats, labels, rest} = pollSeats(initial.seats, initial.n_seats);
  const names = Object.keys(seats);
  renderHemicycle(el, seats, {
    colors: mapNames(names, (name) => catalog.color(name)),
    order: (list) => catalog.order(list),
    nSeats: initial.n_seats,
    majority: initial.majority,
    fullnames: mapNames(names, (name) => catalog.fullname(name)),
    labels,
    rest,
  });
}

/**
 * Draw the chart of the house in the cycle (`#poll-house-chart`) and wire its controls (window, Partidos/Bloques view
 * and pill legend) and its download button: the polls of the house as points over the average of all the houses,
 * the line, projection and end bar of the mode of the page, and this poll ringed (`highlightSeries`, one hollow
 * circle per party, named as its line so the legend hides them together). The state starts at the whole cycle in the
 * party view with the main parties and never touches the URL.
 *
 * @param {object} initial the page's initial data (`meta`, `vote`, `series`, `polls` (the house's only),
 *   `projection`, `poll`, `inks`)
 * @param {Catalog} catalog party catalogue of the page
 */
function wireHouse(initial, catalog) {
  const el = document.getElementById('poll-house-chart');
  if (!el) {
    return;
  }
  const {meta, vote, polls, poll} = initial;
  const views = seriesViews(initial, catalog);
  const anchor = meta.date_last || meta.as_of;
  const legendEl = document.getElementById('house-legend');
  const controls = document.getElementById('house-controls');
  const windowSet = document.getElementById('house-window');
  const viewSet = document.getElementById('house-view');
  const state = {view: 'partidos', window: 'all', selected: null, layers: 'ambos'};
  let chart = null;

  /** Data of the chart (`seriesOption`) in the current state; the ECharts legend stays hidden and selectable,
   * driven by the pill legend. */
  function dataOf() {
    const current = views[state.view];
    return {
      series: current.series,
      polls: current.polls,
      projection: current.projection,
      parties: current.parties,
      selected: state.selected ?? current.defaults,
      colors: current.colors,
      asOf: meta.as_of,
      when: vote ? vote.when : undefined,
      window: state.window,
      anchor,
      legend: 'hidden',
      layers: state.layers,
      breakdown: current.breakdown || null,
    };
  }

  /** The rings over this poll in the current view (its parties summed into blocks in the block view). */
  function ring() {
    const current = views[state.view];
    const [shown] = current.toPolls({parties: polls.parties || [], polls: [poll]}).polls;
    return highlightSeries(shown, current.parties, current.colors);
  }

  /** `option` with the rings over this poll added after its series. */
  const withRing = (option) => ({...option, series: [...option.series, ...ring()]});

  /** Option of the chart on screen: `seriesOption` of the state with the rings. */
  const optionWithRing = () => withRing(seriesOption(dataOf()));

  /** Option of the image of the chart: `seriesImageOption` of the state at the zoom on screen, with the rings. */
  const imageWithRing = () => withRing(seriesImageOption(dataOf(), chartZoom(chart)));

  /** Draw the chart, fill its pill legend and, the first time, listen to the legend selection. */
  function draw() {
    const drawn = mountChart(el, optionWithRing());
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

  for (const input of document.querySelectorAll('#house-window input[name=hwindow]')) {
    input.checked = input.value === state.window;
  }
  for (const input of document.querySelectorAll('#house-view input[name=hview]')) {
    input.checked = input.value === state.view;
  }
  markSegments();
  tryDraw(draw, 'the house in the cycle');
  wireDownload(document.getElementById('house-download'), imageWithRing, showError, {
    caption: () => viewCaption(views[state.view]),
  });

  // The controls only work with JavaScript, so the server renders them hidden.
  if (controls) {
    controls.hidden = false;
  }
  if (windowSet) {
    // `click` rather than `change`: clicking the option already checked after dragging the slider must restore
    // its window too.
    windowSet.addEventListener('click', (event) => {
      if (event.target.matches('input[name=hwindow]') && chart) {
        state.window = event.target.value;
        const current = views[state.view];
        setWindow(chart, state.window, {series: current.series, projection: current.projection, anchor, result: current.result});
      }
    });
  }
  if (viewSet) {
    viewSet.addEventListener('change', (event) => {
      if (event.target.matches('input[name=hview]') && views[event.target.value]) {
        state.view = event.target.value;
        state.selected = null;
        tryDraw(draw, 'the house in the cycle');
      }
    });
  }
}

/** Draw the page from the embedded data, each part on its own so that a failure does not stop the others. */
function paint() {
  const initial = readInitial();
  const catalog = new Catalog(initial.catalog || initial.meta);
  tryDraw(() => paintBars(initial, catalog), 'the bars of the poll');
  tryDraw(() => paintSeats(initial, catalog), 'the seats of the poll');
  tryDraw(() => wireHouse(initial, catalog), 'the house in the cycle and its controls');
  tryDraw(() => wireCardImages(document, showError), 'the download of the card');
}

wireControls();
mountWhenReady(paint);
