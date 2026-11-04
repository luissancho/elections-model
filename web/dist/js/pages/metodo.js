// Method pages: the controls; on the backtest subpage, the vote error, the coverage and the seat error of the
// backtest by horizon and the reliability of its bands (the PIT by population and group of horizons, and the
// probabilities of majority against how often there was one); on the average subpage, the live examples: the
// weights of the polls (`charts/weights.js`) and the bullseye of the last polls of one house before every election
// since a date (`charts/bullseye.js`); on the rating subpage, the cloud of the rating check (the differential the
// rating gave each pollster before an election against the one it had, `charts/scatter.js`) with its download, all
// drawn from the data embedded by the server; on every page, the footnotes that unfold in place (`notes.js`). The
// landing page has no data and draws nothing.
import {fmtNum} from '../format.js';
import {THEME} from '../charts/base.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderGrouped} from '../charts/grouped.js';
import {renderLines} from '../charts/line.js';
import {renderScatter, scatterOption} from '../charts/scatter.js';
import {renderWeights} from '../charts/weights.js';
import {wireNotes} from '../notes.js';
import {IMAGE, wireDownload} from '../share.js';
import {wireBullseye} from './bullseye-wire.js';
import {mountWhenReady, readInitial, showError, tryDraw, wireControls} from './common.js';

// Series colours: the model stands out, the baselines are lighter.
const COLORS = {model: '#1f4e79', last_poll: '#c0504d', mean_4w: '#e69f00', prev_result: '#009e73'};
// One colour per coverage level, distinct from the grey dashed reference lines.
const COVERAGE_COLORS = {'50': '#e69f00', '80': '#009e73', '95': '#1f4e79'};
const NAMES = {model: 'Modelo', last_poll: 'Último sondeo', mean_4w: 'Media de 4 semanas', prev_result: 'Elección anterior'};
const COVERAGE_LEVELS = ['50', '80', '95'];
// The bullseye of the house effects example, as the one of a profile (`casas.js`): general elections in the
// accent and over the regional ones, which take a contrasting tone; every point labelled with its election.
const NATIONAL_SCOPE = 'es';
const KIND_GENERAL = 'Generales';
const KIND_REGIONAL = 'Autonómicas';
const REGIONAL_COLOR = '#c0672e';
// The reliability card: one colour per group of horizons (the server sends them in order, shortest first), the ten
// tenths of the band and the share each one gets from a calibrated band.
const PIT_COLORS = ['#1f4e79', '#e69f00', '#c0504d'];
const PIT_BINS = 10;
const PIT_EXPECTED = 1 / PIT_BINS;
const PIT_TENTHS = Array.from({length: PIT_BINS}, (_, k) => `${10 * k}-${10 * (k + 1)}`);

/** `[horizon, value]` pairs of an aligned list, without the missing values. */
function pairs(horizons, values) {
  return horizons.map((horizon, index) => [horizon, values[index]])
    .filter(([, value]) => value !== null && value !== undefined);
}

/**
 * Reveal the Publicada/Corregida switch of the weights example (`#weights-field`, hidden without JS) and redraw the
 * points with the chosen figure when it changes: the published one, or the one net of the house effect that the
 * line is fitted on.
 *
 * @param {HTMLElement} el the chart container
 * @param {object} example the `example` of the page's initial data
 */
export function wireWeightsField(el, example) {
  const field = document.getElementById('weights-field');
  if (!field) {
    return;
  }
  field.hidden = false;
  field.addEventListener('change', (event) => {
    const value = event.target && event.target.value;
    if (value === 'pct' || value === 'pct_adj') {
      tryDraw(() => renderWeights(el, example, {field: value}), 'the weights example');
    }
  });
}

/**
 * Draw the reliability card: per population, the ten bars of each group of horizons with the line of 0.10, and the
 * probabilities of majority given by the model against the frequency observed, over the diagonal.
 *
 * @param {object} reliability the `reliability` of the page's initial data (`pit`, `majority`)
 */
function paintReliability({pit = {}, majority = []} = {}) {
  Object.entries(pit).forEach(([population, groups]) => {
    const el = document.getElementById(`pit-${population}-chart`);
    const series = Object.entries(groups).map(([group, observed], index) => ({
      name: `${group} días`, color: PIT_COLORS[index % PIT_COLORS.length], data: observed,
      labels: observed.map(() => ''),  // thirty values would hide the bars: the tooltip gives them
    }));
    if (el && series.length) {
      // from 0 (no room below it) to the next tenth above the highest bar, at least 0.2 so the line sits inside
      tryDraw(() => renderGrouped(el, PIT_TENTHS, series, {
        formatter: (x) => fmtNum(x, 2), references: [PIT_EXPECTED],
        max: ({max}) => Math.max(2 * PIT_EXPECTED, Math.ceil(max * PIT_BINS) / PIT_BINS),
      }), `the reliability of the ${population}`);
    }
  });

  const majorityEl = document.getElementById('majority-reliability-chart');
  if (majorityEl && majority.length) {
    const rows = majority.map((row) => ({...row, name: `${row.n} ${row.n === 1 ? 'caso' : 'casos'}`}));
    tryDraw(() => renderScatter(majorityEl, rows, {
      x: 'p_mean', y: 'observed', name: 'name', diagonal: true,
      axes: {xMin: 0, xMax: 1, yMin: 0, yMax: 1, interval: 0.2},
      xFormatter: (v) => fmtNum(v, 2), yFormatter: (v) => fmtNum(v, 2),
      xName: 'Probabilidad de mayoría que daba el modelo', xLabel: 'Probabilidad', yLabel: 'frecuencia',
    }), 'the reliability of the majorities');
  }
}

// The diameter of the points of the rating check, in pixels: smaller than the default, the cloud has hundreds
const RATING_CHECK_POINT = 5;

/** An axis tick (a whole number) without decimals, any other value of the cloud with two. */
function differentialText(value) {
  return fmtNum(value, Number.isInteger(value) ? 0 : 2);
}

/**
 * The options of the cloud of the rating check: one point per pollster and election (its name and the election in
 * the tooltip), the diagonal of an exact forecast and the fitted line, over the symmetric axes of the server.
 *
 * @param {object} check the `rating_check` of the page's initial data (`scatter`, `fit`, `axes`)
 * @returns {object} the options of `scatterOption`
 */
function ratingCheckOptions({fit = null, axes = null} = {}) {
  return {
    x: 'd_prev', y: 'd_obs', name: 'label', fit, diagonal: true, axes, symbolSize: RATING_CHECK_POINT,
    xFormatter: differentialText, yFormatter: differentialText,
    xName: 'Diferencial previsto', yName: 'Diferencial observado', xLabel: 'Previsto', yLabel: 'observado',
  };
}

/**
 * Draw the cloud of the rating check and wire its download button.
 *
 * @param {object} check the `rating_check` of the page's initial data
 */
function paintRatingCheck(check) {
  const el = document.getElementById('rating-check-chart');
  const rows = check.scatter || [];
  if (!el || !rows.length) {
    return;
  }
  const options = ratingCheckOptions(check);
  if (tryDraw(() => renderScatter(el, rows, options), 'the rating check')) {
    // square, as on the page, so the diagonal keeps its 45 degrees in the image
    wireDownload(document.getElementById('rating-check-download'), () => scatterOption(rows, options), showError, {
      chartHeight: IMAGE.width - 2 * IMAGE.pad,
    });
  }
}

/**
 * Draw the charts whose container is present. A scope without a published backtest renders no `#initial-data`
 * island (and no charts), so there is nothing to draw and `readInitial` must not be called.
 */
function paint() {
  if (!document.getElementById('initial-data')) {
    return;
  }

  const data = readInitial();
  const weightsEl = document.getElementById('weights-chart');
  if (weightsEl && data.example) {
    tryDraw(() => renderWeights(weightsEl, data.example), 'the weights example');
    wireWeightsField(weightsEl, data.example);
  }

  const houseEl = document.getElementById('house-chart');
  const house = data.house || {};
  if (houseEl && (house.rows || []).length && (house.blocks || []).length >= 2) {
    tryDraw(() => renderBullseye(houseEl, house.rows, {
      blocks: house.blocks,
      groupOf: (row) => (row.scope === NATIONAL_SCOPE ? KIND_GENERAL : KIND_REGIONAL),
      colors: {[KIND_GENERAL]: THEME.accentColor, [KIND_REGIONAL]: REGIONAL_COLOR},
      legend: true,
      // only the general elections, by their year (as /analisis/sesgo-cis): 41 codes like «2019-26M-es-ar» were
      // clutter once the labels showed again (03f4e6d); the tooltip names every point
      labelOf: (row) => (row.scope === NATIONAL_SCOPE ? row.event_date.slice(0, 4) : ''),
      onTop: (group) => group === KIND_GENERAL,
    }), 'the house effects example');
  }

  // the model scored like a pollster in the backtest page: its bullseye with the filters of a profile
  if (data.bullseye && data.bullseye.length) {
    wireBullseye(data.bullseye, data.bullseye_blocks || []);
  }

  const {horizons = [], mae = {}, coverage = {}, seats = {}, reliability = null} = data;

  const maeEl = document.getElementById('mae-chart');
  if (maeEl && horizons.length) {
    const series = Object.keys(NAMES)
      .map((key) => ({name: NAMES[key], color: COLORS[key], dashed: key !== 'model',
        points: pairs(horizons, mae[key] || [])}))
      .filter((line) => line.points.length);
    if (series.length) {
      tryDraw(() => renderLines(maeEl, series, {formatter: (x) => fmtNum(x, 1)}), 'the error by horizon');
    }
  }

  const coverageEl = document.getElementById('coverage-chart');
  if (coverageEl && horizons.length) {
    const series = COVERAGE_LEVELS
      .map((level) => ({name: `Intervalo del ${level} %`, color: COVERAGE_COLORS[level], points: pairs(horizons, coverage[level] || [])}))
      .filter((line) => line.points.length);
    if (series.length) {
      tryDraw(() => renderLines(coverageEl, series, {
        formatter: (x) => fmtNum(x, 2), yMin: 0, yMax: 1, references: [0.5, 0.8, 0.95],
      }), 'the coverage by horizon');
    }
  }

  const seatsEl = document.getElementById('seats-chart');
  if (seatsEl && horizons.length) {
    // One line: the headline is the central allocation, the same in both modes, so the horizon run adds nothing
    const series = [
      {name: 'Error de escaños', color: COLORS.model, points: pairs(horizons, seats.mae || [])},
    ].filter((line) => line.points.length);
    if (series.length) {
      tryDraw(() => renderLines(seatsEl, series, {formatter: (x) => fmtNum(x, 1)}), 'the seat error by horizon');
    }
  }

  if (reliability) {
    paintReliability(reliability);
  }

  if (data.rating_check) {
    paintRatingCheck(data.rating_check);
  }
}

wireControls();
wireNotes();
mountWhenReady(paint);
