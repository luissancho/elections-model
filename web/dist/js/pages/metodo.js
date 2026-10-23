// Method pages: the controls; on the backtest subpage, the vote error, the coverage and the seat error of the
// backtest by horizon; on the average subpage, the live examples: the weights of the polls (`charts/weights.js`)
// and the bullseye of the last polls of one house before every election since a date (`charts/bullseye.js`), all
// drawn from the data embedded by the server; on every page, the footnotes that unfold in place (`notes.js`). The
// landing page has no data and draws nothing.
import {fmtNum} from '../format.js';
import {THEME} from '../charts/base.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderLines} from '../charts/line.js';
import {renderWeights} from '../charts/weights.js';
import {wireNotes} from '../notes.js';
import {wireBullseye} from './bullseye-wire.js';
import {mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

// Series colours: the model stands out, the baselines are lighter.
const COLORS = {model: '#1f4e79', last_poll: '#c0504d', mean_4w: '#e69f00', prev_result: '#009e73'};
// One colour per coverage level, distinct from the grey dashed reference lines.
const COVERAGE_COLORS = {'50': '#e69f00', '80': '#009e73', '95': '#1f4e79'};
const NAMES = {model: 'Modelo', last_poll: 'Último sondeo', mean_4w: 'Media de 4 semanas', prev_result: 'Elección anterior'};
const COVERAGE_LEVELS = ['50', '80', '95'];
// The horizon run (mode Elección) of the seat error: its own colour, dashed.
const ELECTION_COLOR = '#7f3c8d';
// The bullseye of the house effects example, as the one of a profile (`casas.js`): general elections in the
// accent and over the regional ones, which take a contrasting tone; every point labelled with its election.
const NATIONAL_SCOPE = 'es';
const KIND_GENERAL = 'Generales';
const KIND_REGIONAL = 'Autonómicas';
const REGIONAL_COLOR = '#c0672e';

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
      labelOf: (row) => `${row.event}-${row.scope}`,
      onTop: (group) => group === KIND_GENERAL,
    }), 'the house effects example');
  }

  // the model scored like a pollster in the backtest page: its bullseye with the filters of a profile
  if (data.bullseye && data.bullseye.length) {
    wireBullseye(data.bullseye, data.bullseye_blocks || []);
  }

  const {horizons = [], mae = {}, coverage = {}, seats = {}} = data;

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
    const series = [
      {name: 'Modo Hoy', color: COLORS.model, points: pairs(horizons, seats.mae || [])},
      {name: 'Modo Elección', color: ELECTION_COLOR, dashed: true, points: pairs(horizons, seats.mae_h || [])},
    ].filter((line) => line.points.length);
    if (series.length) {
      tryDraw(() => renderLines(seatsEl, series, {formatter: (x) => fmtNum(x, 1)}), 'the seat error by horizon');
    }
  }
}

wireControls();
wireNotes();
mountWhenReady(paint);
