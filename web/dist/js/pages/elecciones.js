// Past elections page: sector error over time and, in the detail of one election, the error of each house
// and the bullseye of their last poll, drawn from the data embedded by the server.
import {fmtNum} from '../format.js';
import {renderBars} from '../charts/bars.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderLine} from '../charts/line.js';
import {mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

// Height of the error bars: one row per house plus the axes.
const BAR_ROW_HEIGHT = 22;
const BAR_EXTRA_HEIGHT = 60;

/**
 * Draw the charts whose container is present and has data. A scope without published elections renders no
 * `#initial-data` island (and no charts), so there is nothing to draw and `readInitial` must not be called.
 */
function paint() {
  if (!document.getElementById('initial-data')) {
    return;
  }

  const {meta = {}, sector = [], pollsters = [], bullseye = [], blocks = []} = readInitial();

  const sectorEl = document.getElementById('sector-chart');
  const points = sector.filter((row) => row.error_avg !== null && row.error_avg !== undefined)
    .map((row) => [row.date, row.error_avg]);
  if (sectorEl && points.length) {
    tryDraw(() => renderLine(sectorEl, points, {formatter: (x) => fmtNum(x, 1), name: 'Error medio del sector'}),
      'the sector error');
  }

  const bars = document.getElementById('error-chart');
  const rows = pollsters.filter((row) => row.error_blocks !== null && row.error_blocks !== undefined)
    .map((row) => ({name: row.pollster, error_blocks: row.error_blocks}));
  if (bars && rows.length) {
    bars.style.height = `${rows.length * BAR_ROW_HEIGHT + BAR_EXTRA_HEIGHT}px`;
    tryDraw(() => renderBars(bars, rows, {value: 'error_blocks', formatter: (x) => fmtNum(x, 1)}),
      'the house errors');
  }

  const target = document.getElementById('bullseye-chart');
  if (target && bullseye.length && blocks.length >= 2) {
    tryDraw(() => renderBullseye(
      target,
      bullseye.map((row) => ({...row, event: row.pollster, event_date: meta.event_date})),
      {blocks},
    ), 'the bullseye');
  }
}

wireControls();
mountWhenReady(paint);
