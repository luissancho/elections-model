// Houses page and pollster profile: rating bars, house-effects heat map, rating path and bullseye, drawn
// from the data embedded by the server.
import {fmtNum} from '../format.js';
import {renderBars} from '../charts/bars.js';
import {renderBullseye} from '../charts/bullseye.js';
import {renderHeatmap} from '../charts/heatmap.js';
import {renderLine} from '../charts/line.js';
import {mountWhenReady, readInitial, tryDraw, wireControls} from './common.js';

/** Draw the charts whose container is present and has data. */
function paint() {
  const {ranking = [], grid = null, path = [], bullseye = [], blocks = []} = readInitial();

  const bars = document.getElementById('rating-chart');
  if (bars && ranking.length) {
    tryDraw(() => renderBars(bars, ranking.map((row) => ({name: row.pollster, rating: row.rating})), {
      value: 'rating', lo: 'lo', hi: 'hi', formatter: (x) => fmtNum(x, 1),
    }), 'the rating bars');
  }

  const heatmap = document.getElementById('effects-heatmap');
  if (heatmap && grid && grid.rows.length) {
    tryDraw(() => renderHeatmap(heatmap, grid), 'the house effects');
  }

  const pathEl = document.getElementById('rating-path');
  const points = path.filter((row) => row.rating !== null && row.rating !== undefined)
    .map((row) => [row.event_date, row.rating]);
  if (pathEl && points.length) {
    tryDraw(() => renderLine(pathEl, points, {formatter: (x) => fmtNum(x, 1), name: 'Rating'}), 'the rating path');
  }

  const target = document.getElementById('bullseye-chart');
  if (target && bullseye.length && blocks.length >= 2) {
    tryDraw(() => renderBullseye(target, bullseye, {blocks}), 'the bullseye');
  }
}

wireControls();
mountWhenReady(paint);
