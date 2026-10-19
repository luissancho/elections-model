// The page of the CERA study (`/analisis/voto-cera`): the charts of its three questions, drawn from the data
// embedded by the server (`initial`, `_island` of `pages/analisis.py`), the «Ver más» buttons of its tables and the
// scenario selector of question 3, which swaps the table of the prediction and redraws the gap chart.
import {fmtDate, fmtInt, fmtNum, fmtPct} from '../format.js';
import {THEME} from '../charts/base.js';
import {renderCensus} from '../charts/census.js';
import {renderGrouped} from '../charts/grouped.js';
import {renderRadar} from '../charts/radar.js';
import {renderScatter} from '../charts/scatter.js';
import {renderSlope} from '../charts/slope.js';
import {markSegments, mountWhenReady, readInitial, tryDraw, wireControls, wireMore} from './common.js';

// The palette of the study's charts (the approved mockup): the accent for what is, red for what is feared.
const RED = '#ce3b3b';
const AMBER = '#e8912b';
const GREEN = '#2e9e5b';
const PURPLE = '#7b3fa0';
const GREY = '#c7cdd8';
const SAND = '#e9d9c3';
const SLATE = '#9aa7bd';
const LIGHT_AMBER = '#f2c38a';
// One colour per regional process (participation by origin) and per community (the slope chart), in order.
const PROCESS_COLORS = [AMBER, LIGHT_AMBER, GREY, SLATE];
const COMMUNITY_COLORS = [AMBER, THEME.accentColor, GREEN, PURPLE, RED, SLATE];

// The origins of the electors (`region` of `cera/countries.csv`) and the communities of the files of
// `cera/consulados/` (`cyl-2026`): their names on the page. Anything else is shown as it comes.
const ORIGIN_NAMES = {latam: 'Latinoamérica', europe: 'Europa', usa_canada: 'EE. UU. y Canadá', other: 'Resto'};
const LATAM = 'Latinoamérica';
const COMMUNITY_NAMES = {cyl: 'Castilla y León', aragon: 'Aragón', extremadura: 'Extremadura', andalucia: 'Andalucía'};

// The axes of the radar of question 3, in the order of `initial.radar.today`.
const RADAR_AXES = ['Participación CERA', 'Colocación dirigida\nen provincias\nsensibles', 'Voto a la\nizquierda'];
// How many provinces of the ecological cloud are named at each end of each axis.
const NAMED_AT_EACH_END = 2;

/** The name of an origin. */
function originName(origin) {
  return ORIGIN_NAMES[origin] || origin;
}

/** The name of a regional process from its file stem: 'cyl-2026' → 'Castilla y León 2026'. */
function processName(process) {
  const match = /^(.*)-(\d{4})$/.exec(process || '');
  if (!match) {
    return process;
  }
  return `${COMMUNITY_NAMES[match[1]] || match[1]} ${match[2]}`;
}

/** The year of a regional process, from its file stem (0 when it has none). */
function processYear(process) {
  const match = /-(\d{4})$/.exec(process || '');
  return match ? Number(match[1]) : 0;
}

/** Month and year of an ISO date: '2022-10-01' → 'oct 2022'. */
function monthYear(iso) {
  return fmtDate(iso).replace(/^\d+ /, '');
}

/** A number with its sign and a decimal: '+6,2', '-8,3'. */
function signed(value) {
  return `${value > 0 ? '+' : ''}${fmtNum(value, 1)}`;
}

/** A tick of a signed axis, without decimals when whole: '+3', '-1,5'. */
function signedTick(value) {
  return `${value > 0 ? '+' : ''}${fmtNum(value, Number.isInteger(value) ? 0 : 1)}`;
}

/** A percentage without decimals when it is whole (the ticks of an axis) and with one otherwise. */
function pct(value) {
  return fmtPct(value, Number.isInteger(value) ? 0 : 1);
}

/** A number of electors on an axis: thousands as 'k' from a thousand on. */
function thousands(value) {
  return Math.abs(value) >= 1000 ? `${fmtNum(value / 1000, 0)} k` : fmtInt(value);
}

/** Sum of the finite values of a list. */
function total(values) {
  return values.filter(Number.isFinite).reduce((a, b) => a + b, 0);
}

/** The container `#id` when it exists and there is something to draw in it. */
function container(id, ready) {
  const el = document.getElementById(id);
  return el && ready ? el : null;
}

/** Question 1: the census series with its marks. */
function drawCensus(census) {
  const {dates = [], values = []} = census.series || {};
  const el = container('census-chart', dates.length);
  if (el) {
    const points = dates.map((date, i) => [date, values[i]]);
    tryDraw(() => renderCensus(el, points, census.marks || []), 'the census series');
  }
}

/**
 * Question 1: the additions of the two periods the page shows (`initial.census.periods`), stacked as Latin
 * America and the rest, with the Latin American share inside its segment and the total (and, after the law, the
 * pace) above each stack.
 */
function drawPeriods(origins, periods) {
  const el = container('periods-chart', origins.length);
  if (!el) {
    return;
  }
  const latam = origins.filter((row) => originName(row.name) === LATAM);
  const rest = origins.filter((row) => originName(row.name) !== LATAM);
  const sums = (rows) => ['g1', 'g2'].map((key) => total(rows.map((row) => row[key])));
  const [latamSums, restSums] = [sums(latam), sums(rest)];
  const totals = latamSums.map((value, i) => value + restSums[i]);
  const ratio = totals[0] ? totals[1] / totals[0] : null;
  const sign = (value) => `${value > 0 ? '+' : ''}${fmtInt(value)}`;
  const share = (value, i) => (totals[i] ? fmtPct(100 * value / totals[i], 0) : '');
  const [before, after] = periods || [];
  const span = (period) => (period ? `\n(${monthYear(period[0])} – ${monthYear(period[1])})` : '');
  tryDraw(() => renderGrouped(el, [`Antes de la ley${span(before)}`, `Después de la ley${span(after)}`], [
    {name: LATAM, data: latamSums, color: RED, labels: latamSums.map(share)},
    {name: 'Resto', data: restSums, color: SAND, labels: ['', '']},
  ], {
    stacked: true,
    formatter: fmtInt,
    axisFormatter: thousands,
    totals: [sign(totals[0]), ratio === null ? sign(totals[1]) : `${sign(totals[1])}  ×${fmtNum(ratio, 1)}`],
  }), 'the additions per period');
}

/** Question 2: the participation by origin, one series per regional process, the latest first. */
function drawOrigins(participation) {
  const el = container('origin-chart', participation.length);
  if (!el) {
    return;
  }
  const origins = [...new Set(participation.map((row) => row.origin))];
  const processes = [...new Set(participation.map((row) => row.process))]
    .sort((a, b) => processYear(b) - processYear(a));
  const rate = (process, origin) => participation.find((row) => row.process === process && row.origin === origin)?.rate ?? null;
  const series = processes.map((process, i) => ({
    name: processName(process),
    color: PROCESS_COLORS[i % PROCESS_COLORS.length],
    data: origins.map((origin) => rate(process, origin)),
  }));
  tryDraw(() => renderGrouped(el, origins.map(originName), series, {formatter: (v) => fmtPct(v, 1), axisFormatter: pct}),
    'the participation by origin');
}

/** The names of the provinces at both ends of both axes of the ecological cloud. */
function extremes(rows) {
  const named = new Set();
  for (const key of ['latam_share', 'left_share']) {
    const sorted = rows.filter((row) => Number.isFinite(row[key])).sort((a, b) => a[key] - b[key]);
    for (const row of [...sorted.slice(0, NAMED_AT_EACH_END), ...sorted.slice(-NAMED_AT_EACH_END)]) {
      named.add(row.name);
    }
  }
  return [...named];
}

/**
 * The ends of the least squares line of the ecological cloud over its range of `latam_share`: the slope is the
 * published one (`ecological.slope`, an unweighted fit) and the line goes through the means of the points.
 */
function fitLine(rows, slope) {
  const points = rows.filter((row) => Number.isFinite(row.latam_share) && Number.isFinite(row.left_share));
  if (!Number.isFinite(slope) || points.length < 2) {
    return null;
  }
  const xs = points.map((row) => row.latam_share);
  const meanX = total(xs) / points.length;
  const meanY = total(points.map((row) => row.left_share)) / points.length;
  const at = (x) => [x, meanY + slope * (x - meanX)];
  return [at(Math.min(...xs)), at(Math.max(...xs))];
}

/** Question 2: the ecological cloud of the provinces with its fitted line. */
function drawEcological(ecological) {
  const rows = ecological.rows || [];
  const el = container('ecological-chart', rows.length);
  if (el) {
    tryDraw(() => renderScatter(el, rows, {
      x: 'latam_share', y: 'left_share', name: 'name', labelled: extremes(rows), fit: fitLine(rows, ecological.slope),
      xFormatter: pct, yFormatter: pct, xName: '% del censo CERA de origen latinoamericano',
      xLabel: 'Latam', yLabel: 'izquierda',
    }), 'the ecological cloud');
  }
}

/** Question 2: the block margin of the exterior vote from the previous regional election to the last one. */
function drawSlope(slope) {
  const el = container('slope-chart', slope.length);
  if (el) {
    const rows = slope.map((row, i) => ({name: row.community, from: row.previous, to: row.last,
      color: COMMUNITY_COLORS[i % COMMUNITY_COLORS.length]}));
    tryDraw(() => renderSlope(el, rows, {labels: ['Comicio anterior', 'Último comicio']}), 'the block margins');
  }
}

/** Question 2: the change of the PSOE and of the rest of the left in the exterior vote, per community. */
function drawCannibal(cannibal) {
  const el = container('cannibal-chart', cannibal.length);
  if (el) {
    tryDraw(() => renderGrouped(el, cannibal.map((row) => row.community), [
      {name: 'PSOE', color: RED, data: cannibal.map((row) => row.d_psoe)},
      {name: 'Otras izquierdas (SUMAR, Podemos…)', color: PURPLE, data: cannibal.map((row) => row.d_other_left)},
    ], {formatter: signed, axisFormatter: signedTick}), 'the change of the left vote');
  }
}

/**
 * Question 2: the share of the sensitive provinces in the census and in the additions after the law (100 %), in
 * the version of the periods the server chose (`initial.sensitive`: the updated one, or the article's without it).
 */
function drawSensitive(sensitive) {
  const shown = sensitive || {};
  const all = shown.all || {};
  const {stock_share: stock, growth_share: growth} = all;
  const el = container('sensitive-chart', Number.isFinite(stock) && Number.isFinite(growth));
  if (!el) {
    return;
  }
  const [start, end] = shown.period || [];
  const categories = [
    start ? `Censo antes de la ley\n(${monthYear(start)})` : 'Censo antes de la ley',
    end ? `Nuevas altas tras la ley\n(hasta ${monthYear(end)})` : 'Nuevas altas tras la ley',
  ];
  tryDraw(() => renderGrouped(el, categories, [
    {name: 'Provincias sensibles', color: RED, data: [stock, growth]},
    {name: 'Resto de provincias', color: GREY, labelColor: THEME.textColor, data: [100 - stock, 100 - growth]},
  ], {stacked: true, max: 100, formatter: (v) => fmtPct(v, 1), axisFormatter: pct}), 'the sensitive provinces');
}

/**
 * Question 3: the gap to the last seat of the closest provinces against the net effect of the CERA in one
 * scenario, the net in the accent when it favours the runner-up and in red otherwise. Each row is labelled with
 * the siglas of the winner and the runner-up that the island carries: «Girona (JxCat/PP)».
 */
function drawGap(rows) {
  const el = container('gap-chart', rows && rows.length);
  if (!el) {
    return;
  }
  const label = (row) => (row.winner && row.runner_up ? `${row.name} (${row.winner}/${row.runner_up})` : row.name);
  const number = (v) => fmtNum(v, Number.isInteger(v) ? 0 : 1);
  tryDraw(() => renderGrouped(el, rows.map(label), [
    {name: 'Brecha del último escaño', color: GREY, data: rows.map((row) => row.gap), formatter: fmtInt},
    {name: 'Efecto neto del CERA', color: THEME.accentColor, formatter: number,
      data: rows.map((row) => ({value: row.net, color: row.net > 0 ? THEME.accentColor : RED}))},
  ], {horizontal: true, legend: 'bottom', formatter: number, axisFormatter: thousands}), 'the gap to the last seat');
}

/** Question 3: today's levers against «everything at the maximum». */
function drawRadar(radar) {
  const today = (radar && radar.today) || [];
  const el = container('radar-chart', today.length === RADAR_AXES.length);
  if (el) {
    const max = radar.max || RADAR_AXES.map(() => 100);
    const indicators = RADAR_AXES.map((name, i) => ({name, max: max[i]}));
    tryDraw(() => renderRadar(el, indicators, [
      {name: 'Realidad hoy', value: today, color: THEME.accentColor, filled: true},
      {name: 'Todo al máximo', value: max, color: RED, filled: false},
    ]), 'the radar of the levers');
  }
}

/** The scenario checked in the selector of question 3 (`null` without one). */
function checkedScenario() {
  const input = document.querySelector('#scenario input[name=scenario]:checked');
  return input ? input.value : null;
}

/** Show the table of the prediction of `scenario` only, mark the segments and redraw the gap chart. */
function showScenario(data, scenario) {
  for (const table of document.querySelectorAll('#prediction-table table[data-scenario]')) {
    table.hidden = table.dataset.scenario !== scenario;
  }
  markSegments();
  drawGap((data.gap || {})[scenario] || []);
}

/** Draw the charts whose container is present and wire the scenario selector. */
function paint() {
  if (!document.getElementById('initial-data')) {
    return;
  }
  const data = readInitial();
  drawCensus(data.census || {});
  drawPeriods(data.origins || [], (data.census || {}).periods);
  drawOrigins(data.participation || []);
  drawEcological(data.ecological || {});
  drawSlope(data.slope || []);
  drawCannibal(data.cannibal || []);
  drawSensitive(data.sensitive);
  drawRadar(data.radar);

  // the browser may restore a radio other than the first one; without a selector, the first scenario
  const selector = document.getElementById('scenario');
  const scenario = checkedScenario() || Object.keys(data.gap || {})[0];
  if (scenario) {
    showScenario(data, scenario);
  }
  if (selector) {
    selector.addEventListener('change', () => {
      const value = checkedScenario();
      if (value) {
        showScenario(data, value);
      }
    });
  }
}

wireControls();
wireMore('provinces-more', 'provinces-table');
wireMore('check-more', 'check-rows');
markSegments();
mountWhenReady(paint);
