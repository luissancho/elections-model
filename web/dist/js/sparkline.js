// Sparkline of a series of values: the points of a polyline in a box, and its SVG drawing (party cards).

/**
 * Points of the sparkline of `values` in a box of `width` x `height`: one per non-null value, at its index position
 * along the width, the smallest value at the bottom and the largest at the top (3 px of air).
 *
 * @param {Array<number|null>} values the series, in order (gaps are skipped but keep their position)
 * @param {number} width box width
 * @param {number} height box height
 * @returns {number[][]} `[x, y]` pairs
 */
export function sparklinePoints(values, width = 180, height = 28) {
  const known = values.filter((v) => v !== null && v !== undefined);
  if (!known.length) {
    return [];
  }
  const min = Math.min(...known);
  const max = Math.max(...known);
  const span = Math.max(1, values.length - 1);
  const points = [];
  values.forEach((v, i) => {
    if (v === null || v === undefined) {
      return;
    }
    const y = max === min ? height / 2 : height - 3 - ((v - min) / (max - min)) * (height - 6);
    points.push([Math.round((i / span) * width * 100) / 100, Math.round(y * 100) / 100]);
  });
  return points;
}

/**
 * Draw the sparkline of `values` inside `svg` (an SVG element already in the DOM, with its own viewBox): a white
 * polyline and a dot on the last point.
 *
 * @param {SVGElement} svg the target
 * @param {Array<number|null>} values the series
 */
export function sparklineSvg(svg, values) {
  const NS = 'http://www.w3.org/2000/svg';
  const box = (svg.getAttribute('viewBox') || '0 0 180 28').split(/\s+/).map(Number);
  const points = sparklinePoints(values, box[2], box[3]);
  svg.replaceChildren();
  if (!points.length) {
    return;
  }
  const line = document.createElementNS(NS, 'polyline');
  line.setAttribute('points', points.map((p) => p.join(',')).join(' '));
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', '#fff');
  line.setAttribute('stroke-width', '2');
  line.setAttribute('stroke-linejoin', 'round');
  line.setAttribute('stroke-linecap', 'round');
  line.setAttribute('vector-effect', 'non-scaling-stroke');
  const [x, y] = points[points.length - 1];
  const dot = document.createElementNS(NS, 'circle');
  dot.setAttribute('cx', String(x));
  dot.setAttribute('cy', String(y));
  dot.setAttribute('r', '2.6');
  dot.setAttribute('fill', '#fff');
  svg.append(line, dot);
}
