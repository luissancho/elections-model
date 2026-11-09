// HTML legend of the series charts: a pill per party or block that toggles the ECharts legend selection, and the line
// under it that names the parties of each block in a block view.
import {blockMembers} from '../blocks.js';

/**
 * Fill `container` with one pill per item and show it.
 *
 * @param {HTMLElement} container the `.legend` element (emptied first)
 * @param {Array<{name: string, color: string, ink: (string|undefined), on: boolean}>} items names in legend
 *   order; `ink` is the text colour over `color` when the pill is on (white without it)
 * @param {function} onToggle called with the name of the pill pressed
 */
export function renderLegend(container, items, onToggle) {
  container.replaceChildren();
  for (const item of items) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = item.on ? 'lg on' : 'lg';
    button.dataset.name = item.name;
    button.setAttribute('aria-pressed', String(item.on));
    button.style.setProperty('--c', item.color);
    if (item.ink) {
      button.style.setProperty('--ink', item.ink);
    }
    const dot = document.createElement('i');
    dot.className = 'dot';
    button.append(dot, item.name);
    button.addEventListener('click', () => onToggle(item.name));
    container.append(button);
  }
  container.hidden = false;
}

/**
 * Light or dim the pills of `container` after a legend selection.
 *
 * @param {HTMLElement} container the `.legend` element
 * @param {object} selected `{name: boolean}` as ECharts reports it in `legendselectchanged`
 */
export function markLegend(container, selected) {
  for (const button of container.querySelectorAll('button[data-name]')) {
    const on = selected[button.dataset.name] !== false;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  }
}

/**
 * Show, right after `anchor`, the parties of each block (`blockMembers`): one item per block with its dot, its name
 * in bold and its parties; with no blocks the line is hidden. The line is a `p.block-members` sibling of `anchor`,
 * made the first time.
 *
 * @param {HTMLElement|null} anchor the element the line follows (the pill legend, or the switch of the hemicycle)
 * @param {Object<string, string[]>|null} blocks block → party names (`meta.bmaps.blocks`), null in a party view
 * @param {Object<string, string>} [colors] colour of each block
 */
export function renderMembers(anchor, blocks, colors = {}) {
  if (!anchor) {
    return;
  }
  let line = anchor.nextElementSibling;
  if (!blocks && !(line && line.classList.contains('block-members'))) {
    return;
  }
  if (!line || !line.classList.contains('block-members')) {
    line = document.createElement('p');
    line.className = 'block-members';
    anchor.after(line);
  }
  const items = blocks ? blockMembers(blocks) : [];
  line.replaceChildren(...items.map(({block, text}) => {
    const item = document.createElement('span');
    const dot = document.createElement('i');
    dot.className = 'dot';
    dot.style.backgroundColor = colors[block] || '';
    const name = document.createElement('strong');
    name.textContent = `${block}:`;
    item.append(dot, name, ` ${text}`);
    return item;
  }));
  line.hidden = !items.length;
}
