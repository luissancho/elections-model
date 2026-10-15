// HTML legend of the series charts: a pill per party or block that toggles the ECharts legend selection.

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
