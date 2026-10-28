// Party catalogue built from the data of the meta part.
import {REST_BLOCK} from './blocks.js';

export const OTHERS_COLOR = '#9e9e9e';
// Order of the parties by block
export const BLOCK_ORDER = ['Izquierda', 'Separatista', 'Regionalista', 'Derecha'];
// Order of the blocks view (hemicycle, stacked bar): the rest sits between the two sides
export const BLOCK_VIEW_ORDER = ['Izquierda', REST_BLOCK, 'Derecha'];

export class Catalog {
  /** @param {object} meta the `data` member of the meta envelope */
  constructor(meta) {
    this.parties = meta.parties || [];
    this.bmaps = meta.bmaps || {};
    this.byName = new Map(this.parties.map((party) => [party.name, party]));
  }

  color(name) {
    const party = this.byName.get(name);
    return party && party.color ? party.color : OTHERS_COLOR;
  }

  fullname(name) {
    const party = this.byName.get(name);
    return party && party.fullname ? party.fullname : name;
  }

  block(name) {
    const party = this.byName.get(name);
    return party ? party.block : null;
  }

  /** Colour of the first party listed for the block in `bmaps.vs`, then `bmaps.blocks`; the rest block is always grey. */
  blockColor(block) {
    if (block === REST_BLOCK) {
      return OTHERS_COLOR;
    }
    for (const key of ['vs', 'blocks']) {
      const names = (this.bmaps[key] || {})[block];
      if (names && names.length) {
        return this.color(names[0]);
      }
    }
    return OTHERS_COLOR;
  }

  /** Sort block names by BLOCK_VIEW_ORDER (Izquierda, Otros, Derecha); unknown blocks go last, in input order. */
  orderBlocks(names) {
    const rank = (name) => {
      const index = BLOCK_VIEW_ORDER.indexOf(name);
      return index < 0 ? BLOCK_VIEW_ORDER.length : index;
    };
    return [...names].sort((a, b) => rank(a) - rank(b));
  }

  /** Sort party names by block (BLOCK_ORDER, unknown last) and then by catalogue order. */
  order(names) {
    const rank = (name) => {
      const index = BLOCK_ORDER.indexOf(this.block(name));
      return index < 0 ? BLOCK_ORDER.length : index;
    };
    const position = (name) => {
      const index = this.parties.findIndex((party) => party.name === name);
      return index < 0 ? this.parties.length : index;
    };
    return [...names].sort((a, b) => rank(a) - rank(b) || position(a) - position(b));
  }
}
