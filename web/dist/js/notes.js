// Footnotes that unfold in place: a call (`.note-ref a`, from the `note` macro) links to its note of the «Notas» card,
// which stays visible at the end of the page. With JS, a click opens a copy of the note right after the block that
// holds the call (its paragraph or list item), with a «×» that closes it; Esc closes it too and gives the focus back
// to the call. Each note opens and closes on its own, and the pops of one block keep the order of their calls. The
// calls get their button semantics here (role, `aria-controls`, `aria-expanded`, Space): without JS they are plain
// links that jump to the note.

const NOTE_PREFIX = '#nota-';
// The blocks a pop can follow, from the call outwards: the first one found holds it.
const BLOCKS = ['li', 'p', 'td', 'dd', 'blockquote'];

/**
 * The key of a note from the href of its call.
 *
 * @param {string} href the href of the call (`#nota-{key}`)
 * @returns {string|null} the key, or `null` when the href is not a note
 */
export function noteKeyOf(href) {
  return href && href.startsWith(NOTE_PREFIX) && href.length > NOTE_PREFIX.length ? href.slice(NOTE_PREFIX.length) : null;
}

/**
 * Which ancestor of a call the pop of its note follows.
 *
 * @param {string[]} ancestors the tag names of the ancestors of the call, lower case, from the closest one
 * @returns {number} the index of the first block (`BLOCKS`), or -1 when there is none
 */
export function noteAnchorBlock(ancestors) {
  return ancestors.findIndex((tag) => BLOCKS.includes(tag));
}

/**
 * The ancestors of an element, from its parent outwards.
 *
 * @param {Element} element the element
 * @returns {Element[]} its ancestors
 */
function ancestorsOf(element) {
  const out = [];
  for (let node = element.parentElement; node; node = node.parentElement) {
    out.push(node);
  }
  return out;
}

/**
 * Close the pop of a call and give the focus back to it.
 *
 * @param {HTMLAnchorElement} call the call
 * @param {HTMLElement} pop its pop
 * @param {boolean} focus whether to focus the call
 */
function closePop(call, pop, focus) {
  pop.remove();
  call.setAttribute('aria-expanded', 'false');
  if (focus) {
    call.focus();
  }
}

/**
 * Open the pop of a call after its block, with a copy of the note.
 *
 * @param {Document} doc the document
 * @param {HTMLAnchorElement} call the call
 * @param {Element} note the note (`li#nota-{key}`)
 * @param {string} key the key of the note
 */
function openPop(doc, call, note, key) {
  const ancestors = ancestorsOf(call);
  const index = noteAnchorBlock(ancestors.map((node) => node.tagName.toLowerCase()));
  const block = index >= 0 ? ancestors[index] : call;
  // after a list item the pop is a list item too, so the list stays valid
  const pop = doc.createElement(block.tagName === 'LI' ? 'li' : 'div');
  pop.className = 'note-pop';
  pop.id = `nota-${key}-open`;
  pop.setAttribute('role', 'note');

  const close = doc.createElement('button');
  close.type = 'button';
  close.className = 'note-close';
  close.setAttribute('aria-label', 'Cerrar la nota');
  close.textContent = '×';
  close.addEventListener('click', () => closePop(call, pop, true));

  const body = doc.createElement('div');
  body.className = 'note-body';
  const number = doc.createElement('strong');
  number.textContent = `${call.textContent}. `;
  body.append(number);
  note.childNodes.forEach((child) => body.append(child.cloneNode(true)));

  pop.dataset.note = key;
  pop.append(close, body);
  pop.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closePop(call, pop, true);
    }
  });

  // the pops of one block keep the order of their calls: before the first open pop whose call comes later
  let after = block;
  for (let next = block.nextElementSibling; next && next.classList.contains('note-pop'); next = next.nextElementSibling) {
    const other = doc.getElementById(`llamada-${next.dataset.note}`);
    if (other && call.compareDocumentPosition(other) & Node.DOCUMENT_POSITION_FOLLOWING) {
      break;
    }
    after = next;
  }
  after.after(pop);
  call.setAttribute('aria-expanded', 'true');
}

/**
 * Wire every footnote call of a page.
 *
 * @param {Document} [root] the document
 */
export function wireNotes(root = document) {
  root.querySelectorAll('.note-ref a[href^="#nota-"]').forEach((call) => {
    const key = noteKeyOf(call.getAttribute('href'));
    const note = key && root.getElementById(`nota-${key}`);
    if (!note) {
      return;
    }

    call.setAttribute('role', 'button');
    call.setAttribute('aria-controls', `nota-${key}-open`);
    call.setAttribute('aria-expanded', 'false');
    // a button opens with Space too (a link only with Enter)
    call.addEventListener('keydown', (event) => {
      if (event.key === ' ') {
        event.preventDefault();
        call.click();
      }
    });
    call.addEventListener('click', (event) => {
      event.preventDefault();
      const open = root.getElementById(`nota-${key}-open`);
      if (open) {
        closePop(call, open, false);
      } else {
        openPop(root, call, note, key);
      }
    });
  });
}
