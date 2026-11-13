// State shared in the URL fragment: each control owns some keys and rewrites only those.

/**
 * Tokens of a fragment: its `&`-separated parts, without the leading `#` and without empty ones.
 *
 * @param {string} hash the fragment, with or without the leading `#`
 * @returns {string[]} the tokens, in order
 */
function tokens(hash) {
  return (hash || '').replace(/^#/, '').split('&').filter((token) => token !== '');
}

/**
 * Key of a `key=value` token.
 *
 * @param {string} token a fragment token
 * @returns {string|null} the key, null for a loose token (without `=`)
 */
function keyOf(token) {
  const at = token.indexOf('=');
  return at > 0 ? token.slice(0, at) : null;
}

/**
 * Fragment from tokens.
 *
 * @param {string[]} parts the tokens
 * @returns {string} `#` plus the tokens joined by `&`, or `''` without tokens
 */
function join(parts) {
  return parts.length ? `#${parts.join('&')}` : '';
}

/**
 * Replace the pairs of some keys in the current fragment, keeping everything else.
 *
 * @param {string} current the current fragment
 * @param {string} fragment the new fragment of those keys (may be empty)
 * @param {string[]} keys the keys the `fragment` owns
 * @returns {string} `current` without the pairs of `keys`, plus the pairs of `fragment` at the end; the loose token
 *   and the other pairs keep their order; `''` when nothing is left
 */
export function mergeHash(current, fragment, keys) {
  const kept = tokens(current).filter((token) => !keys.includes(keyOf(token)));
  return join(kept.concat(tokens(fragment)));
}

/**
 * Set the section anchor of a fragment.
 *
 * @param {string} current the current fragment
 * @param {string|null} id the section id, null for none
 * @returns {string} `id` as the only loose token, first, followed by the pairs of `current`
 */
export function withSection(current, id) {
  const pairs = tokens(current).filter((token) => keyOf(token) !== null);
  return join(id ? [id].concat(pairs) : pairs);
}

/**
 * Section to scroll to.
 *
 * @param {string} search the query string (`location.search`)
 * @param {string} hash the fragment (`location.hash`)
 * @param {string[]} ids the valid section ids
 * @returns {string|null} the `seccion` of the query when it is in `ids`; else the first loose token of the fragment
 *   in `ids`; else null
 */
export function sectionTarget(search, hash, ids) {
  const wanted = new URLSearchParams(search || '').get('seccion');
  if (wanted !== null && ids.includes(wanted)) {
    return wanted;
  }
  return tokens(hash).find((token) => keyOf(token) === null && ids.includes(token)) || null;
}

/**
 * Where to land for a section asked for: the territory card (`districts`) when the section is the seats one
 * (`escanos`), the query names a district (`region`: the links and the form of the territory table, an old seats
 * address) and the page has that card, since the section opens with «Escaños que bailan», which shows another
 * district; else the section itself.
 *
 * @param {string|null} section the section asked for (`sectionTarget`)
 * @param {string} search the query string (`location.search`)
 * @param {boolean} hasDistricts whether the page has the territory card
 * @returns {string|null} the id to scroll to, null when no section is asked for
 */
export function sectionLanding(section, search, hasDistricts) {
  if (section === 'escanos' && hasDistricts && new URLSearchParams(search || '').has('region')) {
    return 'districts';
  }
  return section;
}

/**
 * Remove a parameter from a query string.
 *
 * @param {string} search the query string (`location.search`)
 * @param {string} name the parameter to remove
 * @returns {string} the remaining query with `?`, or `''` when none is left
 */
export function dropParam(search, name) {
  const params = new URLSearchParams(search || '');
  params.delete(name);
  const text = params.toString();
  return text ? `?${text}` : '';
}
