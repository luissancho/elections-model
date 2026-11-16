// Consent for the measurement of visits (Google Tag Manager, which serves Google Analytics). Nothing of Google is
// loaded, and no cookie is written, until the reader accepts in the notice (`_consent.html`); the choice is kept in
// the local storage of the browser for the months the notice declares and asked again after them. Rejecting later
// (from the privacy page) deletes the cookies of Google and reloads the page, so that the container is gone.
//
// The container is loaded as the snippet of Google does (`gtm.js`, `dataLayer`, `gtag('js', new Date())`), from
// this module instead of an inline script because the security policy of the site allows no inline script, and
// with the signals of Consent Mode v2 before it: advertising always denied (the site has none) and the
// measurement granted.

/** Key of the choice in the local storage. */
export const STORAGE_KEY = 'consent';

const GTM_SCRIPT = 'https://www.googletagmanager.com/gtm.js';
const GTM_ID = /^GTM-[A-Z0-9]+$/;
// Cookies of Google Analytics (`_ga`, `_ga_<ID>`) and the ones of older tags a container might add
const ANALYTICS_COOKIE = /^_(ga|gid|gat|gcl)(_|$)/;
const CHOICES = ['granted', 'denied'];

/**
 * Choice kept in the local storage, if it still holds.
 *
 * @param {string|null} raw the stored value (`storedChoice`), or null when nothing is stored
 * @param {Date} now the current time
 * @param {number} months months a choice is valid for
 * @returns {string|null} `'granted'` or `'denied'`; null without a choice, when it cannot be read (another
 *   format, another value, a broken or future date) or when it is `months` old or older
 */
export function readChoice(raw, now, months) {
  let stored = null;
  try {
    stored = JSON.parse(raw);
  } catch (error) {
    return null;
  }
  if (!stored || !CHOICES.includes(stored.analytics) || typeof stored.at !== 'string') {
    return null;
  }
  const at = new Date(stored.at);
  if (Number.isNaN(at.getTime()) || at > now || !(months > 0)) {
    return null;
  }
  const end = new Date(at);
  end.setUTCMonth(end.getUTCMonth() + months);
  return now < end ? stored.analytics : null;
}

/**
 * Value to keep in the local storage for a choice.
 *
 * @param {string} choice `'granted'` or `'denied'`
 * @param {Date} now the time of the choice
 * @returns {string} JSON of `{analytics, at}`, `at` being the ISO time
 */
export function storedChoice(choice, now) {
  return JSON.stringify({analytics: choice, at: now.toISOString()});
}

/**
 * Signals of Consent Mode v2 for a choice.
 *
 * @param {string|null} choice `'granted'`, `'denied'` or null
 * @returns {object} the advertising signals always denied and `analytics_storage` granted only for `'granted'`
 */
export function consentSignals(choice) {
  return {
    ad_storage: 'denied',
    ad_user_data: 'denied',
    ad_personalization: 'denied',
    analytics_storage: choice === 'granted' ? 'granted' : 'denied',
  };
}

/**
 * URL of the script of a Tag Manager container.
 *
 * @param {string} id container id (`GTM-…`)
 * @returns {string|null} the URL of `gtm.js` for the container, or null when `id` is not a container id
 */
export function gtmUrl(id) {
  return typeof id === 'string' && GTM_ID.test(id) ? `${GTM_SCRIPT}?id=${id}` : null;
}

/**
 * Names of the cookies of Google in a cookie string.
 *
 * @param {string} cookie `document.cookie`
 * @returns {string[]} the names that match `ANALYTICS_COOKIE`, in order
 */
export function analyticsCookies(cookie) {
  return (cookie || '').split(';')
    .map((pair) => pair.split('=')[0].trim())
    .filter((name) => ANALYTICS_COOKIE.test(name));
}

/**
 * Domains a cookie of the site may have been set on: Google Analytics writes on the highest one it can.
 *
 * @param {string} hostname host of the page
 * @returns {string[]} the host and each of its parents with two labels or more, with a leading dot; none for a
 *   host of one label (`localhost`)
 */
export function cookieDomains(hostname) {
  const labels = (hostname || '').split('.').filter((label) => label !== '');
  const domains = [];
  for (let start = 0; start + 2 <= labels.length; start += 1) {
    domains.push(`.${labels.slice(start).join('.')}`);
  }
  return domains;
}

/**
 * Delete the cookies of Google of the site, on the host and on every domain they may live on.
 *
 * @param {Document} doc the document
 * @param {string} hostname host of the page
 */
function forgetCookies(doc, hostname) {
  const domains = cookieDomains(hostname);
  analyticsCookies(doc.cookie).forEach((name) => {
    const expired = `${name}=; Max-Age=0; path=/`;
    doc.cookie = expired;
    domains.forEach((domain) => {
      doc.cookie = `${expired}; domain=${domain}`;
    });
  });
}

/**
 * Stored value of the choice; null when the browser gives no access to the storage.
 *
 * @param {Window} win the window
 * @returns {string|null} the stored value
 */
function load(win) {
  try {
    return win.localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

/**
 * Keep the choice; when the browser does not let it, the choice only lasts for the page.
 *
 * @param {Window} win the window
 * @param {string} value the value to store (`storedChoice`)
 */
function save(win, value) {
  try {
    win.localStorage.setItem(STORAGE_KEY, value);
  } catch (error) {
    // private mode or blocked storage: the reader is asked again on the next page
  }
}

/**
 * Wire the notice of the page (`#consent`): show it while there is no valid choice, load the container when the
 * reader has accepted and attend the buttons `[data-consent]` (of the notice and of the privacy page). The texts
 * `[data-consent-when]` show the one of the current choice (`granted`, `denied` or `unset`). A page restored from
 * the back/forward cache reads the choice again, because another page may have changed it. Without the notice,
 * or with an id that is not a container, it does nothing.
 *
 * @param {object} [env] the environment, for the tests
 * @param {Document} [env.doc] the document
 * @param {Window} [env.win] the window (`localStorage`, `location`, `dataLayer`)
 * @param {Function} [env.clock] returns the current time
 */
export function wireConsent({doc = document, win = window, clock = () => new Date()} = {}) {
  const notice = doc.getElementById('consent');
  const url = notice ? gtmUrl(notice.dataset.gtmId) : null;
  if (!url) {
    return;
  }
  const months = Number(notice.dataset.months);
  let choice = readChoice(load(win), clock(), months);
  let started = false;

  const start = () => {
    if (started) {
      return;
    }
    started = true;
    win.dataLayer = win.dataLayer || [];
    // the queue of Google tells commands from events by the `arguments` object: an array would be ignored
    function gtag() {
      win.dataLayer.push(arguments);
    }
    gtag('consent', 'default', consentSignals('granted'));
    gtag('js', new Date());
    const script = doc.createElement('script');
    script.async = true;
    script.src = url;
    doc.head.appendChild(script);
  };

  const show = () => {
    notice.hidden = choice !== null;
    doc.querySelectorAll('[data-consent-when]').forEach((node) => {
      node.hidden = node.dataset.consentWhen !== (choice || 'unset');
    });
  };

  const choose = (value) => {
    if (!CHOICES.includes(value)) {
      return;
    }
    choice = value;
    save(win, storedChoice(value, clock()));
    if (value === 'granted') {
      start();
    } else {
      forgetCookies(doc, win.location.hostname);
      if (started) {
        // the container cannot be unloaded: a fresh page without it is the only clean way out
        win.location.reload();
      }
    }
    show();
  };

  if (choice === 'granted') {
    start();
  }
  doc.querySelectorAll('[data-consent]').forEach((button) => {
    button.addEventListener('click', () => choose(button.dataset.consent));
  });
  win.addEventListener('pageshow', (event) => {
    if (!event.persisted) {
      return;
    }
    choice = readChoice(load(win), clock(), months);
    if (choice === 'granted') {
      start();
    } else if (started) {
      win.location.reload();
    }
    show();
  });
  show();
}
