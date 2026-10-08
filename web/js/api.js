// API client: every network request of the site goes through this module.

const cache = new Map();

/** Error raised for any failed API call; status 0 means no connection. */
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Origin of the API: '' (same origin) unless `?api=` points to a local server (development only). */
export function apiBase() {
  const value = new URLSearchParams(location.search).get('api');
  if (!value) {
    return '';
  }
  try {
    const url = new URL(value);
    if (url.protocol === 'http:' && (url.hostname === 'localhost' || url.hostname === '127.0.0.1')) {
      return url.origin;
    }
  } catch (error) {
    return '';
  }
  return '';
}

/** GET a JSON document from the API; parsed responses are cached per page. */
export async function getJSON(path) {
  const url = apiBase() + path;
  if (cache.has(url)) {
    return cache.get(url);
  }
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new ApiError(0, 'Sin conexión con la API');
  }
  if (!response.ok) {
    let message = null;
    try {
      const body = await response.json();
      message = body.message || null;
    } catch (error) {
      message = null;
    }
    throw new ApiError(response.status, message || response.statusText);
  }
  const data = await response.json();
  cache.set(url, data);
  return data;
}

export async function getManifest() {
  return getJSON('/api/v1/manifest');
}

export async function getScopes() {
  return getJSON('/api/v1/scopes');
}

/** Build the path of a forecast part, with optional mode, run and format query. */
export function forecastUrl(scope, part, {mode = null, run = null, format = null} = {}) {
  let path;
  if (part === 'meta') {
    path = `/api/v1/forecast/${scope}`;
  } else if (part === 'runs') {
    path = `/api/v1/forecast/${scope}/runs`;
  } else if (mode) {
    path = `/api/v1/forecast/${scope}/${mode}/${part}`;
  } else {
    path = `/api/v1/forecast/${scope}/${part}`;
  }
  const query = new URLSearchParams();
  if (run) {
    query.set('run', run);
  }
  if (format) {
    query.set('format', format);
  }
  const text = query.toString();
  return text ? `${path}?${text}` : path;
}

export async function getForecast(scope, part, opts) {
  return getJSON(forecastUrl(scope, part, opts));
}
