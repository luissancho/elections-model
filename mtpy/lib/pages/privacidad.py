"""
Privacy page (``/privacidad``) and the settings of the measurement of visits that every page carries in its cookie
notice (``_consent.html``).

The visits are measured with Google Analytics, served by a Google Tag Manager container (``WEB_GTM_ID``), and only
for the readers who accept: the notice and its module (``web/dist/js/consent.js``) load nothing of Google before.
Without a container the pages carry neither.
"""
from typing import Optional

from ..webapi import settings
from .common import common_context

# Months the choice of a reader (accept or reject) is kept in the browser before asking again; the notice hands it
# to ``consent.js`` and the privacy page tells it
CONSENT_MONTHS = 12


def analytics_config() -> Optional[dict]:
    """
    Settings of the measurement of visits, for the cookie notice of the pages.

    Returns
    -------
    dict or None
        ``{'gtm_id', 'months'}``: the Tag Manager container (``settings()['gtm_id']``) and ``CONSENT_MONTHS``;
        ``None`` without a container, which leaves the pages without notice and without measurement.
    """
    gtm_id = settings()['gtm_id']

    return {'gtm_id': gtm_id, 'months': CONSENT_MONTHS} if gtm_id else None


def privacidad_context(state: dict) -> dict:
    """
    Template context of the privacy page.

    The text lives in ``privacidad.html``; the page is not in the main menu and lights no entry of the tab bar.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).

    Returns
    -------
    dict
        The common context plus ``consent_months`` (``CONSENT_MONTHS``).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    context = common_context(state, None)
    context['consent_months'] = CONSENT_MONTHS

    return context
