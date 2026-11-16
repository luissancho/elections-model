"""
The rules the link previews of the pages that do not change share (the page of a poll and the detail of a past
election): how many parties their image shows, the version of its drawing, the fingerprint of what it draws, how a
change is told and the figures that fit a description. Each page composes its preview as plain data
(``sondeos.preview_of``, ``elecciones.result_preview``) and ``pages.images`` draws it.
"""
import hashlib
import json
from typing import Optional, Sequence

from .filters import SNIPPET_LENGTH, fmt_num

# The parties an image shows and the version of its drawing. The version goes into the fingerprint of the address
# of every image: bump it when the drawing changes, so the networks that keep an image by its address ask for the
# new one
PREVIEW_ROWS = 6
PREVIEW_DESIGN = 1


def preview_change(change: Optional[float]) -> Optional[dict]:
    """
    Change of a party against what its row is compared with, as an image shows it.

    Parameters
    ----------
    change : float or None
        The change in points, with one decimal (against the previous delivery of the house in the image of a poll,
        against the previous election in the image of a past election).

    Returns
    -------
    dict or None
        ``direction`` (``'up'``, ``'down'`` or ``'same'``) and ``text`` (the points without sign, ``'0,4'``, or
        ``'='`` when nothing changes); ``None`` when there is nothing to compare with.
    """
    if change is None:
        return None

    if change == 0:
        return {'direction': 'same', 'text': '='}

    return {'direction': 'up' if change > 0 else 'down', 'text': fmt_num(abs(change), 1)}


def preview_version(drawn: dict) -> str:
    """
    Fingerprint of what an image draws, for the ``?v=`` of its address.

    Parameters
    ----------
    drawn : dict
        Every text and figure the image draws (its title, its grey line, its rows and its footnote).

    Returns
    -------
    str
        8 hex digits of the MD5 of ``PREVIEW_DESIGN`` and of ``drawn``: the same for the same drawing, another
        one when a figure or the version of the drawing changes.
    """
    fingerprint = json.dumps([PREVIEW_DESIGN, drawn], sort_keys=True, ensure_ascii=False).encode('utf-8')

    return hashlib.md5(fingerprint).hexdigest()[:8]


def with_figures(lead: str, figures: Sequence[str], limit: int = SNIPPET_LENGTH) -> str:
    """
    A description that says figures: as many as fit whole after its lead.

    Parameters
    ----------
    lead : str
        What the description starts with, with its trailing space (``'Generales 2023: '``).
    figures : sequence of str
        The figures, in the order to tell them (``'PP 33,1%'``).
    limit : int, optional
        Characters the description has at most.

    Returns
    -------
    str
        ``lead`` and the leading figures that fit, joined with ``', '`` and closed with a full stop; a figure is
        told whole or not at all. ``lead`` without its trailing space when no figure fits or there is none.
    """
    kept = []

    for figure in figures:
        if len(lead + ', '.join(kept + [figure]) + '.') > limit:
            break

        kept.append(figure)

    return lead + ', '.join(kept) + '.' if kept else lead.rstrip()
