"""
Method page context: the fixed list of sections of the static explanation of the model and its subpages.
"""
from typing import Optional

from .common import common_context

SECTIONS = (
    ('datos', 'Los datos'),
    ('promedio', 'El promedio'),
    ('casas', 'Las casas'),
    ('simulacion', 'La simulación'),
    ('escanos', 'Los escaños'),
    ('evaluacion', 'La evaluación'),
    ('publicacion', 'La publicación'),
    ('limitaciones', 'Limitaciones'),
)
# Subpages of the method (``/metodo/{id}``), in menu order. A subpage without ``template`` has no text yet: it
# is not linked and its route is a 404.
METODO_SECTIONS = {
    'promedio': {'title': '¿Cómo promediamos las encuestas?', 'template': None},
    'escanos': {'title': '¿Cómo simulamos los escaños?', 'template': None},
    'rating': {'title': '¿Cómo calculamos el rating de las encuestadoras?', 'template': None},
    'backtest': {'title': '¿Cuánto acierta el modelo?', 'template': 'metodo_backtest.html'},
}


def subpages(active: Optional[str]) -> list:
    """
    Subnavigation of the method.

    Parameters
    ----------
    active : str or None
        Id of the current subpage, ``None`` on the method landing page.

    Returns
    -------
    list of dict
        ``{'id', 'title', 'href', 'active', 'available'}`` per entry of ``METODO_SECTIONS``, in
        order; ``available`` tells whether the subpage has a text (``template``).
    """
    return [
        {'id': section, 'title': entry['title'], 'href': '/metodo/' + section, 'active': section == active,
         'available': entry['template'] is not None}
        for section, entry in METODO_SECTIONS.items()
    ]


def method_href(section: str) -> str:
    """
    Link to a subpage of the method, for the closing cards and the event bar.

    Parameters
    ----------
    section : str
        Id of the subpage in ``METODO_SECTIONS``.

    Returns
    -------
    str
        ``'/metodo/{section}'`` when the subpage has a text, otherwise ``'/metodo'``.
    """
    entry = METODO_SECTIONS.get(section)

    return '/metodo/' + section if entry and entry['template'] is not None else '/metodo'


def metodo_context(state: dict, active: str = '/metodo') -> dict:
    """
    Template context of the method page.

    The text lives in ``metodo.html``; the context only adds the ordered sections, which the template
    uses for the navigation and for the anchors. The page has no ``initial`` data; its module
    (``metodo.js``) only wires the header controls.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``sections``, a list of ``{'id', 'title'}`` in ``SECTIONS`` order.

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    context = common_context(state, active)
    context['sections'] = [{'id': section_id, 'title': title} for section_id, title in SECTIONS]

    return context
