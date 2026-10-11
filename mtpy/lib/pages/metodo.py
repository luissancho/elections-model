"""
Method page context: the fixed list of sections of the static explanation of the model.
"""
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


def metodo_context(state: dict, active: str = '/metodo') -> dict:
    """
    Template context of the method page.

    The text lives in ``metodo.html``; the context only adds the ordered sections, which the template
    uses for the navigation and for the anchors. The page has no ``initial`` data and no script.

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
