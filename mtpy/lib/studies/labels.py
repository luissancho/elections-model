"""
Public labels of the elections the studies name, shared by every study.

The months are abbreviated as the site does (``filters.MONTHS``, copied: the studies keep independent of the web
layer).
"""
NATIONAL = 'es'
MONTHS = ('ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sept', 'oct', 'nov', 'dic')


def election_code(scope: str, event_date: str, month: bool = False) -> str:
    """
    Short code of an election: ``GEN 23`` for the national scope, ``AN 18`` for ``es-an`` 2018.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    month : bool, default False
        Add the initial of the month after the year (``GEN 19A`` and ``GEN 19N`` for the two general elections of
        2019).

    Returns
    -------
    str
        The code.
    """
    code = 'GEN' if scope == NATIONAL else scope.upper().removeprefix('ES-')
    initial = MONTHS[int(event_date[5:7]) - 1][0].upper() if month else ''

    return f'{code} {event_date[2:4]}{initial}'


def election_label(scope: str, name: str, event_date: str, month: bool = False) -> str:
    """
    Label of an election: ``Generales 2023`` for the national scope, ``Andalucía 2018`` for the rest.

    Parameters
    ----------
    scope : str
        Scope code.
    name : str
        Name of the scope (unused for the national scope).
    event_date : str
        Election date, ``YYYY-MM-DD``.
    month : bool, default False
        Put the abbreviated month before the year (``Generales abr. 2019`` and ``Generales nov. 2019``).

    Returns
    -------
    str
        The label.
    """
    year = event_date[:4]
    when = f'{MONTHS[int(event_date[5:7]) - 1]}. {year}' if month else year

    return f'Generales {when}' if scope == NATIONAL else f'{name} {when}'


def short_year_label(event_date: str, repeated: bool) -> str:
    """
    Label of an election by its year alone: ``2023``; ``2019 abr.`` when another election of the same scope shares
    the year (the month as the site abbreviates it, ``sept.`` included, like ``herd.data.cycle_label``).

    Parameters
    ----------
    event_date : str
        Election date, ``YYYY-MM-DD``.
    repeated : bool
        Whether another election labelled together falls in the same year.

    Returns
    -------
    str
        The label.
    """
    year = event_date[:4]

    return f'{year} {MONTHS[int(event_date[5:7]) - 1]}.' if repeated else year
