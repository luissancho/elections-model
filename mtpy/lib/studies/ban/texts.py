"""
Public sentences of the clandestine polls study that more than one layer writes.

The published subtitle (``publish.ban``: the card of ``/analisis``, the description and the links) and the lead of the
page (``pages.ban``) say the same sentence about the general elections with a model, so it lives here, below both.
"""
# A small count in words, feminine («cinco generales», «en una»): the index is the count
COUNT_WORDS = ('ninguna', 'una', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez')


def count_word(n: int) -> str:
    """
    A small count in words, feminine.

    Parameters
    ----------
    n : int
        The count.

    Returns
    -------
    str
        «cinco», «una», «ninguna» for 0; the figure from eleven on.
    """
    return COUNT_WORDS[n] if isinstance(n, int) and 0 <= n < len(COUNT_WORDS) else str(n)


def lead_sentence(n: int, k: int, year: str, k_closer, n_cells) -> str:
    """
    The sentence of the general elections with a model: where the clandestine consensus beat the frozen average and
    how many party figures ended closer to the result.

    The polls of the consensus are every poll made during the ban, and the sentence says so: the ones published
    abroad on those days and the ones kept until the polling stations close (the polls of the election night).

    Parameters
    ----------
    n : int
        General elections with a model.
    k : int
        Those where the consensus beat the frozen average.
    year : str
        Year of the first of them («2015»).
    k_closer : int or str
        Party figures where the consensus ended closer to the result.
    n_cells : int or str
        Party figures compared.

    Returns
    -------
    str
        «En las cinco generales desde 2015, el consenso de los sondeos hechos esos días, …, estuvo más cerca del
        resultado que el promedio congelado en todas, y quedó más cerca del resultado en 18 de las 24 cifras de
        partido.»; «en cuatro» or «en ninguna» as ``k`` says, and with a single election «En la única general…,
        estuvo (no estuvo) más cerca…» without «en todas» or «en ninguna».
    """
    if n == 1:
        elections, verb, where = 'la única general', 'estuvo' if k == 1 else 'no estuvo', ''
    else:
        elections, verb = 'las {} generales'.format(count_word(n)), 'estuvo'
        where = ' en todas' if k == n else ' en ' + count_word(k)

    return ('En {} desde {}, el consenso de los sondeos hechos esos días, los que salen fuera y los que esperan al '
            'cierre de las urnas, {} más cerca del resultado que el promedio congelado{}, y quedó más cerca del '
            'resultado en {} de las {} cifras de partido.').format(elections, year, verb, where, k_closer, n_cells)
