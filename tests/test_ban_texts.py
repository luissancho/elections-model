"""Tests de `mtpy/lib/studies/ban/texts.py`: la frase del estudio `andorra` que comparten el subtítulo y la página."""
import pytest

from mtpy.lib.studies.ban.texts import count_word, lead_sentence

POLLS = ('el consenso de los sondeos hechos esos días, los que salen fuera y los que esperan al cierre de las urnas, ')
TAIL = ', y quedó más cerca del resultado en 18 de las 24 cifras de partido.'


@pytest.mark.parametrize('n, k, want', [
    (5, 5, 'En las cinco generales desde 2015, ' + POLLS + 'estuvo más cerca del resultado que el promedio congelado '
           'en todas' + TAIL),
    (5, 4, 'En las cinco generales desde 2015, ' + POLLS + 'estuvo más cerca del resultado que el promedio congelado '
           'en cuatro' + TAIL),
    (5, 0, 'En las cinco generales desde 2015, ' + POLLS + 'estuvo más cerca del resultado que el promedio congelado '
           'en ninguna' + TAIL),
    (1, 1, 'En la única general desde 2015, ' + POLLS + 'estuvo más cerca del resultado que el promedio congelado'
           + TAIL),
    (1, 0, 'En la única general desde 2015, ' + POLLS + 'no estuvo más cerca del resultado que el promedio congelado'
           + TAIL),
    (12, 11, 'En las 12 generales desde 2015, ' + POLLS + 'estuvo más cerca del resultado que el promedio congelado '
             'en 11' + TAIL),
])
def test_lead_sentence_counts_the_elections_in_words(n, k, want):
    """La frase dice que el consenso junta los sondeos que salen fuera y los que esperan al cierre de las urnas, y
    cuenta las generales con palabras: «en todas», «en ninguna», «la única general» (nunca «las una»)."""
    assert lead_sentence(n, k, '2015', 18, 24) == want


def test_count_word_is_feminine_and_falls_back_on_the_figure():
    assert [count_word(n) for n in (0, 1, 2, 10, 11)] == ['ninguna', 'una', 'dos', 'diez', '11']
