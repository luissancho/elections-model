"""Las notas al pie desplegables de `_macros.html`: la llamada numerada por el orden de una lista de claves, la tarjeta
«Notas» y los enlaces de ida y vuelta entre las dos, que funcionan sin JavaScript."""
import re

import pytest

from mtpy.lib import pages

KEYS = ['tope', 'diseno']
SOURCE = ("{% import '_macros.html' as m %}<p>Uno{{ m.note('tope', keys) }}</p><ul><li>Dos{{ m.note('diseno', keys) }}</li></ul>"
          "{% call m.notes_card(keys) %}{% call m.note_item('tope') %}Nota A{% endcall %}"
          "{% call m.note_item('diseno') %}Nota B{% endcall %}{% endcall %}")


def test_note_calls_are_numbered_by_the_order_of_the_keys_and_link_to_their_notes():
    page = pages.environment().from_string(SOURCE).render(keys=KEYS)
    assert re.findall(r'<a id="llamada-([a-z]+)" href="#nota-\1"[^>]*>(\d+)</a>', page) == [('tope', '1'), ('diseno', '2')]
    assert 'id="nota-tope"' in page and 'id="nota-diseno"' in page
    assert 'href="#llamada-' not in page and 'note-back' not in page  # no link back from the notes (2026-10-22)
    assert page.index('id="nota-tope"') < page.index('id="nota-diseno"') and 'id="notas"' in page
    # without JS the call is a plain link: the button semantics are set by notes.js
    assert 'aria-expanded' not in page and 'aria-controls' not in page and 'role="button"' not in page
    assert re.search(r'<ol class="notes">\s*<li id="nota-tope">Nota A', page)


def test_an_unknown_note_key_fails_the_render():
    with pytest.raises(Exception):
        pages.environment().from_string("{% import '_macros.html' as m %}{{ m.note('x', ['a']) }}").render()
