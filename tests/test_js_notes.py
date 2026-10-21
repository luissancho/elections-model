"""Funciones puras de las notas al pie desplegables (`web/dist/js/notes.js`), comprobadas con `node` (se salta sin
él). El comportamiento en la página (abrir, cerrar, Esc) se comprueba en el navegador."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTES_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'notes.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    """Evalúa `expr` con `n` (el módulo) definido y devuelve su JSON."""
    script = 'import * as n from "{}"; console.log(JSON.stringify({}));'.format(NOTES_JS, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_note_key_of_reads_only_note_hrefs():
    assert run_js('[n.noteKeyOf("#nota-tope"), n.noteKeyOf("#otra"), n.noteKeyOf("")]') == ['tope', None, None]


def test_the_pop_goes_after_the_list_item_or_the_paragraph():
    assert run_js('n.noteAnchorBlock(["sup", "li", "ul", "section"])') == 1
    assert run_js('n.noteAnchorBlock(["sup", "strong", "p", "section"])') == 2
    assert run_js('n.noteAnchorBlock(["sup", "section"])') == -1
