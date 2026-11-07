# Rating: tope de la calidad previa para casas sin elecciones. Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que la calidad previa de una encuestadora sin elecciones no pueda superar la mediana de los ratings de la tabla, y que el tramo de calidad por encima de esa mediana sólo se acredite en proporción a su evidencia.

**Architecture:** Un parámetro `quality_cap` del `Computer` y una segunda pasada dentro de `Computer.pollster_ratings` (`mtpy/lib/computer.py`), que calcula el listón sobre los ratings de la primera pasada y vuelve a mezclar con la calidad efectiva. Nada nuevo en la base ni en el `Forecaster`. Textos públicos en /metodo/rating, /casas y el tooltip de «Confianza»; método 17 en `docs/analisis-2026-09`; comprobación con dos backtests de `es` emparejados (ratings recalculados en memoria, sin tocar la base).

**Tech Stack:** Python 3.11 (`/Users/luiss/.pyenv/versions/3.11.9/bin/python`), pandas/numpy, pytest, Jinja2.

**Spec:** `docs/superpowers/specs/2026-11-07-rating-tope-calidad-previa-design.md`

## Global Constraints

- Fórmula: `R0 = (W·R_aj + k·Q)/(W + k)`, `k = (σ/τ)² = (0,014/0,01)² = 1,96`; listón `c = mediana(R0)` (o media con `'mean'`), sobre las casas con rating (las de `drop_mtypes` quedan NaN y fuera); `Q_ef = min(Q, c) + max(Q − c, 0)·W/(W + k)`; `R = (W·R_aj + k·Q_ef)/(W + k)`; `weight_rating = log1p(R)/log1p(media de R)`.
- `Computer(quality_cap='median')`, valores `'median'`, `'mean'`, `None`; sin parámetro en `params.json`; sin columnas nuevas en `pollsters_ratings` (nada de `ALTER TABLE`).
- Identificadores en inglés neutro; docstrings numpy en todas las funciones; PEP8.
- Textos públicos en español: «sólo» con tilde, encuestadora/casa, nunca «empresa»; la fórmula de `Q_ef` en la nota técnica, en el cuerpo una frase en palabras.
- Nunca escribir en `backtest/results`; comprobar con `ps aux | grep job.py` que no hay trabajos de Luis antes de correr nada largo; tests nunca con `out=None`.
- Luis commitea en paralelo en el mismo árbol: `git add` con rutas explícitas, nunca `-a`; no tocar `data/pollsters-quality.csv` ni las notas de evaluación (cambios suyos sin commit). Mensajes de commit en español, terminados en `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

- Una sola casa con rating en el cálculo: el listón es su propio `R0`, y si su calidad lo supera se recorta hacia su rating medido. Esperado (no hay tabla con la que compararla) y finito; test en la Tarea 1.
- Varias casas sin evidencia por encima de la mediana: todas caen al mismo listón (ninguna queda por encima de otra por su papeleo). Test en la Tarea 1.
- Casas de `drop_mtypes`: su `R0` es NaN; no entran en el listón ni reciben rating; `median()` las salta. Test en la Tarea 1.
- `quality_cap` con un valor no previsto: `ValueError` claro al calcular, no un `AttributeError` tardío. Test en la Tarea 1.
- Textos: el tope cambia la frase «una encuestadora sin sondeos evaluables tiene como rating su calidad previa» en el cuerpo y en la nota de la mezcla; la frase sobre el peso de reserva de las casas sin fila de rating (`R` por `Q`, `R̄` por la calidad media) sigue siendo cierta y se queda. Test en la Tarea 2.

---

### Task 1: `quality_cap` en el rating

**Files:**
- Modify: `mtpy/lib/computer.py:534-535` (firma del constructor), `:580-587` (docstring de los parámetros), `:614-615` (atributos), `:2326-2348` (`pollster_ratings`, mezcla y peso)
- Test: `tests/test_computer_unit.py` (`bare_computer` ll. 189-197; `test_rating_follows_the_grouped_formula` l. 296; tests nuevos tras la l. 310; fixture del panel online hacia la l. 609)

**Interfaces:**
- Produces: `Computer.__init__(..., bias_dev_sigma=.014, quality_cap: Optional[Literal['median', 'mean']] = 'median', ...)`, atributo `self.quality_cap`; `Computer.pollster_ratings` devuelve las mismas columnas de hoy con `rating` y `weight_rating` ya con el tope.

- [ ] **Step 1: Añadir `c.quality_cap = 'median'` a `bare_computer` y fijar `c.quality_cap = None` en `test_rating_follows_the_grouped_formula`** (ese test comprueba la mezcla sin tope; con el tope también pasaría, porque C y D valen 0,6 y la mediana es 0,6, pero debe decir lo que prueba).

- [ ] **Step 2: Escribir los tests del tope, tras `test_rating_follows_the_grouped_formula`**

```python
K = (.014 / .01) ** 2


def first_pass(out):
    """R0 de cada casa a partir de las columnas devueltas: la mezcla sin tope."""
    return (out['num_events_w'] * out['rating_adj'] + K * out['quality']) / (out['num_events_w'] + K)


def test_quality_above_the_median_is_only_credited_with_evidence():
    """B2: C (sin elecciones, calidad 90) baja a la mediana; D (calidad 20) no cambia; A (una elección,
    calidad 60) sólo acredita el tramo sobre la mediana en W/(W + k)."""
    c = bare_computer()
    c.pollsters['quality'] = [60., 60., 90., 20.]
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    r0 = first_pass(out)
    cap = r0.median()
    assert out.loc[3, 'num_events_w'] == 0 and out.loc[3, 'rating'] == pytest.approx(cap)
    assert out.loc[4, 'rating'] == pytest.approx(r0.loc[4]) and out.loc[4, 'rating'] == pytest.approx(.2)
    w = out.loc[1, 'num_events_w']
    q_eff = cap + (.6 - cap) * w / (w + K)
    assert out.loc[1, 'rating'] == pytest.approx((w * out.loc[1, 'rating_adj'] + K * q_eff) / (w + K))
    assert out.loc[1, 'rating'] < r0.loc[1]
    assert np.allclose(out['weight_rating'], np.log1p(out['rating']) / np.log1p(out['rating'].mean()))


def test_quality_cap_none_reproduces_the_plain_blend():
    c = bare_computer()
    c.quality_cap = None
    c.pollsters['quality'] = [60., 60., 90., 20.]
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    assert np.allclose(out['rating'], first_pass(out))


def test_quality_cap_mean_uses_the_mean_of_the_first_pass():
    c = bare_computer()
    c.quality_cap = 'mean'
    c.pollsters['quality'] = [60., 60., 90., 20.]
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    assert out.loc[3, 'rating'] == pytest.approx(first_pass(out).mean())


def test_quality_cap_rejects_an_unknown_value():
    c = bare_computer()
    c.quality_cap = 'max'
    with pytest.raises(ValueError, match='quality_cap'):
        c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters)


def test_a_single_rated_pollster_is_capped_by_its_own_rating():
    """Una sola casa con rating: el listón es su R0 y su calidad sobrante se acredita en W/(W + k)."""
    c = bare_computer()
    c.pollsters = c.pollsters.iloc[:1].copy()          # sólo A
    c.pollsters['quality'] = [90.]
    out = c.rate_events(pool([('es', '2019-11-10', 'A')]), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    r0 = first_pass(out).loc[1]
    w = out.loc[1, 'num_events_w']
    q_eff = r0 + (.9 - r0) * w / (w + K)
    assert np.isfinite(out.loc[1, 'rating'])
    assert out.loc[1, 'rating'] == pytest.approx((w * out.loc[1, 'rating_adj'] + K * q_eff) / (w + K))


def test_every_pollster_without_evidence_above_the_median_falls_to_it():
    """Con el ámbito `es` a peso 0 sólo C conserva evidencia (su sondeo de Madrid); A (0,8) y B (0,6), sin
    elecciones y por encima de la mediana de R0, caen las dos a la mediana; D (0,2) no cambia."""
    c = bare_computer()
    c.pollsters['quality'] = [80., 60., 40., 20.]
    out = c.rate_events(pool(ES19 + [('es-md', '2021-05-04', 'C')]), [JULY23], {'es': 0., 'es-md': 1.},
                        c.pollsters).set_index('pollster_id')
    assert (out.loc[[1, 2, 4], 'num_events_w'] == 0).all() and out.loc[3, 'num_events_w'] > 0
    cap = first_pass(out).median()
    assert out.loc[1, 'rating'] == pytest.approx(cap) and out.loc[2, 'rating'] == pytest.approx(cap)
    assert out.loc[4, 'rating'] == pytest.approx(.2)
```

- [ ] **Step 3: Añadir un test con una casa excluida (`drop_mtypes`), junto al fixture del panel online (hacia la l. 609)**

```python
def test_an_excluded_pollster_stays_out_of_the_quality_cap():
    """B es un panel online: sin rating, y fuera de la mediana que hace de listón."""
    c = online_computer()   # el fixture existente de esa sección, con B en drop_mtypes
    c.pollsters['quality'] = [60., 60., 90., 20.]
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    assert np.isnan(out.loc[2, 'rating'])
    cap = first_pass(out.drop(index=2)).median()
    assert out.loc[3, 'rating'] == pytest.approx(cap)
```

Usar el nombre real del fixture de esa sección (leerlo antes); si construye el `Computer` con `bare_computer()`, ya tendrá `quality_cap`.

- [ ] **Step 4: Correr los tests nuevos y comprobar que fallan**

Run: `/Users/luiss/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_computer_unit.py -q -k "quality_cap or capped or falls_to_it or credited"`
Expected: FAIL (`rating` igual a `first_pass`, o `AttributeError: quality_cap` si el fixture no se tocó).

- [ ] **Step 5: Implementar el parámetro y la segunda pasada en `mtpy/lib/computer.py`**

Constructor: `quality_cap: Optional[Literal['median', 'mean']] = 'median'` tras `bias_dev_sigma`; docstring: «Statistic of the first-pass ratings that caps the prior quality of a pollster without evidence (`'median'` by default, `'mean'`, or `None` for no cap): the quality above the cap is only credited in proportion `W / (W + k)` to its evidence (design 2026-11-07)»; `self.quality_cap = quality_cap`.

`pollster_ratings`, tras calcular `dr['rating']` y tras poner a NaN `rating_adj`/`rating` de las casas excluidas (ll. 2341-2344), y antes de `weight_rating`:

```python
if self.quality_cap is not None:
    if self.quality_cap not in ('median', 'mean'):
        raise ValueError("quality_cap: expected 'median', 'mean' or None, got {!r}".format(self.quality_cap))
    cap = getattr(dr['rating'], self.quality_cap)()          # NaN (excluded) skipped
    credit = dr['num_events_w'] / (dr['num_events_w'] + k)
    quality_eff = np.minimum(dr['quality'], cap) + (dr['quality'] - cap).clip(lower=0) * credit
    rated = dr['rating'].notna()
    dr.loc[rated, 'rating'] = (
        (dr['num_events_w'] * dr['rating_adj'] + k * quality_eff) / (dr['num_events_w'] + k)
    )[rated]
    if self.verbose > 1:
        print('Quality cap ({}): {:.4f}'.format(self.quality_cap, cap))
```

Ampliar la docstring de `pollster_ratings` (ll. 2241-2243) con la frase de `Q_ef` y el listón. Actualizar también la docstring de `compute_ratings` (l. 1483-1489, punto 4) con una línea.

- [ ] **Step 6: Correr toda la unidad del Computer y las páginas que leen la firma**

Run: `/Users/luiss/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_computer_unit.py tests/test_rating_check_unit.py -q`
Expected: PASS (incluido `test_prior_quality_is_taken_at_the_date_of_each_target`: con C y D a la misma calidad la mediana es esa calidad y el rating de C sigue siendo su calidad).

- [ ] **Step 7: Commit (spec, plan, código y tests)**

```bash
git add docs/superpowers/specs/2026-11-07-rating-tope-calidad-previa-design.md docs/superpowers/plans/2026-11-07-rating-tope-calidad-previa.md mtpy/lib/computer.py tests/test_computer_unit.py
git commit -m "Rating: la calidad previa de una casa sin elecciones no supera la mediana de los ratings (quality_cap='median'): el tramo por encima sólo se acredita en W/(W + k); spec y plan del 2026-11-07

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: textos públicos

**Files:**
- Modify: `web/templates/metodo_rating.html:340-341` (cuerpo de «La evidencia se cuenta por elecciones»), `:969` (nota técnica `rating`, «Contracción, nota y mezcla»)
- Modify: `web/templates/casas.html:58` (intro del ranking)
- Modify: `mtpy/lib/pages/casas.py:179` (tooltip `quality`)
- Test: `tests/test_pages.py::test_metodo_rating_text` (l. 3121) y el test del ranking de /casas que compruebe la intro (buscar `Cada fila es una encuestadora` en `tests/test_pages.py`; si no hay, añadir la aserción al test de /casas más cercano)

**Interfaces:**
- Consumes: nada de la Tarea 1 (son textos); el `assert_public_wording` de `tests/test_pages.py` recorre /metodo/rating en el bucle de la l. 1173.

- [ ] **Step 1: Añadir al test `test_metodo_rating_text` las frases nuevas y vetar la antigua**

En la lista de frases presentes: `'sólo lo da cuando hay elecciones que la respalden'`, `'la mediana de los ratings de la tabla'`. En la lista de ausentes: `'tiene como rating su calidad previa.'` (con el punto: la frase vieja entera) y `'<var>R</var> = <var>Q</var>.'`. En el test de /casas: `'no cuenta por encima de la mediana de los ratings'`.

- [ ] **Step 2: Correr y comprobar que fallan**

Run: `/Users/luiss/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_pages.py -q -k "metodo_rating_text or casas"`
Expected: FAIL por las frases nuevas.

- [ ] **Step 3: Escribir los textos**

`metodo_rating.html` ll. 340-341, sustituir «Una encuestadora sin sondeos evaluables tiene como rating su calidad previa. Es la regla de un modelo jerárquico sencillo:» por: «Una encuestadora sin sondeos evaluables tiene como rating su calidad previa o, si la suya es mayor, la mediana de los ratings de la tabla: la calidad previa puede quitar peso desde el primer sondeo, pero sólo lo da cuando hay elecciones que la respalden (la fórmula, en la nota de la mezcla). La mezcla es la regla de un modelo jerárquico sencillo:».

Nota `rating` l. 969, sustituir «Una encuestadora sin sondeos evaluables tiene <var>W</var> = 0 y <var>R</var> = <var>Q</var>.» por un párrafo con la fórmula en MathML (`<math display="block" aria-label="Q efectiva igual a el mínimo de Q y m, más el máximo de Q menos m y cero, por W entre W más k">`) de `Q_ef = min(Q, m) + max(Q − m, 0)·W/(W + k)`, y el texto: «Antes de la mezcla, <var>Q</var> se sustituye por <var>Q</var><sub>ef</sub>, con <var>m</var> la mediana de los ratings de la tabla calculados con <var>Q</var> (las encuestadoras excluidas no cuentan): el tramo de la calidad previa por encima de la mediana sólo se acredita en la proporción de la evidencia. Una encuestadora sin sondeos evaluables tiene <var>W</var> = 0 y <var>R</var> = min(<var>Q</var>, <var>m</var>).» La frase siguiente sobre los agregadores y el peso de reserva se queda como está.

`casas.html` l. 58: «…con su confianza (la calidad previa del método), que pesa más cuantas menos elecciones ha sondeado y que, sin ninguna, no cuenta por encima de la mediana de los ratings.»

`casas.py` l. 179: «…pesa en el rating mientras ha sondeado pocas elecciones; sin ninguna, no cuenta por encima de la mediana de los ratings.»

- [ ] **Step 4: Correr las páginas**

Run: `/Users/luiss/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_pages.py -q`
Expected: PASS (`test_metodo_rating_text` cuenta `<math display="block"` ≥ 12: uno más no estorba; `test_metodo_rating_explains_evidence_by_election` sigue leyendo σ, τ y k de la firma).

- [ ] **Step 5: Commit**

```bash
git add web/templates/metodo_rating.html web/templates/casas.html mtpy/lib/pages/casas.py tests/test_pages.py
git commit -m "Textos del tope de la calidad previa: /metodo/rating (cuerpo y nota de la mezcla), intro del ranking de /casas y tooltip de Confianza

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: documentación del método

**Files:**
- Create: `docs/analisis-2026-09/metodo-17-tope-calidad-previa.md`
- Modify: `docs/analisis-2026-09/computer-ratings.md:151` (punto 8, fórmula desfasada con `min_polls = 3`)

- [ ] **Step 1: Escribir `metodo-17-tope-calidad-previa.md`** con el formato de los métodos 15 y 16: título «Método 17: tope de la calidad previa para las casas sin elecciones (07-11-2026)»; «Cambios» (parámetro, segunda pasada, textos, con los ficheros); «Tests»; «Problema» (Opina 360 2,05 ≈ Sigma Dos; rúbrica sin poder predictivo: pendiente 0,14, correlación 0,12 con `W ≥ 3`); «Método» (fórmula, listón por cálculo, por qué la mediana y no la media); «Alternativas» (la tabla de la spec); «Resultados» con la tabla de la spec (casa nueva 0,81; SW Demoscopia 1,05 → 1,12; CEMOP 2,14 → 2,06; GAD3 2,26 → 2,46; Sigma Dos 2,12 → 2,30; Opina 360 con una elección 1,40, con tres 1,68) y un hueco «Backtest: pendiente (Tarea 4)». «Fuera de alcance»: el peso de reserva del `Forecaster`.

- [ ] **Step 2: Corregir `computer-ratings.md` l. 151**: `rating_p = (W · rating_adj_p + k · Q_ef) / (W + k)`, `k = (σ/τ)² = 1,96` elecciones ponderadas (no `min_polls = 3` sondeos), y una línea con `Q_ef` y el enlace al método 17.

- [ ] **Step 3: Commit**

```bash
git add docs/analisis-2026-09/metodo-17-tope-calidad-previa.md docs/analisis-2026-09/computer-ratings.md
git commit -m "Método 17: documentación del tope de la calidad previa; computer-ratings.md con la fórmula vigente (k = (σ/τ)², Q_ef)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: comprobación con dos backtests emparejados de `es`

**Files:**
- Lee: `backtest/run_backtest.py` (`--recompute-ratings`, `--out`), `backtest/compare_results.py`
- Modify: `docs/analisis-2026-09/metodo-17-tope-calidad-previa.md` (sección «Resultados», hueco del backtest)

Sin código nuevo: los dos backtests recalculan los ratings en memoria con `--recompute-ratings` (`rating_overrides`, `backtest.py:1110`), uno desde un worktree en el commit anterior a la Tarea 1 (sin tope) y otro desde el árbol de trabajo (con tope). Misma base, misma semilla, mismo `n_sim`. Nada se guarda en la base ni en `backtest/results`.

- [ ] **Step 1: Comprobar que no hay trabajos de Luis en marcha**

Run: `ps aux | grep -E "job.py|run_backtest" | grep -v grep`
Expected: sin líneas (si las hay, esperar: los dos backtests leen la base durante casi una hora).

- [ ] **Step 2: Worktree del código sin tope**

Run: `git worktree add /private/tmp/claude-501/-Users-luiss-HD-Proyectos-Code-elections-model/02dae2bb-4787-481d-b487-df8433e9cb7f/scratchpad/wt-sin-tope <hash del commit anterior a la Tarea 1>` y copiar `.env` al worktree (`cp .env <worktree>/.env`).

- [ ] **Step 3: Lanzar los dos backtests en segundo plano, cada uno a su carpeta del scratchpad**

Desde el worktree: `/Users/luiss/.pyenv/versions/3.11.9/bin/python backtest/run_backtest.py --scope es --recompute-ratings --out <scratchpad>/bt-sin-tope > <scratchpad>/bt-sin-tope.log 2>&1`
Desde el árbol de trabajo: la misma orden con `--out <scratchpad>/bt-con-tope`.
Expected: ~50 minutos cada uno; los dos `run.json` con `recompute_ratings: true`, misma semilla (42) y `n_sim` (1000).

- [ ] **Step 4: Comparar**

Run: `/Users/luiss/.pyenv/versions/3.11.9/bin/python backtest/compare_results.py <scratchpad>/bt-sin-tope <scratchpad>/bt-con-tope --scopes es --label-a "sin tope" --label-b "con tope"`
Expected: tabla del criterio 17 (MAE de voto, CRPS de escaños, Brier de mayorías, cobertura) con diferencias dentro del ±2 %, mejor/peor repartidos.

- [ ] **Step 5: Escribir los resultados en el método 17 y commitear; borrar el worktree**

```bash
git add docs/analisis-2026-09/metodo-17-tope-calidad-previa.md
git commit -m "Método 17: resultado del backtest emparejado de es con y sin tope (ratings recalculados en memoria)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git worktree remove <scratchpad>/wt-sin-tope
```

Si el backtest saliera peor fuera del ±2 % en varias métricas, no cambiar el default: parar y contárselo a Luis con la tabla.

---

## Después del plan (Luis)

Ratings de todos los ámbitos (`python job.py compute '{"what":["ratings"],"scopes":"all"}'`), imagen y update: en vivo, Ateneo del Dato pasa de 1,40 a 0,81 y Opina 360 de 1,63 a 0,81.
