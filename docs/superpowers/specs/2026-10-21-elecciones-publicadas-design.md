# Elecciones publicadas sin entrar en el modelo

Fecha: 2026-10-21. Estado: aprobada por Luis el 2026-10-21 («Aprobado»). La descripción se acordó en el chat del 21 de
octubre («Haz la 3», «En todos los sitios en los que se muestren los errores de las casas por elección», «Me vale,
quiero al menos ver los resultados de todas»). Esta spec describe solo lo que cambia respecto de `dev` en `dcd24d1`.
Revisión del constructor hecha el mismo día contra `ad9eff4`: no quedaron preguntas abiertas, y sus dos
observaciones están incorporadas en D2 y D5.

## Intención

Luis vio que en la web faltan las autonómicas del 26 de mayo de 2019 de Aragón, Asturias, Baleares, Canarias,
Cantabria, Castilla-La Mancha, Extremadura y La Rioja. También falta Castilla y León 2019, aunque él no la mencionó.

No es un fallo de carga. Las nueve elecciones tienen su resultado oficial en `events_data`, todos los sondeos de
Wikipedia que traen cifras están en `polls`, y sus errores están calculados y guardados (`polls.error_avg`, `bias`).
Faltan porque no son `featured`. `update_featured` (`mtpy/lib/data.py`) exige resultado oficial y al menos
`min_polls = 10` sondeos útiles: calculados, con `weight_over > 0`, que no sean agregadores ni paneles online
(`DROP_MTYPES`) y que no estén hechos en veda ni sean encuestas a pie de urna. Estas elecciones se quedan en 5-9,
porque los 16 o 17 sondeos de ElectoPanel que tiene cada una no cuentan.

`featured` decide dos cosas a la vez:

- **Qué alimenta el modelo**: efectos de casa (M6), deriva (M5), rating y su evidencia por elección, «Camino al
  rating», rebaño (M12), datos de calibración del error y backtest. Todo pasa por `Computer.filter_polls(featured=True)`
  o por `get_event_dates(featured=True)`.
- **Qué se publica**: `publish events --event-date all` solo publica las elecciones `featured`, y el `poll-errors` de
  `publish analysis` solo lleva sondeos de elecciones `featured`. De esas dos publicaciones salen /elecciones y todas
  las vistas de errores por elección.

Luis eligió separar las dos cosas: estas elecciones se muestran en la web, pero el modelo no cambia. Con 5-9 sondeos,
muchos de ellos de uno o dos años antes de la votación, aportan poco a la estimación. Aun así, el histórico de la web
tiene que estar completo.

## Qué elecciones se publican

**Publicada** = elección de un ámbito con fecha anterior a hoy y resultado oficial (`events_data`, `region_id = 0`,
`votes IS NOT NULL`). Las `featured` son un subconjunto de las publicadas.

Hoy son 94 elecciones: las 79 `featured` y 15 más:

| Ámbito | Elecciones nuevas | Sondeos útiles |
|---|---|---|
| `es` | 1977, 1979 | 9, 7 |
| `es-as` | 2012, 2015, 2019 | 7, 7, 7 |
| `es-ar` | 2015, 2019 | 9, 8 |
| `es-cl` | 2015, 2019 | 7, 6 |
| `es-ib`, `es-ex`, `es-cm`, `es-cn`, `es-ri`, `es-cb` | 2019 | 9, 8, 7, 6, 6, 5 |

Luis aceptó expresamente las seis más antiguas (1977, 1979, Asturias 2012 y Aragón, Asturias y Castilla y León
2015): «quiero al menos ver los resultados de todas».

Hoy todas las elecciones con resultado oficial tienen al menos un sondeo útil (comprobado en la base el 21-10-2026),
así que todas tendrán página completa.

## Qué no cambia (anti-objetivos)

Si cambia cualquiera de estas cosas, el trabajo está mal aunque la web enseñe las elecciones nuevas:

- El rating, el ranking de /casas, «Camino al rating», la evidencia del rating por elección y la calidad previa.
- Los efectos de casa (M6, tabla `pollsters_parties`, parte `house-effects` del análisis), la deriva (M5), el rebaño
  (M12) y los datos de calibración del error (parte `error-data`).
- El promedio, el pronóstico y el backtest (`backtest/results/` no se toca; el backtest sigue evaluando solo
  `featured`).
- La regla de `featured` (`min_polls = 10` y los mismos filtros).

La prueba de que no cambian: con los mismos datos, las salidas del modelo para un ámbito con elecciones nuevas (por
ejemplo `es-ar`) son idénticas antes y después del cambio.

## Dónde aparecen

Todas las vistas que muestran el error de las casas por elección (decisión de Luis):

1. **/elecciones**: la elección en la lista del ámbito y su ficha (resultados y hemiciclo, sondeos del ciclo, último
   sondeo de cada casa, reparto por bloques y diana). Sin bloque «Lo que dijo el modelo» ni curva del promedio,
   porque el backtest no las evalúa. La ficha ya se adapta cuando faltan `model` o `series`: es el caso
   `has_model = False` / `has_series = False` de `publish_event`.
2. **/casas, Histórico de errores**: filas, cifras, complementos y tendencia por elección, con sus filtros
   (ámbito, elección, encuestadora y años).
3. **Perfil de encuestadora**: la diana y «Último sondeo por elección». La tarjeta «Efectos de casa por elección»
   se queda solo con las elecciones analizadas (decisión de Luis, 21-10-2026). Sale de los efectos de casa del modelo
   (`pollsters_parties`), no de `poll-errors`.
4. **/metodo/promedio, ejemplo del CIS**: sale del mismo `poll-errors` (`HOUSE_EXAMPLE`, «el último sondeo del CIS
   antes de cada elección analizada desde septiembre de 2018»). El CIS publicó preelectoral de las nueve autonómicas
   del 26-M a unos 33 días de la votación, dentro de `drange = (6, 42)`. Por eso sus cifras vivas (hoy «21 de 23»)
   y la probabilidad binomial van a cambiar. Las cifras son vivas, pero hay que revisar que ningún texto de /metodo
   cite números fijos de ese ejemplo. Ver también la spec `2026-10-21-metodo-promedio-538-design.md`.

## Cómo (lo que ya está decidido)

### D1. La marca «publicada» se deriva, no se guarda

`get_events(scope)` (`mtpy/lib/data.py`) añade una columna booleana `has_results`: si la elección tiene resultado
oficial en `events_data` (`region_id = 0`, `votes IS NOT NULL`). No hay columna nueva en la base, así que Luis no
tiene que hacer ningún `ALTER` y la marca no puede quedar desfasada de los resultados. (En el chat propuse una
columna nueva en `events`; lo cambio por esto, que es más simple.)

### D2. El `Computer` sabe filtrar por «elecciones con resultado»

`Computer.filter_polls` y `get_polls_metric` aceptan una segunda selección de elecciones además de la actual: los
sondeos de las elecciones del `Computer` con resultado oficial. En esta selección se sigue exigiendo que el sondeo esté
calculado (`polls.computed`), y se aplican igual los demás filtros (`drop_mtypes`, `drop_ctypes`, `weight_over > 0`,
`drange`, `n_last`). El valor por omisión sigue siendo la selección `featured`, de modo que ningún estimador cambia.
Ojo: en `load_events`, `computed` de una elección es su `featured`, así que la nueva selección no puede apoyarse en
`df['computed']` de las filas de elecciones.

La forma de la API (un valor nuevo de `featured`, un parámetro aparte, etc.) se deja al constructor, con una
condición: `featured=False` tiene que seguir significando «sin filtro de elecciones», porque `party_first_polls`
(`computer.py:2424`) depende de eso.

### D3. `publish analysis`

La parte `poll-errors` usa la selección de D2 (todas las elecciones con resultado) con los mismos
`POLL_ERRORS_PARAMS`. `house-effects`, `herding` y `error-data` no cambian.

### D4. `publish events`

- `--event-date all` publica las elecciones con `has_results` anteriores a hoy, no solo las `featured`.
- `publish_event` acepta cualquier elección con `has_results`. El mensaje de `NothingToPublish` pasa a decir que la
  elección no tiene resultados, no que «no es featured».
- La parte `cycle` y los errores de la ficha usan la selección de D2. Hoy `cycle = comp.filter_polls(featured=True)`,
  que en un `Computer` de una sola elección no `featured` sale vacía.
- `model` y `series` se publican si el backtest los tiene, como ahora. Para las elecciones nuevas no los tiene.

### D5. Un aviso en la ficha de una elección no analizada

En la ficha de una elección publicada que no es `featured`, una línea en el sitio donde iría «Lo que dijo el
modelo», con el mismo tono que el resto de avisos: dice que tiene menos de 10 sondeos útiles y por eso no cuenta
para el rating de las encuestadoras ni para el backtest. No hay que decir que «no entra en el modelo», porque no es
del todo cierto: la edad de los partidos (`party_first_polls`) y las medianas de muestra de `compute_weights` ya
leen sondeos de todas las elecciones, y eso era así antes de este cambio. Para saberlo, la ficha necesita la marca `featured` de la elección: en la parte
`meta` del paquete de la elección o leída de `query_events`; lo decide el constructor. (Decisión mía: Luis pide que
todo esté a la vista y, sin el aviso, la falta del bloque del modelo parecería un fallo.)

### D6. La API

`/api/v1/events/{scope}` ya devuelve `featured` y `has_page`. No cambia: con D4, `has_page` pasa a ser cierto también
en las elecciones nuevas.

## Lo que se deja al constructor

- La forma de la API de D2 y los nombres (en inglés neutro, como el resto del código).
- Dónde viaja la marca `featured` hasta la ficha (D5) y el texto exacto del aviso, revisado con las reglas de
  redacción de Luis (nosotros, sin rótulos fijos).
- Revisar textos públicos que digan «elecciones analizadas» refiriéndose a lo que ahora son «publicadas»: por ejemplo
  el aviso «Esta encuestadora no tiene sondeos en las elecciones analizadas» de `web/templates/casa.html`, que se
  muestra donde se pintan errores. Hay que cambiar la palabra solo donde ya no sea verdad.
- Los tests: unitarios para D1-D5 con las inyecciones que ya usan `publish_event` y `publish_analysis`, y uno de
  regresión de los anti-objetivos (las salidas de los estimadores no cambian con la selección nueva).

## Fuera de alcance

- Página solo con resultados de una elección sin ningún sondeo útil: hoy no existe ningún caso. Si aparece,
  `publish_event` seguirá saltando con `NothingToPublish` y se verá en el log. (Decisión mía.)
- Cambiar `min_polls` o meter ElectoPanel u otros paneles online en el recuento (ver la decisión del 21-10-2026 sobre
  agregadores y paneles).

## Lo que tendrá que hacer Luis después

Por este orden, sin `ALTER` ni recálculo del modelo:

1. `publish analysis` de todos los ámbitos, para que `poll-errors` lleve las elecciones nuevas.
2. `publish events --event-date all` de todos los ámbitos.
3. Recorrer /elecciones (por ejemplo, Aragón 2019), el Histórico de /casas, el perfil del CIS y /metodo/promedio.
