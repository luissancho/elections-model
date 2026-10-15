# Web, fase 9: el ejemplo vivo del promedio en el método

Fecha: 2026-10-15. Estado: cerrada el 2026-10-15 en `dev` (plan
`docs/superpowers/plans/2026-10-15-web-fase-9-ejemplo-promedio.md`, ocho commits desde `89b1f0d`; 856 tests).
Descripción aprobada por Luis en el chat el 15 de octubre ("Nada que cambiar") y documento aprobado el mismo día
("Adelante") tras la revisión del constructor (tres preguntas, resueltas como decisiones mías en la sección
"Decisiones tomadas tras la revisión"). Pendiente de Luis: publicar un run (`publish forecast`) para que el bloque
aparezca con datos reales. Sigue a la fase 8
(`docs/superpowers/specs/2026-10-15-web-fase-8-rediseno-visual-design.md`). Esta spec solo describe lo que
cambia respecto de lo que hay en `dev` (`ada3a17`).

## Intención

Luis quiere que lo que hoy solo se ve en los notebooks de `notebooks/lab/` sirva en la web como ejemplo visual
de lo que explica el método. Empezamos por uno: en `/metodo/promedio`, una vez explicado cómo se calcula el
promedio, un bloque nuevo que muestre, para el partido que lidera las encuestas a día de hoy (ahora el PP), la
salida de `Forecaster.print_weights` (la tabla de los sondeos que pesan en la estimación y sus pesos) y la de
`Forecaster.plot_weights` (los sondeos como puntos del tamaño de su peso, la recta local con su banda y la campana
del kernel), esta última interactiva: en cada punto, el nombre y la fecha del sondeo y su peso. Con eso se da por
probada la fase; la idea de una sección propia de "laboratorio" se descartó ("yo no le daría tanto").

## Decisiones de Luis (2026-10-15)

1. Un bloque dentro de `/metodo/promedio`, no una sección nueva ni una página nueva.
2. Primero la tabla de `print_weights`, después el gráfico de `plot_weights`.
3. El gráfico es interactivo: tooltip por sondeo con encuestadora, fecha y peso; se ven el kernel gaussiano, la
   banda de error y la recta local.
4. El partido es el líder del voto estimado a día de hoy, sin selector.
5. Es suficiente para esta fase: nada más.

## Objetivos

1. Un método único del Forecaster que calcule los pesos de los sondeos en una fecha y la recta local de ese día, del
   que tiren `print_weights`, `plot_weights` y la publicación (D1).
2. Una parte nueva del run, `weights`, escrita por `publish forecast` con las demás, con schema, gemelo CSV y
   contrato (D2).
3. El bloque "Un ejemplo" en `/metodo/promedio`, servido desde el último run nacional publicado, que desaparece
   sin ruido cuando ese run no trae la parte (D3), con la tabla (D4) y el gráfico (D5).
4. Los notebooks siguen funcionando igual (D6).

## No objetivos

- Ninguna ruta ni página nueva. La API sirve la parte nueva por las rutas que ya existen
  (`/api/v1/forecast/{scope}/weights`, también en CSV) sin código nuevo en los controladores.
- Ningún selector de partido ni de fecha en el bloque. La parte lleva todos los partidos del run (cuesta lo mismo y
  deja la puerta abierta), pero la página solo pinta al líder.
- No se dibuja el promedio publicado (la curva de `series`) en el gráfico del ejemplo: el gráfico es el de
  `plot_weights`, con la recta de ese día. Podrá añadirse después.
- Ningún cambio en el modelo: ni en los pesos, ni en el kernel, ni en la regresión. Solo se expone lo que ya
  calcula.
- Nada de nginx, Docker ni CSP. Los otros dos notebooks (`PollstersOptimize`, `EventsSeats`) no se tocan.

## Anti-objetivos (lo que haría fracasar la fase aunque funcionara)

- Que las cifras del bloque no coincidan con las de la página de Promedios: el valor, la fecha y la banda del
  ejemplo deben ser los que el sitio ya publica para ese partido (misma fecha de la estimación, misma banda con
  error agrupado por encuestadora).
- Que la página del método se rompa o se quede en blanco con un run antiguo sin la parte `weights`: el bloque se
  omite y el resto de la página sale igual que hoy.
- Perder la lectura sin JavaScript: la tabla y los textos salen del servidor; solo el gráfico lo dibuja el
  navegador, como en el resto del sitio.
- Un bloque pesado: la isla `#initial-data` de la página lleva solo la rebanada del partido líder.

## D1. El método del Forecaster: `weights_at`

Hoy `print_weights` (`mtpy/lib/forecaster.py:1488`) y `plot_weights` (`:1593`) repiten el mismo cálculo con tres
diferencias que la web no puede permitirse: construyen el estimador sin los grupos por encuestadora (la banda que
pintan no es la agrupada de `fit`, `:526`), cortan el índice de predicción en el último sondeo (con `date=None` leen
el último sondeo, no hoy, y una fecha posterior desborda el índice) y uno pasa `df.tfs` como serie y el otro
`df[name]`.

Se extrae un método público:

```
Forecaster.weights_at(name: str, date: Optional[str | pd.Timestamp] = None) -> dict
```

- `name`: partido o bloque. `date`: el día en que se lee el promedio; por defecto `self.date_fit_last` si el
  promedio está ajustado (`fit_forecast`), y si no, el mínimo entre hoy y el último sondeo, como ahora.
- Sondeos: los de `fc_series` con valor para `name` y `weight > 0`, como `fit`. Estimador `LocalKernelEstimator`
  sobre `df[name]` (valores netos de efecto de casa, los que ajusta el promedio), con `weights`, con `groups` por
  `pollster_id` cuando `reg_params['cov_type'] == 'cluster'` y con `**self.reg_params`, exactamente como `fit`.
- Rejilla de predicción: la misma que `fit`, porque la anchura adaptativa depende de toda la rejilla
  (`Kernel.get_bw_adaptive` normaliza por la densidad sobre todos los puntos) y solo con la misma rejilla la recta
  coincide con la serie publicada. `fit_forecast` guarda su `max_fc` en `self.max_fc` (0 hasta que se ajusta) y
  `weights_at` construye la rejilla diaria desde el primer sondeo útil del partido hasta su último sondeo útil
  + `self.max_fc`, como `fit`. Si `date` cae después del final de la rejilla (un partido ausente de los últimos
  sondeos, cuyo voto publicado es un arrastre), se lee en ese final y `stats['date']` lo registra; si cae antes
  del primer sondeo, el método devuelve `None`.
- Pesos: `kws = est.build_pred(ix).get_kernel().get_weights(est.pred)`; en la posición de `date`, el peso por
  kernel de cada sondeo (`weight_kernel`, gaussiana sin normalizar, 1 a distancia cero) y el peso combinado
  `weight = weight_kernel · weight_i`. Se conservan exactamente los sondeos que entran en la regresión de ese día:
  la máscara de `get_local_estimator` (`stat.py:986`), que aplica `1e-2` a los pesos ya reescalados por
  `Estimator.build_input` para que sumen `neff`, no a los brutos. Así `len(polls) == stats['nobs']` y la tabla
  es, literalmente, los sondeos de la recta. El constructor expone esa máscara desde el estimador en lugar de
  reproducir el reescalado.
- Recta local: `est.get_local_estimator(kws, pos)` ajustado con `alpha=self.alpha` sobre los días de la ventana.
  La ventana va del primer sondeo con `weight_kernel >= 1e-2` hasta el máximo entre el último de ellos y `date`:
  en un run publicado `date` cae días después del último sondeo, y la recta, la campana y el eje tienen que llegar
  hasta el día que se lee. Ese estimador es una sola recta: su valor en `date` es el promedio del día y sus
  `cmin`/`cmax` son la banda publicada.
- Campana del kernel: la gaussiana centrada en `date` con la anchura que el kernel adaptativo da a ese punto de la
  rejilla completa (la posición de `date` en las anchuras de `get_bw_adaptive(p=rejilla)`, las que usan los
  pesos; con un solo punto el método devuelve la anchura fija), evaluada cada día de la ventana, con máximo 1.

Devuelve un `dict` con cuatro miembros:

| Miembro | Tipo | Contenido |
|---|---|---|
| `polls` | `DataFrame` indexado por fecha | una fila por sondeo conservado: `pollster_id`, `pollster`, `sponsor`, `sample_size`, `pct` (valor publicado, de `fc_series_raw`), `pct_adj` (valor neto del efecto de casa, de `fc_series`), `weight`, `weight_kernel`, `weight_over`, `weight_sample`, `weight_rating`, ordenadas por fecha |
| `line` | `DataFrame` indexado por día | la recta local en la ventana: `mean`, `cmin`, `cmax` |
| `kernel` | `Series` indexada por día | la campana en la ventana, en `[0, 1]` |
| `stats` | `dict` | `date` (la fecha leída, que solo difiere de la pedida si la rejilla del partido acaba antes), `mean`, `cmin`, `cmax`, `err`, `nobs`, `neff` (los del estimador local en `date`, como los `fc_stat` de `fit`) y `bandwidth` (la anchura del kernel ese día, en días) |

`print_weights` y `plot_weights` conservan sus firmas y su salida; pasan a llamar a `weights_at` y a formatear o
pintar lo que devuelve. Cambian en lo que la coherencia exige: la banda es la agrupada, la máscara es la del
estimador (alguna fila del borde, de peso en torno a 0,01, puede entrar o salir respecto de hoy) y la ventana llega
hasta `date` cuando esta es posterior al último sondeo (en el notebook no lo es, así que ahí no se nota).
`print_weights(show=False)` sigue devolviendo la tabla con las columnas de hoy (`date`, `pollster`, `weight`,
`tfs`, `tte`, `sample_size`, `weight_kernel`, `weight_over`, `weight_sample`, `weight_rating`) para no romper al
notebook; `tfs`/`tte` se derivan de la fecha como ahora.

Decisión mía: el factor de dispersión de la encuestadora (`f_c`, el rebaño) no aparece como columna propia porque
el modelo lo lleva dentro de `weight_sample` (`forecaster.py:1026`); los textos lo dicen así, y la fórmula de la
página ya lo presenta como la muestra efectiva.

## D2. La parte `weights` del run

En `mtpy/lib/bundle.py`:

- `RUN_PARTS` gana `'weights'` al final: `('meta', 'headline', 'series', 'polls', 'fan', 'house-effects',
  'dispersion', 'weights')`. Al ir al final, `CSV_PARTS = RUN_PARTS[2:]` (`webapi/checks.py:22`) y
  `check_part(part, RUN_PARTS)` del controlador `Forecast` la sirven sin tocarlos.
- `SCHEMAS['weights'] = {'date': str, 'parties': list, 'rows': list, 'line': dict, 'kernel': dict, 'stats': dict}`.

En `mtpy/lib/publish/forecast.py`, `export_weights(fc, names)` devuelve `(data, frame)` como las demás y entra en
la tupla `cycle` de `publish_forecast` tras `dispersion`:

- `date`: la fecha en que el run lee el voto estimado, `sim.as_of` en ISO; `export_weights` recibe esa fecha y
  llama a `weights_at(name, date=as_of)`. En un run normal coincide con `date_fit_last` (último sondeo +
  `max_fc`); en un run retrospectivo (`limit_date`) y en la última semana antes de la elección `as_of` es anterior
  (`Simulator`: `min(limit_date, fit_last)`), y es la fecha de la que salen `headline` y `vote`, así que el
  ejemplo lee el mismo día que ellos.
- `parties`: los partidos de `names` con ajuste (los que `weights_at` no puede ajustar, por tener menos de tres
  sondeos útiles o por no tener sondeos antes de `date`, se omiten sin error).
- `rows`: formato largo, una fila por partido y sondeo: `name`, `date` (ISO), `pollster_id`, `pollster`, `sponsor`,
  `sample_size`, `pct`, `pct_adj`, `weight`, `weight_kernel`, `weight_over`, `weight_sample`, `weight_rating`.
  Porcentajes a `PCT_DECIMALS`; pesos a `PROB_DECIMALS` (3), como el `weight` de `polls`.
- `line`: `{partido: {'dates': [...], 'mean': [...], 'lo': [...], 'hi': [...]}}`, columnar como `series`, solo
  los días de la ventana del partido.
- `kernel`: `{partido: {'dates': [...], 'values': [...]}}`, los mismos días, valores a 3 decimales.
- `stats`: `{partido: {'date', 'mean', 'lo', 'hi', 'err', 'nobs', 'neff', 'bandwidth'}}` (`date` es el día
  leído para ese partido, igual al `date` de la parte salvo para un partido cuya rejilla acaba antes; `lo`/`hi`
  son `cmin`/`cmax`, nombres del sitio; `bandwidth` en días con 1 decimal; `nobs` entero; `neff` con 1 decimal).

CSV `csv/weights.csv`: la tabla `rows` tal cual (una fila por partido y sondeo). La recta y la campana no tienen
gemelo CSV: se derivan de los sondeos y la web no las sirve en tabla.

`docs/web/contrato.md`: el árbol del run gana `weights.json` y `csv/weights.csv`; una sección `### weights`
bajo "Ficheros del run" con la fuente (`Forecaster.weights_at` en `date_fit_last`), los miembros y el CSV; la
tabla de rutas de la API (`/api/v1/forecast/{scope}/{part}`) y la lista de partes con gemelo CSV incluyen
`weights`. Los runs anteriores no la tienen, y el contrato lo dice, como hizo con `seats` en la enmienda de la fase
7c.

El doble de pruebas `synthetic_forecaster` (`tests/fakes.py`) gana un `weights_at(name, date=None)` que devuelve
las cuatro piezas con los seis sondeos sintéticos (pesos cualesquiera coherentes: `weight_kernel` en `[0, 1]`,
`weight` producto de los factores) y una recta y una campana de unos días; el export no contempla un Forecaster sin
el método.

## D3. El bloque en `/metodo/promedio`

### Contexto

`mtpy/lib/pages/metodo.py` gana `weights_example() -> Optional[dict]`, al estilo de `national_figures()`:

- Lee del último run de `NATIONAL_SCOPE` las partes `headline`, `meta` y `weights` (`part(...)`). Si `weights`
  da 404, o `headline` no tiene partidos, devuelve `None`. Cualquier otro `HttpError` se propaga, como en
  `national_figures`.
- El líder es el partido de mayor `pct` en `headline['nowcast']['parties']` (vienen en el orden de `names`, no
  de voto: `vote_forecast` itera los nombres). Si no está en `weights['parties']`, devuelve `None`.
- Devuelve `{'name', 'fullname', 'color'}` (del partido en `meta['parties']`), `'date'` (`weights['date']`),
  `'stats'` (los del líder), `'rows'` (las filas del líder, por fecha ascendente, con `weight_max` y los `vmax` de
  las barras calculados o fijos, ver D4), `'n_rows'`, `'line'` y `'kernel'` (los del líder) y `'run'` (el id del
  run, para el enlace al CSV).

`metodo_section_context` añade `context['example'] = weights_example()` solo cuando `section == 'promedio'`, y,
si no es `None`, `context['initial'] = {'state': ..., 'example': {'name', 'color', 'date', 'stats', 'rows'
(solo lo que el gráfico usa: `date`, `pollster`, `sponsor`, `sample_size`, `pct`, `pct_adj`, `weight`,
`weight_kernel`, `weight_over`, `weight_sample`, `weight_rating`), 'line', 'kernel'}}`. `base.html` ya pinta la
isla cuando `initial` está definido.

### Plantilla

En `web/templates/metodo_promedio.html`, dentro de `{% if example %}`:

- En el índice de secciones (`nav.section-nav.toc`), entre «La banda del promedio» y «De las encuestas al voto
  estimado», la entrada `<a href="#ejemplo">Un ejemplo: el {{ example.name }} hoy</a>`.
- En la misma posición entre las tarjetas, la tarjeta `m.card_start('ejemplo', 'Un ejemplo: el promedio del
  {{ example.name }} hoy', lead)`, con el lead: «Los sondeos que pesan en el promedio del {name} leído el {fecha}
  y la recta que ese día se ajusta a ellos: lo que las secciones anteriores explican, con los datos de hoy.»
- Un párrafo de entrada con las cifras: el promedio leído ese día (`stats.mean` con el filtro `pct`), su banda
  («de {lo} a {hi}»), cuántos sondeos entran en la recta (`stats.nobs`, que es `n_rows`: la tabla y la regresión
  son los mismos sondeos) y a cuántos equivalen por sus pesos (`stats.neff` con un decimal: «que por sus pesos
  equivalen a unos {neff} sondeos de igual peso»), y la anchura de la campana ese día («una campana de unos
  {bandwidth} días de anchura a cada lado»). Sin KPIs ni tarjetas: prosa, como el resto de la página.
- La tabla (D4), el gráfico (D5) y un «Cómo leerlo» (`m.how()`) con: qué es cada columna y cada elemento del
  gráfico; que la recta es solo la de ese día (cada día tiene la suya, y el promedio publicado une sus valores);
  que los puntos muestran la cifra publicada y la recta se ajusta a las cifras netas del efecto de casa, por lo que
  no pasa «por el medio» de los puntos; que la banda es la del promedio, con el error agrupado por encuestadora;
  y que el factor de muestra lleva dentro la dispersión. Enlaza a `#peso`, `#efectos-de-casa`, `#regresion` y
  `#banda`.
- Un enlace al CSV del run (`api_url(NATIONAL_SCOPE, 'weights', fmt='csv')`, como los `csv_links` del backtest),
  con el texto «Descargar los pesos de todos los partidos (CSV)».

El texto de la tarjeta «De las encuestas al voto estimado» no cambia. El bloque no lleva selector de ámbito ni de
modo: la página del método no los tiene.

## D4. La tabla

Dentro de `.table-wrap`, con `<caption>`, por fecha ascendente (el orden del notebook y del eje del gráfico).
Columnas:

| Columna | Origen | Formato |
|---|---|---|
| Fecha | `date` | `fmt_date_short` |
| Encuestadora | `pollster` (+ ` / sponsor` si lo hay, como en las tablas de sondeos) | texto |
| Muestra | `sample_size` | `int`, guion si falta |
| {Partido} | `pct` | `pct` (1 decimal con `%`) |
| Peso | `weight` | `num(2)` con barra en celda, escala 0 a `max(3, weight_max)` |
| Cercanía | `weight_kernel` | `num(2)` con barra, escala 0 a 1 |
| Sobrerrepresentación | `weight_over` | `num(2)` con barra, escala 0 a 1 |
| Muestra (factor) | `weight_sample` | `num(2)` con barra, escala 0 a 2 |
| Rating | `weight_rating` | `num(2)` con barra, escala 0 a 2 |

Las barras reutilizan el patrón `td.cell-bar > i.fill` del ranking (`casas.html:39`, `site.css:734`) con un
relleno neutro nuevo (`.fill.w`, tinta del sitio a baja opacidad; no los `pos`/`neg` verde y rojo) y el filtro
`bar(lo, hi)` para el ancho. Los `vmax` de los cuatro factores son los de `print_weights` (1, 1, 2, 2); el del
peso total es 3 salvo que algún peso lo supere. Cabeceras cortas con `title` explicativo; el «Cómo leerlo» da el
detalle.

Se muestran las primeras 20 filas y un botón «Cargar más» que las enseña de 20 en 20 (`wireMore` de
`pages/common.js`, como las tablas de sondeos). En un run nacional los sondeos con peso apreciable rondan el
centenar (`nobs` 100 en el notebook del 11 de septiembre). En móvil, la tabla se desplaza en horizontal con la
primera columna fija, como las demás de la fase 8.

## D5. El gráfico

Un módulo nuevo `web/dist/js/charts/weights.js` sobre `charts/base.js`, exportando `weightsOption(example, opts)`
(la opción ECharts, comprobable con el arnés de `tests/test_js_charts.py`) y `renderWeights(el, example)`.
`pages/metodo.js` lo dibuja cuando existe `#weights-chart` (hoy `paint()` sale sin isla; con ella, los gráficos
del backtest siguen sin dibujarse porque sus contenedores no están).

Contenido, el de `plot_weights`:

- Eje X temporal acotado a la ventana (`line.dates`), que por D1 llega hasta la fecha de la estimación: las guías
  y el extremo de la recta quedan dentro del eje, con los últimos días sin sondeos a la derecha. Eje Y izquierdo en
  puntos porcentuales con `%` (`fmtPct`), con margen alrededor de los puntos. Eje Y derecho de 0 a 1, sin
  etiquetas, para la campana.
- Los sondeos como `scatter` en `(fecha, pct)`: tamaño proporcional al peso combinado (de unos 6 px al peso mínimo
  a unos 20 px al máximo) y color del partido con opacidad creciente con el peso (los de poco peso, muy claros),
  como la rampa de azules del notebook. Tooltip del punto: encuestadora en negrita (+ medio), fecha y muestra
  (`pollTooltipLines` da esas dos líneas), «Publicado {pct} · corregido {pct_adj}», «Peso {weight}» y una línea con
  los factores «cercanía {k} · sobre. {o} · muestra {s} · rating {r}» (`fmtNum` a 2 decimales).
- La recta local: `mean` como línea del color del partido, con la banda `lo`–`hi` con `bandSeries` de
  `charts/series.js` (ya exportada), con la opacidad de las bandas del sitio. Tooltip de la recta:
  `meanTooltipLine` del día.
- La campana del kernel en el eje derecho, línea fina discontinua de un color neutro (el `THEME` del sitio), con
  tooltip «Peso por cercanía {valor}».
- Guías discontinuas grises (`markLine`) en la fecha de la estimación y en su valor, sin etiqueta sobre el
  gráfico: las cifras ya están en el párrafo de entrada.
- Leyenda con «Sondeos», «Recta del día», «Banda 95 %» y «Kernel»; la leyenda de ECharts, no las píldoras del
  sitio, porque aquí no selecciona partidos.
- Contenedor `div.chart#weights-chart` con `role="img"` y `aria-label` que diga qué muestra; altura la de `.chart`
  (340 px, 250 en móvil). Tipografías y colores del `THEME`, como el resto.

## D6. El notebook

`notebooks/lab/PollsOptimize.ipynb` no cambia: las celdas 7 y 8 llaman a `print_weights` y `plot_weights` con la
misma firma. Antes de `fc.fit` el promedio no está ajustado, así que `date=None` sigue leyendo el último sondeo,
como hasta ahora, y la ventana no se alarga. Lo único que puede verse distinto es alguna fila del borde de la
tabla (D1, máscara del estimador) y la banda, ahora la agrupada. El constructor comprueba que las dos celdas se
ejecutan tras el cambio (contra la base local).

## Comprobación

Tests nuevos o ampliados (hoy 759):

- `test_forecaster_unit.py`: `weights_at` sobre una serie sintética: `len(polls) == stats['nobs']`;
  `weight == weight_kernel · weight_i`; tras `fit_forecast(max_fc=10)`, en `date_fit_last` la recta vale lo mismo
  que `forecast` ese día y su banda coincide con `fc_stat` (misma rejilla, error agrupado), también para un
  partido ausente de los últimos sondeos (se lee al final de su rejilla y `stats['date']` lo dice); la ventana
  llega hasta `date`; una `date` anterior al primer sondeo devuelve `None`; `print_weights(show=False)` devuelve
  las columnas de siempre.
- `test_publish_unit.py`: `export_weights` cumple el schema, redondea, escribe el CSV largo y lleva en `date` el
  `as_of` del simulador (en el test de backfill, `limit_date`);
  `test_publish_forecast_writes_the_whole_layout_in_order` pasa con la parte nueva (sale de `RUN_PARTS`).
- `test_bundle_unit.py` y `test_web_api.py`: la parte se valida y se sirve en JSON y CSV por la ruta de siempre.
- `test_pages.py`: con el fixture del run, `/metodo/promedio` trae `id="ejemplo"`, el nombre del líder en el
  título y en el índice, la tabla con sus columnas, el botón «Cargar más», la isla con solo el líder y el enlace al
  CSV; sin la parte `weights` (fixture sin ella) la página sale como hoy, sin `id="ejemplo"` ni isla;
  `test_metodo_promedio_text` sigue pasando.
- `test_js_charts.py`: `weightsOption` produce un `scatter` con tamaños crecientes con el peso, una línea con su
  banda, una serie en `yAxisIndex: 1` para el kernel y dos `markLine`.
- `docs/web/contrato.md` actualizado y el fixture de la web (`tests/fixtures`) con un `weights.json` sintético.

Comprobación manual de Luis: publicar un run (`publish forecast`) y ver el bloque en `/metodo/promedio` con datos
reales, en escritorio y en móvil; comprobar que el promedio y la fecha del párrafo coinciden con los de `/es`.

## Decisiones tomadas tras la revisión del constructor (2026-10-15)

Un revisor leyó el documento como constructor y devolvió tres preguntas y dos avisos. Las cinco se resuelven aquí
como decisiones mías, ya incorporadas en D1, D2 y D3; ninguna cambia lo que Luis ve en la página, todas afectan a
que las cifras del ejemplo sean exactamente las publicadas.

1. La ventana de la recta, la campana y el eje llega hasta la fecha leída, no solo hasta el último sondeo: en un
   run publicado la fecha cae diez días después del último sondeo, y con la ventana del notebook las guías y el
   final de la recta quedaban fuera del gráfico.
2. La tabla son exactamente los sondeos de la regresión de ese día (la máscara del estimador, sobre los pesos
   reescalados), no los de peso bruto `>= 0,01`: así `n_rows == nobs` y el párrafo cuenta una sola cosa. En el
   notebook puede entrar o salir alguna fila del borde.
3. `weights_at` usa la misma rejilla que `fit` (hasta el último sondeo del partido + `max_fc`, que `fit_forecast`
   pasa a guardar), porque la anchura adaptativa depende de toda la rejilla; y la parte se lee en `sim.as_of`, la
   fecha de la que salen el titular y el voto, que en los runs retrospectivos y en la última semana es anterior a
   `date_fit_last`. Un partido cuya rejilla acaba antes se lee al final de ella y lo registra.
4. La anchura de la campana es la de la rejilla completa en la posición de la fecha; `get_bw_adaptive` con un solo
   punto devuelve la anchura fija.
5. El líder es el partido de mayor `pct` del titular, no el primero: el titular viene en el orden del catálogo.

## Lo que queda al constructor

- Nombres internos, reparto en funciones y el orden de los pasos del plan.
- Los tamaños exactos de los puntos, la rampa de opacidad y los colores neutros de la campana y de las barras.
- Cómo expone la máscara del estimador y cómo calcula la campana diaria (en Python con el `Kernel` del
  estimador).
- El fixture sintético de `weights.json` y los datos del doble `weights_at`.
- La redacción final del «Cómo leerlo» y de las cabeceras de la tabla, en el tono del resto de la página.
