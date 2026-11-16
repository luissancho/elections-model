# Web: descripciones de las páginas

Cambio pequeño, cuarto paso de la ronda de SEO y redes sociales. Luis dio paso a la tarea el 2026-11-16 («Sigue con
las descripciones»); los textos y el diseño no los ha revisado: todo lo que no viene de él está en «Decisiones
mías, por revisar». Plan: `docs/superpowers/plans/2026-11-16-web-descripciones.md`.

## Intención

Que la descripción de cada página (`<meta name="description">` y `og:description`: el texto bajo el enlace en
Google y en la vista previa de WhatsApp, Telegram o Facebook) diga con exactitud qué hay en ella, se lea entera y
no prometa lo que la página no tiene.

## Hechos de partida (2026-11-16, leídos en producción)

- Google enseña unos 155-160 caracteres en un ordenador y unos 120 en un móvil, y corta el resto; no hay tope
  para la etiqueta.
- Tres páginas del método pasan de 300 (303, 380 y 413): son la lista entera de sus secciones.
- Cuatro estudios usan como descripción su subtítulo publicado, que llega a 297, 255, 340 y 447 caracteres.
- Dos empiezan por la etiqueta del menú, que fuera de la web no dice nada: «Histórico Errores: …», «Ranking
  Encuestadoras: …».
- La de un ámbito acaba en «…, de Elecciones generales.» (en uno autonómico, «de Elecciones autonómicas ·
  Madrid.»); la de un sondeo dice «para Elecciones generales 2026».
- La de una encuestadora sin rating (agregador, panel en línea, llamadas automáticas) promete «su rating».
- La de una elección pasada no la nombra («la elección del 23 jul 2023») y dice «y el modelo» aunque la página no
  tenga la tarjeta del modelo.

## Enfoques

1. **Reescribir las que fallan para que quepan en 160 y cortar en `base.html` lo que aun así no quepa**
   (elegido): los textos fijos se leen enteros, y los que dependen de datos (el subtítulo de un estudio, un
   sondeo con un nombre largo) acaban en una frase completa o en una palabra entera con «…», en vez de donde caiga
   el corte de cada plataforma.
2. **Sólo reescribir**, y dejar los subtítulos de los estudios como están. Descartado: son los más largos y los
   que más se comparten.
3. **Una descripción corta propia en cada estudio** (un campo nuevo publicado). Descartado: el subtítulo ya es la
   mejor frase del estudio, y su primera oración lleva el hallazgo.

## Diseño

### El corte (`snippet`)

Filtro de las plantillas, aplicado en `base.html` a la descripción de todas las páginas (160 caracteres):

- Si el texto cabe, queda igual.
- Si no, se queda con las primeras frases completas que quepan, siempre que sumen al menos la mitad del tope.
- Si ni así, corta en la última palabra entera que quepa, quita las palabras de tres letras o menos que queden al
  final («y», «de», «los») y añade «…».

### Textos

| Página | Hoy | Nueva |
|---|---|---|
| `/es` | Encuestas, promedio de sondeos y escaños previstos por partido, bloque y circunscripción, con la probabilidad de mayoría absoluta, de Elecciones generales. | Elecciones generales 2026: promedio de las encuestas, escaños previstos por partido y circunscripción y probabilidad de mayoría absoluta. |
| un sondeo | Sondeo de Sigma Dos con trabajo de campo hasta el 5 oct 2026 para Elecciones generales 2026: voto y escaños por partido, su cambio frente a la entrega anterior, rating y peso en el promedio. | Elecciones generales 2026: sondeo de Sigma Dos realizado hasta el 5 oct 2026. Voto y escaños por partido, cambio frente al anterior y peso en el promedio. |
| `/elecciones` | Histórico Errores: cuánto se equivocaron las encuestas frente al resultado en cada elección pasada. | Cuánto se equivocaron las encuestas frente al resultado oficial en cada elección general y autonómica: su error medio y hacia qué bloque fallaron. |
| una elección pasada | Resultado de la elección del 23 jul 2023 y cuánto se equivocaron frente a él los sondeos, cada encuestadora y el modelo. | Elecciones Generales Julio 2023: el resultado oficial y cuánto se equivocaron frente a él los sondeos, cada encuestadora y el modelo. |
| `/casas` | Ranking Encuestadoras: qué encuestadoras aciertan más, cuánto se separa cada una del promedio y cuáles se ciñen al resto. | Qué encuestadoras aciertan más: el ranking por rating, cuánto se separa cada una del promedio de todas (su efecto de casa) y cuáles se ciñen al resto. |
| una encuestadora | Perfil de la encuestadora GAD3: su rating, sus errores en cada elección y su rebaño. | Perfil de la encuestadora GAD3: su rating, lo que se sabe de cómo trabaja, sus errores en cada elección y sus efectos de casa. |
| una casa sin rating | (la misma) | Perfil de Agregador: de qué tipo es, lo que se sabe de cómo trabaja y sus errores en cada elección. Sus sondeos no entran en el promedio y no tiene rating. |
| `/metodo` | El método del modelo: sus dos partes, el promedio de encuestas y la simulación de escaños, cómo se califica a las encuestadoras, cuánto acierta y dónde están sus límites. | El método del modelo: el promedio de encuestas, la simulación de escaños, el rating de las encuestadoras, cuánto acierta y dónde están sus límites. |
| `/metodo/promedio` | (303 caracteres) | Cómo calculamos el promedio de encuestas: qué sondeos entran, cuánto pesa cada uno, cómo ajustamos las diferencias entre encuestadoras y cómo lo comprobamos. |
| `/metodo/escanos` | (380 caracteres) | Cómo pasamos del promedio de encuestas a los escaños: el sorteo del voto, el reparto por circunscripciones con la regla D'Hondt y qué significa cada cifra. |
| `/metodo/rating` | (413 caracteres) | Cómo calculamos el rating de cada encuestadora: qué error medimos, con qué sondeos lo comparamos, su calidad previa y cuánto pesa en el promedio. |

Detalles:

- El evento y el sondeo nombran la elección como el título («Elecciones autonómicas Madrid 2027: …»), sin el punto
  medio de la barra del evento. El sondeo dice «realizado hasta el»: su fecha es el último día del trabajo de
  campo, nunca la de publicación. Con «con trabajo de campo hasta el» y «rating», el sondeo más corriente (con
  escaños y entrega anterior) pasaba de 160 y perdía su segunda frase; con un medio de nombre largo se queda en la
  primera, que dice de quién es y de cuándo.
- La elección pasada lleva su nombre (sin nombre, «Elección del 23 jul 2023») y sólo dice «y el modelo» si la
  página tiene la tarjeta «Lo que dijo el modelo».
- Los estudios conservan su subtítulo publicado; el corte lo deja en su primera oración o en unas 25 palabras.
  Con los subtítulos de hoy: «Desde el nombramiento de José Félix Tezanos en 2018, el CIS ha dado al bloque de
  izquierdas más voto del que obtuvo en 38 de 41 elecciones generales…».
- Sin cambios: la portada, `/metodo/backtest`, `/analisis`, el estudio del voto CERA, `/privacidad` y la
  descripción por defecto (la de la página de error).

### Código

- `mtpy/lib/pages/filters.py`: `snippet(text, limit=160)`; registrado en `environment()` como filtro `snippet`.
- `web/templates/base.html`: la descripción se captura una vez y se corta para las dos etiquetas.
- Las plantillas de la tabla.

### Pruebas

`tests/test_pages_descriptions.py`:

- `snippet`: lo que cabe queda igual; frases completas; palabra entera con «…» y sin palabra corta al final; un
  texto ya escapado se corta sin partir una entidad.
- Cada página del sitemap, y `/privacidad`: descripción no vacía, de 160 caracteres como mucho, igual en
  `og:description`, sin la etiqueta del menú delante y distinta de las demás.
- El texto exacto de cada tipo de página de la tabla, y que los fijos acaban en punto (no cortados).
- Una elección sin tarjeta del modelo, un sondeo con un nombre largo y un ámbito autonómico de nombre largo.

## Fuera de este paso

- Descripciones con cifras vivas («el PP lidera con…»): cambian cada día y las redes guardan la vista previa.
- Datos estructurados, `twitter:site`, las canónicas con `?scope=`: siguen en la ronda.

## Decisiones mías, por revisar

- **Todos los textos de la tabla.**
- **El corte a 160** y su regla (frases completas, o palabra entera con «…»), también para los subtítulos de los
  estudios, que son texto de Luis: la página los sigue enseñando enteros.
- **«su rebaño» sale de la descripción del perfil** (fuera de la web no se entiende) y entra «sus efectos de casa».
- La portada no se toca.
