# /metodo/promedio al estilo de 538: el texto completo y las notas desplegables

Fecha: 2026-10-21. Rama: `dev`. Página: `web/templates/metodo_promedio.html` («¿Cómo promediamos las encuestas?»).
Redactado sección a sección con Luis en conversación; este documento recoge el texto aprobado y lo que hace falta
construir para publicarlo.

## 1. Intención

Luis quiere que la página explique el promedio con mucho más detalle: paso a paso, qué significa y cómo se calcula
cada variable, por qué esos valores y qué medimos para elegirlos, con el contexto de cómo se hacen las encuestas en
España y nombrando los campos de la base (Polls y tablas relacionadas). El modelo de referencia son las páginas
de metodología de FiveThirtyEight («How Our Polling Averages Work», «How Our Pollster Ratings Work»): cada decisión
viene con lo que se midió, se reconocen los límites y hay casos reales.

Dos lectores a la vez:

- una persona formada que sabe cómo funcionan unas elecciones, que debe poder leer la página entera y entender cómo
  se pasa de cientos de sondeos a una línea;
- un experto (Narciso Michavila, Kiko Llaneras) que debe salir impresionado por el nivel de detalle y recomendarla
  («mirad, este sitio explica muy bien cómo funciona este mundo»).

El proyecto es además la carta de presentación de Luis como experto en análisis de datos y estadística aplicada.

## 2. Objetivos, no objetivos, antiobjetivos

Objetivos:
- El texto de la sección 5, tal cual (salvo erratas), en la plantilla, con sus fórmulas en MathML.
- Notas al pie desplegables (sección 6) con el componente de la sección 7.1.
- Los cambios de la tarjeta del ejemplo del PP (7.3) y la firma (7.4).

No objetivos:
- Las demás páginas de /metodo (escaños, rating, backtest): vendrán después con el mismo criterio. El componente
  de notas debe poder reutilizarse en ellas, pero no se usa todavía.
- Cambios en el modelo. Ya hechos en esta sesión: `poll_overweights` agrupa las rachas para que sumen 1
  (`2b3d13a`); la probabilidad binomial de la tarjeta del CIS (`2fe0171`, `3fab08f`).
- Separar el factor de dispersión del de muestra en la parte `weights` del run (aplazado por Luis).

Antiobjetivos (fallaría aunque funcione):
- Decir en público que una corrección no mejora el backtest (dispersión, rebaño, recentrado). Se justifican por
  coherencia. Sí se pueden dar mediciones favorables y limitaciones medidas (en nota).
- Esconder texto del cuerpo tras un clic: todo lo que no es nota se lee a simple vista. Las notas, además de
  desplegables, siguen visibles al final de la página.
- Las palabras «casa/s» y «patrocinador/es» en el texto visible (regla `assert_public_wording`): se dice
  «encuestadora», «efecto de encuestadora» y «medio». Las rutas e ids (`/casas`, `#efectos-de-casa`) no cambian.
- Rótulos fijos repetidos en cada sección («El problema», «La idea», «Los campos», «Un caso real»): cada bloque
  lleva un título propio que dice algo, o ninguno.

## 3. Reglas de redacción (de Luis)

- Voz: un «nosotros» de proyecto («medimos», «decidimos»), con la firma de Luis al pie.
- Cada paso va del problema a la idea, a la fórmula, a por qué esos valores y a un caso real, sin rótulos fijos.
- Cada fórmula en el cuerpo, con cada letra explicada en palabras y su campo de la base cuando lo tiene.
- El contexto de cómo funcionan las encuestas va dentro de cada paso (el problema y la justificación), no en una
  introducción aparte.
- Cuando un valor es una convención que no hemos medido, se dice, en nota.
- Notas al pie solo para lo que interesa a un experto: limitaciones medidas, contrastes de calibración,
  comprobaciones, convenciones sin medir, detalles de implementación y referencias.

## 4. Estructura de la página

Orden de las tarjetas (ids actuales conservados; los nuevos, en cursiva):

| # | id | Título |
|---|---|---|
| — | — | titular (`m.headline`) |
| — | — | índice (`section-nav toc`) |
| 1 | `resumen` | En resumen |
| 2 | `que-entra` | Qué sondeos entran |
| 3 | `peso` | Cuánto pesa cada sondeo (subsecciones con h3: *`muestra`*, *`dispersion`*, *`sobrerrepresentacion`*, *`rating`*, *`cuentas`*) |
| 4 | `efectos-de-casa` | Los efectos de encuestadora |
| 5 | `ejemplo-cis` | Un ejemplo: el CIS desde 2018 (ya reescrita en `2fe0171`; sin cambios salvo notas si las hubiera) |
| 6 | `regresion` | La regresión local (subsecciones h3: *`anchura`*, *`consecuencias`*) |
| 7 | `banda` | La banda del promedio |
| 8 | `ejemplo` | Un ejemplo: el promedio del PP hoy (el partido es el líder del run, vivo) |
| 9 | `voto-estimado` | De las encuestas al voto estimado |
| 10 | *`notas`* | Notas |
| — | — | firma |
| — | `closing` | cierre actual (`m.closing`), sin cambios |

El índice enlaza las tarjetas 2 a 10 (sin el resumen), con las dos tarjetas de ejemplo condicionadas como ahora.

Las cifras de las secciones son una foto fechada del cálculo publicado el 20 de octubre de 2026 (se dice en el texto
o en el pie de cada tabla: «datos del cálculo del 20 de octubre de 2026»). Las cifras vivas son las de las dos tarjetas
de ejemplo, `figures.n_sim` y las que ya eran vivas. Ver 8.1.

En la sección 5, `{…}` marca una variable de plantilla viva; `[nota N]` marca una llamada a nota (sección 6);
las fórmulas se escriben en texto y se pasan a MathML como las de la página actual (`<math display="block"
aria-label="…">`, con `aria-label` en palabras); `*x*` es una variable (`<var>`).

## 5. Texto final

### Titular

> El promedio es una línea que sigue a las encuestas día a día: cada sondeo pesa según lo que sabe, a cada
> encuestadora le restamos su inclinación y, para cada día, ajustamos una recta a los sondeos cercanos.

### 1. En resumen

Lead: «El promedio de encuestas en cuatro ideas.»

- Entran los sondeos del ciclo terminados al menos seis días antes de la votación, salvo agregadores, paneles en
  línea abiertos y sondeos de los partidos.
- Cada sondeo pesa según su muestra, cuánto oscilan las cifras de su encuestadora, si publica en racha y su rating.
- A cada encuestadora le restamos su inclinación en cada partido, sin mover el nivel del promedio.
- Para cada día, una recta ajustada a los sondeos cercanos; su banda mide el consenso de las encuestas, no el voto.

### 2. Qué sondeos entran

Lead: «Los sondeos publicados desde la elección anterior, con cuatro exclusiones, cada una con su motivo.»

Recogemos los sondeos de las tablas de intención de voto de Wikipedia, la fuente más completa y abierta que existe
para España (ver <a href="/metodo#datos">Los datos</a>). El promedio de una elección usa los sondeos publicados desde
la anterior; el resultado de aquella no entra: el promedio solo habla de lo que dicen las encuestas. En las generales
de este ciclo entran {n_polls} sondeos de {n_pollsters} encuestadoras. De cada uno guardamos su encuestadora
(`pollster`), el medio que lo encarga (`sponsor`), su trabajo de campo (`start_date` y `end_date`; este último también
es `date`, su lugar en el tiempo) y los días que lo separan de la elección (`days`).

No todo lo que se publica es un sondeo que debamos promediar. Quedan fuera:

- **Los de los últimos días.** Solo entran los sondeos que terminaron su trabajo de campo al menos seis días antes de
  la votación (`days` ≥ 6). La ley prohíbe publicar sondeos en los cinco días anteriores (artículo 69.7 de la LOREG);
  los que se hacen en ese tiempo, que se publican fuera de España o el mismo día, y las encuestas a pie de urna no se
  hicieron en las mismas condiciones que el resto y falsearían la comparación entre elecciones.
- **Los agregadores**, que publican su propia media de otras encuestas, como KeyData para Público. Promediar una
  media contaría dos veces los mismos sondeos.
- **Los paneles en línea abiertos**, como ElectoPanel, de Electomanía, en los que la muestra la forman quienes deciden
  participar: no es una muestra elegida por la encuestadora y su error no se comporta como el de un sondeo. El tipo
  de cada encuestadora se guarda en `mtype` (`aggr` para los agregadores, `online` para los paneles).
- **Los sondeos internos de los partidos.** Si quien firma es un partido, no se carga. Si un partido encarga un
  sondeo a una encuestadora («Sigma Dos / PP»), sí entra, porque el trabajo es de la encuestadora; se carga sin medio.

#### Cuando los partidos cambian

Los partidos se fusionan, se refundan o concurren en coalición, y las encuestas tardan en reflejarlo. Para cada
elección definimos qué partido hereda a cuáles: en 2023, SUMAR heredó a Unidas Podemos, Más País y Chunta
Aragonesista, entre otros. Los sondeos que aún listan a los predecesores cuentan para el sucesor en lugar de perderse en «otros»; si un
sondeo lista a la vez al predecesor y al sucesor, se suman, aunque la suma de las partes suele sobrestimar a la
coalición. Los partidos que no seguimos en esa elección van a «otros».

### 3. Cuánto pesa cada sondeo

Lead: «El peso de un sondeo es el producto de cuatro factores: su muestra, cuánto oscilan las cifras de su
encuestadora, si esta publica en racha y su rating.»

No todos los sondeos informan lo mismo. Uno de 3.000 entrevistas sabe más que uno de 600; una encuestadora que publica
cada dos días no tiene más razón que una que publica cada mes; y una que lleva años equivocándose más que las demás
merece menos confianza. El peso de cada sondeo recoge todo eso como el producto de cuatro factores:

  wᵢ = oᵢ · √(nᵢ / ñ) · f_c · r_c

Tabla de cuatro filas (Factor | Qué mide):
√(*nᵢ*/*ñ*) | la muestra · *f_c* | si las cifras de la encuestadora oscilan lo que deberían · *oᵢ* | que una
encuestadora no pese más por publicar más · *r_c* | cuánto se ha equivocado antes.

Los cuatro son números en torno a 1: un sondeo con todos en 1 pesa 1, y cada factor lo sube o lo baja en proporción.

#### La muestra: cuántas entrevistas hay detrás  (h3, id `muestra`)

Un sondeo pregunta a unos cientos o unos miles de personas para deducir lo que harán millones. Aunque las eligiera
perfectamente al azar, su resultado variaría de una muestra a otra: es el error de muestreo, el único que la
estadística sabe calcular de antemano. Para un partido que ronda el 30 %, un sondeo de 1.000 entrevistas tiene un
margen de ±2,8 puntos (al 95 %); uno de 4.000, de ±1,4. En España las muestras van de unos pocos cientos a varios
miles: la mediana en las generales de este ciclo es de 1.200 entrevistas y el barómetro del CIS pasa de 4.000. No
tendría sentido que todos pesaran igual.

Tampoco que pesaran en proporción a su muestra. El error de muestreo es solo una parte del error de un sondeo, y a
menudo la menor: a quién se llama y quién contesta, cómo se pondera por recuerdo de voto y cómo se pasa de la
intención directa a la estimación (la «cocina») añaden errores que no se reducen por hacer más entrevistas. Lo hemos
medido: en este ciclo, las inclinaciones de las encuestadoras en el voto del PP tienen una dispersión de 2,4 puntos, muy
por encima de lo que explicarían sus muestras; y el CIS, con más de 4.000 entrevistas por barómetro, oscila de
un mes a otro como lo haría una muestra de 400 a 460.

Por eso más entrevistas dan más peso, pero cada vez menos: el peso crece con la raíz cuadrada de la muestra. Un sondeo
con cuatro veces más entrevistas pesa el doble, no el cuádruple:

  F_muestra = √(nᵢ / ñ)

*nᵢ* son las entrevistas del sondeo *i*, ya completadas y recortadas (más abajo), y *ñ*, la mediana de las de los
sondeos de esa elección: hoy, unas 1.300 en las generales. Dividir por *ñ* no cambia el promedio, porque lo que cuenta
es cuánto pesa cada sondeo frente a los demás; solo pone el peso en una escala legible, en la que un sondeo de muestra
típica pesa 1, uno de cuatro veces la mediana pesa 2 y uno de la cuarta parte, 0,5. Se guarda en `weight_sample`.

##### Por qué una raíz y no otra cosa

Si el único error fuera el de muestreo, el peso óptimo sería proporcional a *n*, la inversa de la varianza
*p*(1 − *p*)/*n*. Si todo el error fuera de método y la muestra no importara, todos los sondeos deberían pesar igual. La
realidad queda en medio. A modo de ilustración: con un error de método de 1,5 puntos que la muestra no reduce, el peso
óptimo de un sondeo de 4.000 entrevistas frente a uno de 500 sería 2,3 veces mayor; proporcional a *n* serían 8 veces,
y con la raíz, 2,8. La raíz es el compromiso habitual en los promedios de encuestas, y la misma forma que usaba
FiveThirtyEight.

##### Cuando falta la muestra

No todas las encuestadoras publican sus entrevistas: de los 2.935 sondeos de generales que entran en el promedio desde
1977, 558 (el 19 %) no la dan. La muestra publicada se guarda en `sample_size`, vacía si falta. A los sondeos sin ella
les asignamos, por este orden, la mediana de las de su encuestadora en esa elección; si no hay, la de su encuestadora
en todas las elecciones del ámbito; y si es su primer sondeo, la mediana de todos los sondeos del ámbito, sin
agregadores ni paneles. Una encuestadora suele repetir su diseño de un sondeo a otro, así que su propia historia es la
mejor pista.

##### Cuando la muestra es desmesurada

Recortamos las muestras a un máximo: el menor entre 5.000 entrevistas y la mayor muestra que no se aparta más de tres
desviaciones típicas de la media de todos los sondeos del ámbito desde 1977.[nota 1] Más allá de unos miles de
entrevistas, el error de muestreo ya es pequeño frente al de método; sin tope, una macroencuesta de 15.000 entrevistas
pesaría casi cuatro veces lo que un sondeo típico, por una precisión que en la práctica no tiene. La muestra que usa el
modelo, completada y recortada, se guarda en `proc_sample`.

##### El CIS, por su muestra

El barómetro del CIS de abril tuvo 4.020 entrevistas: por su muestra pesaría 1,76, casi el doble que un sondeo típico.
Pero sus cifras oscilan mucho más de lo que esas entrevistas permiten, y en la sección siguiente veremos que acaba
pesando como un sondeo de unas 600.

#### La dispersión: ¿oscilan las cifras lo que deberían?  (h3, id `dispersion`)

Si una encuestadora repitiera el mismo sondeo cada semana sin que la opinión cambiara, sus cifras subirían y bajarían
algo de una entrega a otra por puro azar del muestreo, y cuánto se puede calcular a partir de su muestra. Eso nos
permite comprobar si los sondeos se comportan como dicen sus muestras. Muchos no lo hacen, y fallan por los dos lados:

- **Unos oscilan más de lo que permite su muestra.** El CIS publica barómetros de más de 4.000 entrevistas, pero sus
  cifras saltan de un mes a otro casi tres veces más de lo que correspondería a esa muestra: algo en su método (la
  estimación, el diseño, el campo) añade ruido que las entrevistas no compensan.
- **Otros oscilan menos.** Es lo más común en España: casi todas las encuestadoras se mueven menos de lo que el azar
  permitiría. Puede ser porque entrevistan siempre a un mismo panel, porque ponderan por recuerdo de voto (lo que ata
  cada sondeo al anterior) o porque corrigen hacia lo que publican las demás, lo que llamamos rebaño. Sea cual sea la
  causa, sus cifras aportan menos información nueva de la que sugiere su muestra.

Comparamos cuánto se aparta cada encuestadora del promedio con cuánto debería apartarse por azar. Si se aparta de más,
pesa como la muestra que produciría esa oscilación; si se aparta de menos, también pesa menos. Para la encuestadora
*c* sumamos, sobre todos sus sondeos *i* y todos los partidos, lo que observamos y lo que esperaríamos por azar:

  SS_obs = Σ (rᵢ − r̄)²     SS_esp = (1 − 1/k) · Σ 10⁴ · pᵢ(1 − pᵢ) / nᵢ

*rᵢ* es lo que se aparta el sondeo del promedio de ese día, en puntos. Le restamos su media, *r̄*, porque esa es la
inclinación propia de la encuestadora, que corregimos aparte (en los efectos de encuestadora): aquí solo interesa
cuánto oscila alrededor de ella. *pᵢ* es el voto del partido según el promedio, en tanto por uno, y *nᵢ*, la muestra
del sondeo: *pᵢ*(1 − *pᵢ*)/*nᵢ* es la varianza del muestreo, y el 10⁴ la pasa a puntos al cuadrado. El factor
(1 − 1/*k*), con *k* sondeos, compensa que al restar la media se gasta un dato.

El cociente R_c = √(SS_obs / SS_esp) dice cuántas veces oscila la encuestadora lo que le tocaría: 1 es lo esperado, 2
el doble, 0,5 la mitad. Con pocos sondeos es poco fiable, así que lo acercamos a 1:

  R̃_c = 1 + (R_c − 1) · k / (k + 5)

Con 5 sondeos la medida cuenta la mitad; con 20, el 80 %; por debajo de 5 no la medimos. El factor de dispersión es

  f_c = 1 / R̃_c si oscila de más;  f_c = H̃_c si oscila de menos (como poco, 0,1)

donde H̃_c es el mismo cociente medido para el rebaño. Multiplica al factor de muestra, así que la muestra efectiva de la
encuestadora es *n* · *f*²: un sondeo de 4.000 entrevistas con *f* = 0,5 pesa como uno de 1.000.[nota 2]

##### Por qué el rebaño se mide de otra manera

Para saber si una encuestadora oscila de menos no basta con la cuenta anterior. La comparamos con un promedio calculado
**sin sus propios sondeos**: si estuvieran dentro, se compararía en parte consigo misma y parecería más pegada al
consenso de lo que está. Solo cuentan los sondeos que **publican su muestra**, porque con una muestra imputada
estaríamos inventando lo que debería oscilar, y solo los **cuatro partidos principales** (hoy PP, PSOE, Vox y Frente
Amplio). La probabilidad de oscilar tan poco por puro azar, según una χ² con *n* − 1 grados de libertad, se publica
con el resto de las medidas del rebaño en el <a href="/casas{scope_query}">Ranking Encuestadoras</a> (tabla
`pollsters_herding`: `n`, `ss_obs`, `ss_exp`, `ratio`, `p_value`). El rebaño no entra en el rating: solo resta peso en
el promedio.

El principio es sencillo: si un sondeo tiene menos información de la que aparenta, no debe pesar como si la tuviera.[nota 3]

Tabla (pie: «Datos del cálculo del 20 de octubre de 2026, generales»):

| Encuestadora | Sondeos | Oscila (veces lo esperado) | Rebaño | Factor *f* |
|---|---|---|---|---|
| CIS | 36 | 2,79 | — | 0,39 |
| Hamalgama | 21 | 0,48 | 0,58 | 0,58 |
| Celeste Tel | 13 | 0,56 | 0,65 | 0,65 |
| NC Report | 23 | 0,56 | 0,67 | 0,67 |
| Simple Lógica | 14 | 0,58 | 0,67 | 0,67 |
| SocioMétrica | 34 | 0,68 | 0,78 | 0,78 |
| GAD3 | 12 | 0,68 | 0,80 | 0,80 |
| 40dB | 35 | 0,79 | 0,83 | 0,83 |
| Sigma Dos | 37 | 0,80 | 0,87 | 0,87 |

##### El CIS y Hamalgama, en los dos extremos

El CIS de abril, con 4.020 entrevistas, pesaba por su muestra 1,76. Sus cifras oscilan 2,79 veces lo esperado;
encogido por sus 36 sondeos, 2,57. Su factor es 1/2,57 = 0,39 y su muestra efectiva, 4.020 × 0,39² ≈ 610 entrevistas:
casi lo mismo que medimos directamente al comparar sus barómetros entre sí, que oscilan como una muestra de 400 a 460.
En el otro extremo, Hamalgama se aparta del promedio menos de la mitad de lo que le correspondería por azar: su factor
es 0,58, y un sondeo suyo de 1.000 entrevistas cuenta como uno de unas 340.

El factor *f* no se guarda en Polls: el promedio lo calcula cada vez.

#### La sobrerrepresentación: que nadie pese más por publicar más  (h3, id `sobrerrepresentacion`)

Las encuestadoras no publican al mismo ritmo. Unas sacan un sondeo al mes; otras, en campaña, uno cada día. Si cada
entrega contara como un sondeo nuevo, el promedio acabaría pareciéndose a la encuestadora más prolífica y no al
conjunto. Y muchas de esas entregas no son sondeos nuevos, sino el mismo trabajo de campo con unos días más.

En las generales de 2023, GAD3 publicó para ABC un seguimiento diario que empezó el 29 de junio y fue acumulando
entrevistas: 1.004 el 30 de junio, 2.001 el 4 de julio, 4.001 el 8 y 8.001 el 17. Dieciséis entregas, cada una con
todas las entrevistas de las anteriores dentro: contarlas como dieciséis sondeos sería contar dieciséis veces a las
primeras mil personas. Sigma Dos hizo lo mismo para El Mundo con una ventana móvil de siete días: cada entrega
compartía seis días de campo con la anterior.

Aplicamos dos reglas. Si el campo de un sondeo se solapa con el de una entrega posterior de la misma encuestadora, solo
cuenta la posterior, la más completa. Y si una encuestadora publica varios sondeos en pocos días sin solaparse, forman
un grupo que pesa como un único sondeo: el primero abre el grupo, entran en él los que terminan en los cuatro días
siguientes, y el primero que queda fuera abre el siguiente:

  oᵢ = 0 si se solapa con una entrega posterior de su encuestadora;  oᵢ = 1 / N_g en otro caso

con *N_g* el número de sondeos de su grupo. Dos sondeos en cuatro días pesan la mitad cada uno; tres, un tercio. Se
guarda en `weight_over`. Del seguimiento de GAD3 solo cuenta la última entrega, la del 17 de julio, y con su muestra
recortada al tope: 4.715 entrevistas en lugar de 8.001.

##### Por qué la encuestadora y no el medio

Una misma encuestadora suele trabajar para varios medios, a veces con el mismo campo. Para nosotros el sondeo es de
quien hace las entrevistas, no de quien las paga. En 2023, el sondeo de Sigma Dos para Antena 3 (del 23 de junio al 3
de julio) y el suyo para El Mundo (del 24 de junio al 4 de julio) se solapan casi por completo, y solo cuenta el
segundo.

##### Por qué cuatro días

Una ventana corta solo actúa en las rachas de campaña y deja en paz el ritmo normal, semanal o mensual, de las
encuestadoras españolas.[nota 4] En el ciclo actual de las generales, solo 6 de los 328 sondeos se descartan por
solaparse con otro y ninguno reparte peso por publicar en racha; en el ciclo de 2023 se descartaron 133 entregas, la
mayoría en las semanas finales de la campaña.

#### El rating: cuánto se ha equivocado antes  (h3, id `rating`)

No todas las encuestadoras aciertan igual, y la diferencia no es pequeña ni aleatoria. En cada elección comparamos a
cada encuestadora con las demás que publicaron en las mismas fechas, que es lo justo: el error que todas cometen a la
vez porque la campaña fue difícil de medir no es culpa de ninguna. De esa comparación, repetida en todas las elecciones
generales y autonómicas desde 1977, sale un rating de 0 a 100. Cómo se calcula se explica en
<a href="{method_href('rating')}">¿Cómo calculamos el rating de las encuestadoras?</a>; aquí interesa cómo entra en el
promedio. El peso crece con el rating, pero cada vez más despacio:

  r_c = ln(1 + R_c/100) / ln(1 + R̄/100)

*R_c* es el rating de la encuestadora *c* calculado para esa elección (se copia en cada sondeo como `rating`), y *R̄*,
la media de los de todas las encuestadoras calificadas: hoy, 27,5 entre 129 encuestadoras, la mayoría pequeñas o solo
autonómicas. Una encuestadora con el rating medio pesa 1 y una con rating 0 no pesa nada. Como en la muestra, dividir
por *R̄* no cambia el promedio, solo pone el factor en una escala legible; las encuestadoras que publican hoy en las
generales están por encima de la media de todas las calificadas, así que casi todas pesan más de 1. El factor se guarda
en `weight_rating`.

Tabla (pie: «Datos del cálculo del 20 de octubre de 2026, generales»):

| Encuestadora | Rating | Factor *r* |
|---|---|---|
| GAD3 | 77,1 | 2,35 |
| 40dB | 60,7 | 1,95 |
| Sigma Dos | 60,6 | 1,95 |
| NC Report | 35,6 | 1,25 |
| Hamalgama | 34,8 | 1,23 |
| InvyMark | 28,1 | 1,02 |
| CIS | 17,2 | 0,65 |

##### Por qué un logaritmo

Con un factor proporcional al rating, GAD3 pesaría 4,5 veces más que el CIS; con el logaritmo, 3,6 veces. El logaritmo
hace pequeña la diferencia entre las mejores encuestadoras y mantiene grande la que las separa de las peores: entre dos
buenas, la mejor no se queda con casi todo el peso, y una con un historial muy malo apenas cuenta. Es una decisión de
diseño, no un valor medido.

##### Por qué el rating de esa elección

El rating que usa el promedio de una elección se calcula solo con las elecciones anteriores: el promedio de 2019 nunca
usa lo que las encuestadoras acertaron en 2023. Es la condición para que las pruebas del modelo sobre elecciones pasadas
sean honestas: no puede aprender de un futuro que entonces no conocía. Una encuestadora sin historial recibe el rating
que le da su calidad previa, una evaluación de su método y su transparencia que se explica en la página del rating, y
se transforma con la misma fórmula.

#### Las cuentas completas de un sondeo  (h3, id `cuentas`)

Con los cuatro factores, las cuentas de dos sondeos reales de septiembre:

| | CIS (4 sept) | GAD3 (3 sept) |
|---|---|---|
| Entrevistas | 4.042 | 1.010 |
| Muestra √(n/ñ) | 1,76 | 0,88 |
| Dispersión *f* | 0,39 | 0,80 |
| Sobrerrepresentación *o* | 1 | 1 |
| Rating *r* | 0,65 | 2,35 |
| **Peso** | **0,45** | **1,66** |

El sondeo de GAD3, con la cuarta parte de entrevistas, pesa 3,7 veces más que el barómetro del CIS. No es un juicio
sobre ninguna institución: las cifras del CIS oscilan casi tres veces más de lo que permite su muestra y, frente a las
demás encuestadoras, se ha equivocado más en las elecciones anteriores. Los dos hechos están medidos y se pueden
comprobar.

### 4. Los efectos de encuestadora

Lead: «Cada encuestadora tiende a dar a cada partido algo más o algo menos que las demás; antes de promediar, le
restamos esa inclinación.»

Cada encuestadora tiene su manera de hacer las cosas: cómo elige a quién preguntar, cómo pondera por recuerdo de voto,
cómo pasa de la intención directa a la estimación. Esas decisiones no producen errores al azar, sino inclinaciones
estables. En este ciclo, el CIS da al PP unos seis puntos menos que el consenso de las demás encuestadoras, barómetro
tras barómetro: no es ruido, es su manera de medir.

Si no lo corregimos, el promedio se mueve según quién haya publicado esa semana: cuando coinciden tres sondeos de
encuestadoras que dan más voto a un partido, el partido parece subir aunque nadie haya cambiado de opinión. Por eso,
antes de promediar, a cada sondeo le restamos la inclinación de su encuestadora en cada partido:

  yᵢ = pᵢ − e_c,p

*pᵢ* es el voto que el sondeo *i* da al partido *p*; *e_c,p*, el efecto de su encuestadora *c* en ese partido, en puntos
(positivo si suele darle más que el consenso, negativo si menos); *yᵢ*, la cifra corregida, la que entra en el promedio.
El efecto se estima con dos fuentes, que luego combinamos.

#### Lo que dice el ciclo

Cuánto se apartan sus sondeos del promedio en el ciclo actual: la media ponderada de las diferencias,

  d_c,p = Σᵢ wᵢ (pᵢ − m(tᵢ)) / Σᵢ wᵢ

con *m*(*tᵢ*) el promedio el día del sondeo y *wᵢ* su peso en el promedio (muestra, dispersión, sobrerrepresentación
y rating), con un 10 % menos por cada año de antigüedad. Su incertidumbre es el error estándar de esa media ponderada,
√(varianza ponderada / *n_ef*), con *n_ef* el tamaño efectivo de Kish de esos pesos. Con menos de cinco sondeos en el
ciclo no se mide.

Hay una trampa: el promedio con el que comparamos ya contiene los sondeos de la propia encuestadora, con su inclinación
dentro, así que la primera medida se queda corta. Por eso iteramos tres veces: medimos los efectos, corregimos los
sondeos, recalculamos el promedio con ellos y volvemos a medir.

#### Lo que dice su historia

Lo que hizo la encuestadora en elecciones anteriores. En cada una tomamos sus sondeos de los 90 días previos y medimos
cuánto se apartaron del resultado; restamos lo que se apartó la media de todas las encuestadoras, porque esa parte es
un error compartido y no una inclinación suya; y lo expresamos en proporción al tamaño del partido (equivocarse en dos
puntos con un partido del 10 % no es lo mismo que con uno del 30 %), para aplicarlo al tamaño actual.

La historia no se usa tal cual. Hemos medido cuánto se repite la inclinación de una encuestadora de una elección a la
siguiente: entre un 40 % y un 70 %, y un 61 % entre 2023 y este ciclo. Persiste, pero solo en parte, así que la historia
entra como un punto de partida prudente: se encoge hacia cero, como si se le sumara una elección imaginaria de diez
sondeos sin inclinación; las elecciones recientes pesan más (un 10 % menos por año) y las autonómicas, la mitad que las
generales; y su incertidumbre es fija, el 8 % del tamaño del partido, que es lo que se separan entre sí las
encuestadoras en una elección típica (medimos entre un 3 % y un 13 %).

#### Por qué fiarnos del consenso

Restar la inclinación es acercar cada encuestadora al consenso de las demás. Tiene sentido porque, frente al resultado,
el consenso suele tener más razón que la encuestadora que se aparta: lo hemos medido en catorce ciclos y 43
encuestadoras. Cuanto más se aleja una encuestadora del consenso, más se equivoca frente al resultado (la correlación
es de 0,35), y las que se apartan más de un punto fallan de media 2,4 puntos, frente a 1,9 las demás.

#### Cómo se combinan

Pesando cada fuente por su precisión, la inversa de su varianza:

  e_c,p = (d / σ_d² + h / σ_h²) / (1/σ_d² + 1/σ_h²)

con *d* y σ_d la desviación del ciclo y su error, y *h* y σ_h, las de la historia. Con muchos sondeos en el ciclo, σ_d
es pequeño y manda el ciclo; con pocos, la historia.[nota 5]

#### Por qué los efectos suman cero

Hay algo que los datos no pueden decirnos: si todas las encuestadoras se equivocan a la vez hacia el mismo lado,
ninguna se aparta de las demás, y desde dentro del ciclo es invisible. Los efectos de encuestadora solo se pueden medir
unos respecto de otros, así que fijamos su referencia: los recentramos para que su media, ponderada con el peso *W_c*
de cada encuestadora en el promedio, sea cero:

  Σ_c W_c · e_c,p = 0

La corrección no mueve el nivel medio del promedio: solo cambia qué encuestadora pesa en qué fecha. El error que
comparten todas no se ignora: va a la simulación, como error histórico de los sondeos frente al resultado (ver
<a href="{method_href('escanos')}">¿Cómo simulamos los escaños?</a>). También probamos a estimar ese error común con la
historia, restando a todas lo que suelen equivocarse juntas en cada partido, y no funcionó: el error común no se repite
de una elección a otra. Está desactivado.[nota 6]

#### El CIS y el PP, hoy

Tabla (pie: «Datos del cálculo del 20 de octubre de 2026»):

| | Efecto | Error |
|---|---|---|
| Ciclo: 36 barómetros frente al promedio | −6,18 | 0,64 |
| Historia: elecciones desde 1977 | −0,72 | 2,57 |
| Combinación por precisión | −5,86 | 0,62 |
| Tras el recentrado | **−5,94** | 0,62 |

La historia apenas aporta: en su conjunto, el CIS no se ha inclinado mucho contra el PP. Pero el ciclo es tan rotundo,
36 sondeos coherentes, que se queda casi con todo el peso. A cada barómetro del CIS le sumamos casi seis puntos de PP
antes de promediarlo.

Los efectos no se guardan en Polls. La historia vive en `pollsters_parties`, una fila por encuestadora y partido:
el tamaño del partido (`level`), los sondeos, la desviación frente al resultado y su error (`n_result`,
`dev_result`, `dev_result_err`), la del sector (`industry`), la desviación descontada la del sector
(`dev_result_c`) y, a título informativo, la del ciclo frente al consenso (`n_cycle`, `dev_cycle`, `dev_cycle_err`).
La desviación de cada sondeo frente al resultado está en `polls_results.error`.[nota 7] Los gráficos de la web
muestran siempre los sondeos tal como se publicaron: la corrección solo entra en el promedio.

### 5. Un ejemplo: el CIS desde 2018

Ya reescrita (`2fe0171`, `3fab08f`). Sin cambios de texto.

### 6. La regresión local

Lead: «Para cada día, ajustamos una recta a los sondeos cercanos en el tiempo y el promedio es la altura de esa recta
en ese día.»

Con los sondeos corregidos y cada uno con su peso, queda sacar de ellos una cifra para cada día. Los sondeos llegan a
saltos, unos días tres y otros ninguno, cada uno con su error. La solución más simple, la media de los últimos sondeos,
tiene un defecto: la media de las últimas cuatro semanas describe el centro de esas cuatro semanas, no el día de hoy.
Cuando el voto se mueve, llega tarde.

Nuestro promedio es una **regresión polinómica local ponderada por kernel** (en inglés, *kernel-weighted local
polynomial regression*).[nota 8] El nombre es largo, pero cada palabra es una pieza del método. Vamos de la última a la
primera.

**Regresión.** Una regresión busca la línea que mejor resume una nube de puntos. Los puntos son los sondeos: en
horizontal, la fecha en que terminó su trabajo de campo (`date`); en vertical, el voto que dan al partido, ya
corregido de su efecto de encuestadora. La línea «que mejor resume» es la que hace más pequeña la suma de las
distancias al cuadrado entre ella y los puntos: el método de mínimos cuadrados.

**Polinómica.** La línea podría ser una recta (un polinomio de grado 1), una parábola (de grado 2) o algo más retorcido.
Usamos una recta: una parábola sigue mejor las curvas, pero en el último día, con sondeos solo a un lado, tiende a
dispararse; la recta es el grado más bajo que sigue una tendencia.

**Local.** No ajustamos una única recta a todo el ciclo (el voto no se mueve en línea recta durante tres años), sino
una distinta para cada día, con los sondeos cercanos a ese día, y nos quedamos con su altura justo en él. Al día
siguiente, otra. La línea que publicamos une esas alturas, una por día.

**Ponderada.** En cada recta, cada sondeo cuenta con un peso ωᵢ, el producto de dos cosas: su peso propio *wᵢ* (el de
<a href="#peso">Cuánto pesa cada sondeo</a>) y su cercanía al día que se calcula.

**Por kernel.** La cercanía la da un *kernel*, un núcleo: una función que vale 1 en el día que se calcula y baja al
alejarse. Usamos el kernel gaussiano, la campana de Gauss:

  K(tᵢ − t) = exp( −(tᵢ − t)² / (2h²) )

con *t* el día que se calcula, *tᵢ* el fin de campo del sondeo y *h* la **anchura de banda** (*bandwidth*), en días,
que decide lo ancha que es la campana: un sondeo a *h* días cuenta el 61 %; a 2*h*, el 14 %; a 3*h*, el 1 %. Elegimos
la campana de Gauss porque es suave, sin escalones (un sondeo no entra ni sale de golpe del promedio de un día al
siguiente), y porque no tiene un corte arbitrario: todos los sondeos cuentan algo, aunque los lejanos casi nada.[nota 9]

Juntando las piezas, para cada día *t* buscamos la altura *a* y la pendiente *b* de la recta que menos se separa de los
sondeos:

  min_{a,b} Σᵢ ωᵢ · ( yᵢ − a − b·(tᵢ − t) )²,  con ωᵢ = wᵢ · K(tᵢ − t)

El promedio del día es *a*, la altura de la recta en *t*; la pendiente *b* dice si el partido sube o baja en ese
momento.

#### Por qué una recta y no una media

La solución de ese problema tiene una forma que lo explica todo:

  a = ȳ_ω + b · (t − t̄_ω)

con ȳ_ω la media ponderada de los sondeos y t̄_ω la de sus fechas, el centro de gravedad de la campana. Un día con
sondeos a los dos lados tiene t̄_ω ≈ *t*, y la recta y la media coinciden. Pero hoy, en el último día, todos los sondeos
están detrás: el centro de gravedad queda semanas atrás y la media describe aquel momento. La recta corrige ese
retraso sumando a la media la pendiente por la distancia hasta hoy: si un partido lleva semanas subiendo, la recta lo
lleva hasta hoy, y la media no. Lo hemos medido sobre las generales pasadas: leído seis días antes de votar, el
promedio se equivoca de media en 1,70 puntos por partido, frente a 1,74 del último sondeo publicado y 1,86 de la media
de las cuatro últimas semanas.

#### La anchura de la campana  (h3, id `anchura`)

Todo depende de *h*. Con una campana estrecha, el promedio sigue cada vaivén, también el ruido de cada sondeo; con una
ancha, es estable pero tarda en reaccionar a los cambios reales. No hay un valor correcto universal, y no lo elegimos a
mano: sale de los datos, en dos pasos.

**Una anchura de partida con el método ISJ.** Elegir la anchura de una campana es un problema clásico de la
estadística, el de estimar la forma de una distribución a partir de una muestra (la estimación de densidad por kernel).
Sheather y Jones (1991) propusieron calcular la anchura óptima con los propios datos, y Botev, Grotowski y Kroese (2010)
mejoraron su método: el *Improved Sheather-Jones* (ISJ) no presupone que los datos tengan forma de campana, así que
funciona también cuando se agrupan en varios montones, como los sondeos, que se acumulan en las campañas y escasean
entre ellas. Lo aplicamos al **calendario** de los sondeos de cada partido, a sus fechas con sus pesos, no a sus cifras:
la pregunta que responde es qué anchura describe mejor cómo se reparten los sondeos en el tiempo.[nota 10]

**Una anchura que se adapta a cada día.** Una sola anchura para todo el ciclo sería un mal compromiso: en campaña se
publican varios sondeos al día, y en verano, uno por semana. La anchura cambia con la densidad de sondeos alrededor de
cada día, siguiendo la idea de la regla de la raíz cuadrada de Abramson (1982):

  h_{k+1}(t) = h_k(t) · √( G / f_k(t) )

*f_k*(*t*) mide cuántos sondeos hay cerca del día *t* (la suma de sus pesos por la campana) y *G* es la media geométrica
de *f_k* en todos los días, para que la anchura media siga en torno a la de partida. Donde hay el doble de sondeos que de
media, la campana se estrecha por √2; donde hay la mitad, se ensancha otro tanto. Repetimos el ajuste tres
veces.[nota 11] Al final de la serie, con sondeos solo a un lado, la densidad cae y la campana se ensancha sola: en
septiembre, la del PP iba de 61 a 130 días según el momento del ciclo, con una mediana de 70 y 129 en la última fecha.

Con la anchura del PP leída el 24 de octubre en el cálculo del 20, 81 días, un sondeo cuenta así según su antigüedad (pie: «Datos del cálculo
del 20 de octubre de 2026»):

| Antigüedad | 1 semana | 1 mes | 2 meses | 3 meses | 6 meses |
|---|---|---|---|---|---|
| Peso de la campana | 100 % | 93 % | 76 % | 54 % | 9 % |

Es una campana ancha: hoy no hay campaña ni avalancha de sondeos, y lo que manda es que ningún sondeo suelto mueva la
línea. Cada partido tiene la suya: los grandes, con más sondeos, en torno a 80 días; los pequeños, que aparecen en
menos, entre 120 y 150.

#### Lo que se sigue de ahí  (h3, id `consecuencias`)

- **El pasado se afina.** La campana mira a los dos lados, así que el promedio de hace dos semanas cambia un poco cuando
  llegan sondeos nuevos: no es una corrección, es más información.
- **Cada partido se promedia por separado**, con los sondeos que lo incluyen, si son al menos tres: una recta necesita
  tres puntos para que tenga sentido hablar de su error.
- **Tras el último sondeo**, la recta se prolonga diez días y después el valor se mantiene.
- **Los gráficos muestran los sondeos tal como se publicaron**, pero la recta se ajusta a sus cifras corregidas: por
  eso no siempre pasa «por el medio» de los puntos.

La regresión no escribe nada en Polls: su peso por cercanía aparece como `weight_kernel` en los pesos descargables de
cada partido, junto a la cifra corregida, `pct_adj`.

### 7. La banda del promedio

Lead: «La banda que acompaña a la línea es su intervalo de confianza del 95 %: cuánto sabemos del consenso de las
encuestadoras ese día.»

La banda responde a una pregunta concreta: con los sondeos que tenemos, ¿con cuánta precisión conocemos el consenso de
las encuestadoras ese día? No dice dónde estará el voto; esa es otra pregunta, y la contestamos más abajo.

Sale del error estándar de la altura *a* de la recta de cada día. La fórmula habitual de ese error supone que cada
sondeo es una medición independiente, y en las encuestas españolas no lo es: los sondeos de una misma encuestadora se
parecen entre sí, porque comparten método, a menudo panel, y lo que quede de su inclinación. Diez sondeos de una
encuestadora que publica cada semana no son diez confirmaciones independientes, sino algo más parecido a una mirada
repetida diez veces; tratados como independientes, la banda saldría engañosamente estrecha.

#### Por qué agrupamos por encuestadora

Calculamos el error agrupando los sondeos por encuestadora (en la jerga, un error estándar robusto con clústeres): las
desviaciones de cada encuestadora se suman antes de combinarse con las de las demás, de modo que si se equivocan
juntas, cuentan juntas:

  Var(a) = c · [ (XᵀΩX)⁻¹ (Σ_g u_g u_gᵀ) (XᵀΩX)⁻¹ ]₁₁,  con u_g = Σ_{i∈g} ωᵢ xᵢ eᵢ

*xᵢ* = (1, *tᵢ* − *t*) es la fila del sondeo *i* en la recta; *eᵢ*, lo que se aparta de ella; Ω, los pesos ωᵢ; *g*
recorre las encuestadoras; y *c* = *G*/(*G* − 1) · (*n_ef* − 1)/(*n_ef* − 2) es la corrección habitual para muestras
pequeñas (CR1), con *G* encuestadoras y *n_ef* = (Σωᵢ)²/Σωᵢ² sondeos efectivos (el tamaño efectivo de Kish).

#### Por qué una t de Student

Con pocas encuestadoras, el propio error estándar es incierto, así que la banda usa una *t* de Student en lugar de una
normal:

  a ± t_{0,975; ν} · √Var(a),  con ν = min(n_ef − 2, G − 1)

Los grados de libertad son, como mucho, las encuestadoras menos una, porque la información independiente la aportan
las encuestadoras, no los sondeos. Con veinte encuestadoras en la ventana, *t* ≈ 2,1; con cinco, *t* ≈ 2,8, y la banda
se ensancha un tercio. En el PP, agrupar por encuestadora sube el error estándar de 0,20 a 0,25 puntos y ensancha la
banda de ±0,41 a ±0,53.

La banda mide dónde está el consenso de las encuestadoras que publican, no dónde estará el voto: el error que cometen
todas a la vez frente al resultado es invisible desde dentro del ciclo, y el resultado cae a menudo fuera de ella sin
que sea un fallo.[nota 12] Ese error se suma más abajo, en el intervalo del voto estimado.[nota 13] La banda no se
guarda en Polls; en los pesos descargables de cada partido están su error estándar, sus límites, los sondeos usados
(`nobs`), los efectivos (`neff`) y la anchura del día.

### 8. Un ejemplo: el promedio del PP hoy

Título, lead, gráfico, tabla y CSV como ahora (`weights_example`, vivo), con los cambios de 7.3. Texto:

Todo lo anterior, con los datos de hoy: el partido que encabeza el promedio, por dentro, en un día concreto. Qué
sondeos entran, cuánto pesa cada uno y por qué.

En el último cálculo publicado, el del {example.published|date}, el promedio del {example.name} leído el
{example.date|date} es del {stats.mean|pct}, con una banda del {stats.lo|pct} al {stats.hi|pct}. {Si
example.date > example.published: «El día de lectura es posterior al del cálculo porque la recta se prolonga diez días
tras el último sondeo.»} Entran {stats.nobs|int} sondeos, que por sus pesos equivalen a {stats.neff|num(1)} sondeos
independientes de igual peso: es el tamaño efectivo de Kish, *n_ef* = (Σωᵢ)²/Σωᵢ². Cuanto más desiguales son los
pesos, más se aleja del número de sondeos: diez sondeos que pesan lo mismo informan como diez, pero si uno pesa por los
otros nueve juntos, informan como dos. La campana de ese día tiene {stats.bandwidth|num(0)} días de anchura.

[gráfico]

Cada punto es un sondeo, más grande y más intenso cuanto más pesa; al pasar el puntero se ve quién lo hizo, cuándo y
con qué peso. La recta es la de ese día, con su banda del 95 %, y la campana, en el eje de la derecha y de 0 a 1, es el
peso por cercanía. Las guías marcan el día leído y el valor que da la recta. Los puntos muestran la cifra que publicó
cada encuestadora, pero la recta se ajusta a las cifras corregidas de su efecto de encuestadora: por eso no pasa «por el
medio» de la nube.

[tabla plegada «Ver los {n_rows} sondeos», como ahora]

La tabla lista los sondeos, los más recientes primero: su fecha de fin de campo; la encuestadora y el medio; las entrevistas
publicadas; el voto que dio al partido (Publicada) y el que entra en la recta, sin su efecto de encuestadora
(Corregida); el peso total, con una barra de 0 a 3; y los factores que lo forman: Cercanía, la campana, que vale 1 el
día leído; Solape, la sobrerrepresentación *o*; Muestra, √(*n*/*ñ*) con la dispersión *f* dentro; y Rating, el factor
*r*. {top: «El sondeo que más pesa es el de {top.pollster} del {top.date|date_short}, con {top.weight|num(2)}.»}

[enlace CSV, como ahora]

### 9. De las encuestas al voto estimado

Lead: «El voto estimado de cada partido es el promedio leído en la fecha de la estimación; su intervalo suma más
errores que la banda.»

El promedio se lee en un día, la fecha de la estimación: normalmente el último que alcanza la línea, diez días después
del último sondeo, y nunca más tarde de seis días antes de la elección. Es la fecha que dan las páginas de cada
elección («el promedio de sondeos leído el…»). Lo que falta hasta el 100 % queda como «otros»: otras candidaturas y
voto en blanco.

Los modos Hoy y Elección comparten ese voto estimado; solo cambia la anchura de su intervalo, mucho mayor que la banda
porque suma hasta tres fuentes de incertidumbre:

  σ_voto = √( s² + e² + d² )

- ***s*** es el error del promedio, el de su banda: cuánto sabemos del consenso de las encuestadoras.
- ***e*** es el error histórico de los sondeos frente al resultado, la pieza que falta en la banda: el error que cometen
  todas a la vez. Lo estimamos con el último sondeo de cada encuestadora antes de cada elección pasada, con una
  regresión que lo predice según el tamaño del partido y si es regional. Crece con el tamaño: en las generales, el
  último sondeo de cada encuestadora se ha equivocado de media en unos 0,4 puntos con los partidos de menos del 5 %,
  unos 2,3 con los de entre el 10 % y el 20 % y unos 3 con los de más del 30 %.
- ***d*** es la deriva, cuánto puede moverse la opinión en los días que faltan hasta la elección. Solo se suma en el
  modo Elección.

Con ese voto y esos errores, el modelo sortea {figures.n_sim|int} elecciones posibles (en las generales), con una *t*
de Student en lugar de una normal para que los errores grandes sean algo más probables que en una campana de Gauss, y
reparte los escaños de cada una. Cómo, y de dónde salen el error histórico y la deriva, se cuenta en
<a href="{method_href('escanos')}">¿Cómo simulamos los escaños?</a>.[nota 14]

### 10. Notas

Ver sección 6 de este documento.

### Firma

> Por Luis Sancho (<a href="{author_url}" rel="me">{author_handle}</a> en X).

## 6. Notas

Numeradas por orden de aparición en la página; el número de cada llamada sale de su orden (7.1).

1. (Muestra desmesurada) Las tres desviaciones típicas son una convención estadística, no un valor medido: no sabemos
   si 2,5 o 4 funcionarían mejor. Como se calculan con todos los sondeos del ámbito, el tope cambia un poco con cada
   carga de sondeos: en septiembre de 2026 era de 4.715 entrevistas.
2. (Dispersión) La varianza esperada supone un muestreo aleatorio simple. Las encuestas reales tienen un efecto de
   diseño que suele multiplicar esa varianza por 1,2 o 1,5: teniéndolo en cuenta, las encuestadoras que oscilan menos
   de lo esperado quedarían aún más lejos de lo que deberían oscilar, así que la medida del rebaño es conservadora.
3. (Dispersión) Los umbrales son elecciones razonables que no hemos comparado con otras: el mínimo de cinco sondeos, el
   encogimiento con cinco, el suelo de 0,1 del factor y los cuatro partidos principales del rebaño.
4. (Cuatro días) Es una convención que no hemos medido; FiveThirtyEight usaba una ventana de catorce días.
5. (Combinación) Para que un puñado de sondeos casi idénticos no dé una certeza desmedida, σ_d nunca baja de la cuarta
   parte de σ_h: el ciclo pesa, como mucho, un 94 %. En el CIS y el PP actúa: su error del ciclo, 0,42, sube a 0,64.
6. (Error común) Probamos a restar a todas las encuestadoras su error medio histórico en cada partido. Empeoraba las
   probabilidades del modelo frente al resultado (la puntuación de Brier) un 21 %.
7. (Convenciones de los efectos) Son elecciones razonables que no hemos comparado con alternativas: los 90 días de
   historia, el mínimo de cinco sondeos en el ciclo, la elección imaginaria de diez sondeos, el 10 % anual, el medio
   peso de las autonómicas y las tres iteraciones.
8. (Referencias) W. S. Cleveland (1979), «Robust locally weighted regression and smoothing scatterplots», *Journal of
   the American Statistical Association*, 74; J. Fan e I. Gijbels (1996), *Local Polynomial Modelling and Its
   Applications*, Chapman & Hall; A. Gajewicz-Skretna, S. Kar, M. Piotrowska y J. Leszczynski (2021), «The
   kernel-weighted local polynomial regression (KwLPR) approach», *Journal of Cheminformatics*, 13.
9. (Corte del kernel) Los sondeos cuyo peso ωᵢ queda por debajo de 0,01 se dejan fuera del ajuste de ese día, para
   ahorrar cálculo; su aportación es despreciable.
10. (ISJ) S. J. Sheather y M. C. Jones (1991), «A reliable data-based bandwidth selection method for kernel density
    estimation», *JRSS B*, 53; Z. I. Botev, J. F. Grotowski y D. P. Kroese (2010), «Kernel density estimation via
    diffusion», *Annals of Statistics*, 38. El ISJ resuelve una ecuación de punto fijo sobre un histograma de 256
    celdas de las fechas, con una transformada de coseno; si no converge, y como mínimo, la anchura es 2π veces la
    separación media entre sondeos.
11. (Adaptativo) I. S. Abramson (1982), «On bandwidth variation in kernel estimates—a square root law», *Annals of
    Statistics*, 10. Solo la primera iteración sigue exactamente su regla; en las siguientes la anchura sigue
    ajustándose a la densidad ya suavizada, y en el final de la serie crece con cada iteración.
12. (Cobertura de la banda) La banda contiene el resultado en un 23 %, un 24 % y un 28 % de los casos leída a 6, 14 y
    30 días de la elección. El intervalo del voto estimado, que suma el error histórico de los sondeos, lo contiene en
    un 96 % de los casos a seis días, para un 95 % nominal.
13. (Límites de la banda) Tampoco recoge la incertidumbre de la anchura de la campana ni la de los efectos de
    encuestadora, y se queda corta cuando pocas encuestadoras pesan de verdad: remuestreando por encuestadoras (un
    bootstrap por clústeres), el error del PP sale de 0,38 puntos, frente a los 0,25 de la fórmula agrupada, con unas
    nueve encuestadoras efectivas.
14. (Escala del error) *e* es el error absoluto medio que predice la regresión, y lo usamos como escala del error del
    sorteo. La cobertura medida del intervalo, un 96 % para un 95 % nominal (nota 12), dice que esa escala es adecuada.

Todas las referencias se verifican (autores, año, título, revista) antes de publicarlas; las de esta lista se dieron de
memoria, salvo Gajewicz-Skretna et al., comprobada en Crossref.

## 7. Cambios de código

### 7.1 Notas desplegables (componente reutilizable)

- **Marcado.** Una macro en `_macros.html` para la llamada (p. ej. `m.note('id')`) que produce un número volado: un
  enlace `<a href="#nota-{id}">` a la nota del final (funciona sin JS), con `id` propio para volver. La numeración
  sale del orden de las llamadas en la página, no se escribe a mano. Una macro (o bloque) para la tarjeta «Notas»: una
  lista ordenada, visible, en la que cada nota tiene `id="nota-{id}"` y un enlace de vuelta a su llamada. El texto de
  cada nota se escribe una sola vez (en la lista final).
- **Comportamiento con JS** (módulo pequeño compartido en `web/dist/js/`, cargado por `metodo.js` o por la página):
  al pulsar la llamada se abre, justo después del bloque que la contiene (el párrafo o el elemento de lista), un
  recuadro con una copia del texto de la nota y una «×» que lo cierra; pulsar otra vez la llamada lo cierra; Esc
  cierra el recuadro con el foco y devuelve el foco a la llamada. Cada nota se abre y cierra por separado (abrir una no
  cierra las demás). La llamada se comporta como un botón con `aria-expanded` y `aria-controls`. Sin JS, el enlace
  salta a la nota de abajo, que sigue siempre visible.
- **Estilo** en `site.css`, coherente con las tarjetas (gris del sitio, sin sombras nuevas); funciona en móvil.
- Tests: la macro produce enlaces e ids enlazados en los dos sentidos y una numeración consecutiva; la tarjeta
  «Notas» lista todas las notas; un test del módulo JS con el patrón de `tests/test_js_*.py` (abrir, cerrar con la
  «×», con Esc, dos notas a la vez).

### 7.2 Plantilla `metodo_promedio.html`

- Sustituir el texto por el de la sección 5, con las subsecciones h3 y sus ids (sección 4), las tablas fechadas (clase
  de tabla del sitio, `table-wrap`) y las fórmulas en MathML con `aria-label` en palabras.
- Índice con las tarjetas 2 a 10.
- Las cifras de este ciclo en «Qué sondeos entran» ({n_polls}, {n_pollsters}) son vivas: salen del run nacional
  publicado (sondeos y encuestadoras del ciclo que entran en el promedio); si no hay run, la frase se omite.
- Firma tras la tarjeta «Notas» y antes del cierre, con `author_url` y `author_handle` del contexto común.

### 7.3 Tarjeta del ejemplo del PP

- Dos fechas: `example.published` (el cálculo) y `example.date` (la lectura), con la frase condicional de la
  sección 5.8. Corrige el error actual («El {published} el promedio se establece en…» atribuía la cifra al día del
  cálculo).
- Columnas de la tabla: Fecha, Casa, Entrevistas, Publicada, Corregida (`pct_adj`, ya en los datos), Peso,
  Cercanía, Solape, Muestra, Rating. «Casa» se mantiene (lo pidió Luis el 2026-10-16) con su excepción
  `WEIGHTS_HOUSE_HEADER` en `assert_public_wording`. La columna de la muestra publicada pasa a «Entrevistas» para no
  repetir «Muestra», que es el factor. El texto de 5.8 se ajusta: «las entrevistas publicadas». Los
  `title` de cada cabecera dicen lo mismo que el texto. El gráfico (`metodo.js`) no cambia.
- `weights_example()` añade `top`: la fila de mayor `weight` del líder (encuestadora, medio, fecha, peso), `None` sin
  filas. Docstring y test.

### 7.4 Tests

- Actualizar `tests/test_pages_ejemplo.py` y `tests/test_pages.py` a los textos nuevos (hoy algunos comprueban frases
  literales de la página). `test_promedio_page_shows_the_example_block` ya falla en `dev` antes de este trabajo
  (espera «En el gráfico, cada punto…» antes de la lista plegada): se reescribe con la tarjeta nueva.
- Comprobar `assert_public_wording` en la página y que el índice, los ids y el orden de las tarjetas son los de la
  sección 4.

## 8. Restricciones

- PEP8, docstrings en todas las funciones, identificadores en inglés neutro; los textos públicos, en español.
- Luis hace commits en paralelo con `git commit -a` en el mismo árbol: revisar `git status`/`git log` antes de cada
  commit, añadir solo los ficheros propios, no reescribir sus commits. Hoy hay cambios suyos sin commit en
  `web/dist/`, `web/templates/{eleccion,escanos,promedios}.html`, `tests/test_js_*.py`, `tests/test_pages.py` y
  `data/`. Decidido por Luis: `tests/test_pages.py` y `web/dist/css/site.css` se editan en su sitio sin tocar
  sus cambios, y **se dejan sin commit** (los incluye él en el suyo); todo lo demás se commitea con normalidad.
- No tocar la base de datos ni lanzar jobs (`compute`, `publish`): son de Luis.
- Tipografía española del sitio: comillas «», coma decimal, punto de miles, «%» separado por espacio fino como en el
  resto de la página (seguir lo que ya hace la plantilla).

### 8.1 Decisiones mías (Claude), para que Luis las revise

- **Fotos fechadas frente a cifras vivas.** Las tablas y cifras dentro de las secciones (dispersión, ratings, cuentas
  CIS/GAD3, efecto CIS-PP, campana, error *e* por tamaño, 19 % sin muestra, mediana 1.200, *ñ* ≈ 1.300) son una foto
  fechada del 20-21 de octubre de 2026, dicha en el pie. Lo vivo son las dos tarjetas de ejemplo, {n_polls} y
  {n_pollsters} y `n_sim`. Mantener vivas todas las cifras exigiría publicar partes nuevas del run.
- **Titular y «En resumen»** reescritos para que cuadren con el texto nuevo.
- **El error común probado (21 %)** pasa a nota y en el cuerpo queda la frase cualitativa.
- **El error *e* por tamaño** se da como media del último sondeo de cada encuestadora en las generales (consulta
  directa, sin ponderar), no como predicción del estimador del modelo: es lo que el texto dice que es.
- **El cálculo ilustrativo de la raíz** (1,5 puntos de error de método) queda en el cuerpo, presentado como
  ilustración.
- **La ilustración del rebaño** no menciona backtests; los umbrales van a nota.
- **«Por qué fiarnos del consenso»** (efectos de encuestadora) es un párrafo nuevo que no estaba en los borradores
  aprobados: sale de `docs/analisis-2026-09/metodo-6-house-effects.md` y justifica la corrección con una medición.
- **«2,4 puntos»** de la muestra es la «dispersión entre casas» de `metodo-6` (septiembre de 2026), con su misma
  palabra; no se precisa si es desviación típica o rango.

### 8.2 Cifras

La foto es el run nacional `site/v1/runs/es/20261020-180003` (lectura del 24-10-2026) y
`backtest/results/by_horizon.csv` del mismo día. El builder puede corregir cualquier cifra contra esos ficheros
publicados (no la base de datos) y debe hacerlo si no cuadra; si corrige una, lo anota en el commit. Las cifras
datadas en septiembre (anchura del PP de 61 a 130 días y 129 en la última fecha; tope de 4.715) vienen de
`docs/analisis-2026-09/` y se quedan con su fecha explícita en el texto («en septiembre»). La anchura «81 días» de la
regresión es la del día leído del run de la foto (81,2): el texto lo dice así («el 24 de octubre, en el cálculo del 20»),
para que no se confunda con la cifra viva de la tarjeta del ejemplo. La tabla de la campana usa h = 81,2 (180 días →
8,6 % → «9 %»).

## 9. Lo que se deja al builder

- El marcado exacto de las tablas, las fórmulas MathML y sus `aria-label`.
- El nombre de la macro y del módulo JS de notas, y su CSS.
- Cómo obtener {n_polls} y {n_pollsters} de las partes del run nacional que ya se publican (sin cambiar el run).
  Definición (decisión mía): {n_polls} son los sondeos del ciclo que pasan los filtros de inclusión de la sección 5.2
  (veda, agregadores, paneles, partidos), incluidos los que luego pesan 0 por solaparse, como el 328 de la foto;
  {n_pollsters}, las encuestadoras de esos sondeos.
- La redacción de los `title` de las cabeceras de la tabla del ejemplo.
- Pequeños retoques de puntuación del texto, sin cambiar su contenido ni sus cifras.

## 10. Pendiente para Luis después

- Lanzar `compute` con `overwrite` (recalcula `weight_over` con los grupos que suman 1, `2b3d13a`) y volver a publicar.
- Revisar el posible error de mapeo «GESOP / The Adelaide Review» en los sondeos de 2023.
