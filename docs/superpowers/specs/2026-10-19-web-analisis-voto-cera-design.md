# Web, Análisis: el voto CERA y la Ley de Nietos, anclado al 29-N

Fecha: 2026-10-19. Estado: **construida** el 19 de octubre por desarrollo dirigido por subagentes: tareas 1–11
en `dev` hasta `fefb3a9`; tareas 12–19 en la rama `analisis-voto-cera`, que sale de `fefb3a9` porque Luis tenía
trabajo sin confirmar en el mismo checkout. Suite completa: 1.096 tests en verde, 75 omitidos y 1 fallo,
`tests/test_pages_ejemplo.py::test_promedio_page_shows_the_example_block`, que ya fallaba en `dev` antes de este
trabajo y no es parte de él. Publicación local sin registro y paseo del HTML servido (sin navegador) hechos;
falta el paseo de Luis con datos reales (sección final). Antes: aprobada por Luis en el chat el 19 de octubre
(«Todo OK», con el umbral de acierto en más de 1 escaño), tras la revisión del constructor (siete preguntas, sección final). Maqueta aprobada en el chat el 19 de octubre
(`docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-mockup.html`, versión 4: tres preguntas con veredicto,
titular y subtítulo de la infografía, tres cifras, sin tarjeta de contexto). Sigue a la fase 9 y al lote de
tablas del 17 de octubre; esta spec solo describe lo que cambia respecto de lo que hay en `dev` (`669f909`,
817 tests).

## Intención

Luis publicó el 30 de junio de 2026 en Medium el artículo «El voto CERA en España y el efecto de la "Ley de
Nietos"» (41 minutos de lectura, 27 figuras y tablas), una infografía con su estructura y dos hilos en X: el del
artículo y la réplica al informe «hasta 23 escaños» de The Objective. Quiere que ese trabajo viva en la web, no como
copia del artículo sino como **página viva anclada a las generales del 29 de noviembre de 2026**: con el censo real
de la convocatoria en lugar de la proyección a 2027, con los datos del censo que el INE cierra cada mes, con la
predicción registrada antes de la elección y con su comprobación cuando se escrute el voto CERA. Es la primera pieza
de una sección nueva, «Análisis».

La página contesta, con datos oficiales, a la pregunta de la infografía: **¿es posible orquestar con éxito un
fraude electoral significativo a través del censo exterior?** Y lo hace en tres preguntas con veredicto, como la
infografía y el hilo:

1. ¿Se dispara el censo de los españoles en el exterior? **SÍ.**
2. ¿Sesgo de izquierdas o reparto en provincias «estratégicas»? **SIN EVIDENCIA.**
3. ¿Pueden alterar el reparto de escaños el 29-N? **NO.**

Más un bloque, «Lo que dice el debate», que replica con los mismos datos las cifras que circulan (23 escaños, 16
provincias, 162 pueblos, el «censo fabricado») y explica por qué no se sostienen, con los análisis de sensibilidad
completos; y una tarjeta de comprobación que, tras el escrutinio CERA, dirá qué pasó provincia a provincia.

## Lo que ha cambiado desde el artículo (contexto que la página incorpora)

- Las generales se convocaron el 6 de octubre de 2026 para el 29 de noviembre. El censo vigente es el cerrado el 1
  de agosto (art. 39.1 LOREG). El INE publica el censo de consulta de la convocatoria con los electores residentes
  en el extranjero por provincia y país (`/oficina_censo/elecgen2026/Elecgral_2026_cc_tab8.xlsx`, total
  2.769.645; por país, `tab6`; por comunidad, `tab7`). El censo cerrado a 1 de agosto era de 2.747.688 (2.328.261
  en las generales de 2023). La serie mensual del INE llega al 1 de septiembre de 2026 (2.762.829).
- El 10 de septiembre el Tribunal Supremo suspendió cautelarmente la inscripción en el CERA, y los efectos
  electorales de las inscripciones ya hechas, de quienes obtuvieron la nacionalidad por la Ley de Nietos acogiéndose
  a la presunción de exilio de la Instrucción de la DGSJFP (salidas de España entre 1936 y 1955), y exigió a la
  Oficina del Censo Electoral que señale individualmente la vía de acceso de cada inscrito. El 12 de octubre mantuvo
  la medida frente a la Abogacía del Estado (5 votos a 1). EFE cifra los afectados en 163.020; la Fiscalía y la
  Abogacía hablaban de unos 170.000 expedientes. Cabe recurso de amparo. Nadie publica todavía el reparto por
  provincia de esas inscripciones.
- El debate público: The Objective (4-9-2026) publicó el informe «Evolución del censo CER y CERA en España» de la
  consultora De Madrid a Europa («hasta 23 escaños»); El Español (12-7-2026), «16 provincias decisivas»; The
  Objective, los «162 pueblos con más electores fuera que dentro». Luis replicó el primero en un hilo (carpeta del
  artículo, `hilo-x-replica-ondarra.md`).

## Decisiones de Luis (2026-10-19)

1. Versión viva anclada al 29-N, no una copia fija del artículo.
2. El bloque del debate entra, «y todo lo necesario para convencer a todos de por qué no se sostiene, incluidos los
   análisis de sensibilidad».
3. La página se organiza como la infografía: tres preguntas con su veredicto, y contesta a las preguntas de los dos
   hilos de X.
4. Titular y subtítulo de la página, los de la infografía: «El voto CERA en España y el efecto de la "Ley de
   Nietos"» / «¿Es posible orquestar con éxito un fraude electoral significativo a través del censo exterior?».
5. Tres cifras arriba (censo del 29-N, inscripciones en litigio, votos esperados); las de escaños «se contestan más
   adelante».
6. Sin tarjeta «Qué ha pasado desde el artículo»: el artículo no se menciona hasta el final, en la fuente.
7. Datos, gráficos y contexto del artículo: carpeta `/Users/luiss/HD/Clientes/Luis Sancho/Artículos/Medium/El voto
   CERA en España y el efecto de la Ley de Nietos` (ficheros fuente en `cera/`, tablas derivadas en `data/`,
   figuras en `medium/img/` e `infografia/`). No hay código: el cálculo se reimplementa en el repo y las tablas
   derivadas sirven de prueba de regresión.

## Objetivos

1. Los datos de entrada del estudio, versionados en `data/cera/`, con un comando de carga que descarga del INE los
   ficheros que cambian (D1).
2. Un módulo de cálculo que reproduce el artículo (serie, crecimiento, participación por origen, voto, modelo de
   incremento, último escaño, sensibilidad) y añade lo nuevo (censo real del 29-N, escenario sin las inscripciones
   en litigio, comprobación con y sin CERA para cualquier elección con escrutinio CERA por partido), validado contra
   las tablas del artículo y contra el reparto oficial de 2023 (D2).
3. Una publicación nueva, los **estudios** (`site/v1/studies/{study}/`), con sus partes, schemas, gemelos CSV,
   manifiesto y contrato; el primer estudio es `voto-cera` (D3). Su API, por las rutas de siempre (D4).
4. La sección «Análisis» en la web: índice `/analisis` y la página `/analisis/voto-cera`, con la estructura de la
   maqueta, textos con cifras vivas, dos estados (antes y después del escrutinio) y la entrada en el menú y en
   Inicio (D5), con sus gráficos (D6) y sus textos (D7).
5. Parámetros editables a mano para lo que nadie publica todavía: la cifra de inscripciones en litigio y la fecha
   de la fuente (D8).

## No objetivos

- Ningún cambio en el modelo de encuestas, en el pronóstico ni en Escaños. El estudio es un cálculo aparte sobre
  resultados oficiales; no toca `Forecaster`, `Simulator` (salvo reutilizar su reparto D'Hondt) ni los runs.
- No se integra el efecto CERA en el pronóstico de escaños del modelo. Podrá hacerse después como bloque de
  Escaños; esta spec no lo contempla.
- No se reproduce el artículo entero: la página lleva lo que la maqueta muestra. El texto largo sigue en Medium,
  enlazado en «Datos y método».
- No se modela la decisión judicial (quién podrá votar al final): el escenario «sin las inscripciones en litigio» es
  un parámetro, no una predicción jurídica.
- Ningún selector de ámbito: el estudio es nacional. La página del estudio no lleva barra de evento ni modo.
- Nada de nginx, Docker, CSP ni modo oscuro.

## Anti-objetivos (lo que haría fracasar la fase aunque funcionara)

- Que las cifras de la página no sean las del cálculo: todo número de los textos sale de la parte publicada (como
  las cifras vivas de Método), nunca escrito a mano en la plantilla, salvo los hechos del contexto (fechas de los
  autos, nombres de los informes) que van en los textos.
- Que el reparto D'Hondt del estudio no reproduzca exactamente los 350 escaños de 2023 por provincia y partido: sin
  esa validación ninguna cifra de escaños vale.
- Que el estudio no reproduzca el artículo con sus mismos datos: el crecimiento de 422.253 electores entre octubre
  de 2022 y mayo de 2026, el ritmo ×2,2 de los dos periodos de 43 meses, la brecha de las diez circunscripciones
  más ajustadas (Girona 284, Cantabria 403, Tarragona 1.247, Madrid 1.339…), los cero cambios del escenario central
  y las rejillas de sensibilidad (tablas 9, 11 y 14). Donde difiera por una decisión de esta spec (censo real en
  lugar de proyección, pool de colocación observado en lugar de proyectado), la diferencia está explicada aquí.
- Que la predicción cambie después de registrada sin dejar rastro (D3, registro).
- Perder la lectura sin JavaScript: textos, veredictos y tablas salen del servidor; solo los gráficos los dibuja el
  navegador, como en el resto del sitio.
- Que la página mencione el artículo antes de «Datos y método» (decisión 6).

## D1. Los datos de entrada: `data/cera/`

Todo lo que el estudio lee vive en `data/cera/`, versionado, con su fila en la tabla de `data/README.md`:

| Fichero | Contenido | Origen |
|---|---|---|
| `ine/cerapropais_{yy}.xlsx` (2015 en adelante) y `.xls` (2006–2014) | Censo CERA cerrado el día 1 de cada mes, por provincia de inscripción y país de residencia; un libro por año con una columna por mes | INE, `https://www.ine.es/oficina_censo/censo_cerrado/cerapropais_{yy}.xlsx`; los de la carpeta del artículo son copias idénticas |
| `ine/elecgen2026_tab6.xlsx`, `tab7`, `tab8` | Censo de consulta de la convocatoria del 29-N: electores residentes en el extranjero por país (total nacional), por comunidad y por provincia × país | INE, `https://www.ine.es/oficina_censo/elecgen2026/Elecgral_2026_cc_tab{n}.xlsx` |
| `escrutinio/es-2019-11-10.csv`, `es-2023-07-23.csv` | Voto CERA por provincia (y comunidad) y candidatura: electores CERA, votantes, nulos, blancos y votos por candidatura, con `region_id` | Ministerio del Interior (ficheros «Resultados voto CERA» de cada elección; en la carpeta del artículo, `data/cera_generales_20xx_partido.csv` y los PDF de `cera/`) |
| `escrutinio/es-2026-11-29.csv` | El mismo fichero del 29-N, cuando Interior lo publique tras el escrutinio general (Luis lo añade a mano) | Ministerio del Interior |
| `autonomicas.csv` | Por proceso autonómico con escrutinio CERA por partido (CyL 2022 y 2026, Andalucía 2022 y 2026, Aragón 2023 y 2026): censo CERA, votantes, votos a PP, PSOE, VOX y otras izquierdas; y los bloques izquierda / derecha / separatista-regionalista | Juntas electorales (en la carpeta, `data/comparativa_autonomicas_cera.csv` y `cyl_cera_bloques.csv`, más los PDF de `cera/`) |
| `consulados/{proceso}.csv` | Censo y votantes por país o consulado: CyL 2026, Extremadura 2023 y 2025, Aragón 2026 | Juntas electorales (en la carpeta, `data/cyl2026_pais.csv`, `extr_2023_pais.csv`, `extr_2025_pais.csv`; Aragón, de `cera/aragon-2026-resumen-resultados.pdf`, tabla 8 del artículo) |
| `countries.csv` | Catálogo de países del INE: código, nombre, región (`latam`, `europe`, `usa_canada`, `other`) y exposición a la Ley de Nietos (`alta`, `media`, `baja`, según la clasificación ex-ante del artículo, tabla 2) | Curación propia |
| `params.json` | Parámetros del estudio (D8) | Curación propia |

Los resultados provinciales de las generales (2019, 2023 y, cuando lleguen, 2026-11) se leen de los ficheros
Infoelectoral que ya están en `data/infoelectoral/es/` (`PROV_02_YYYYMM_1.xlsx`), **con todas las candidaturas**,
no con el mapa a siglas del modelo (el cajón «otros» agregado rompería el reparto). Las columnas «Censo CERA» y
«Total votantes CERA» de esos mismos ficheros dan la participación CERA en cada generales desde 1993 (figura 3
del artículo); el constructor comprueba en qué ficheros antiguos existen y lo documenta.

**Carga.** Un comando nuevo en `load/` (junto a `run_load.py`, `--what cera`) descarga del INE el libro mensual
del año en curso y las tablas del censo de la convocatoria, los guarda en `data/cera/ine/` y dice qué meses nuevos
trae. Los libros de años cerrados no se vuelven a bajar. Es un paso de carga que Luis ejecuta antes de publicar,
como las demás cargas; `publish` no descarga nada.

**Lectura.** `mtpy/lib/studies/cera/data.py` lee cada fuente a un `DataFrame` largo con códigos INE de provincia
(`region_id`, los de `es-provinces.csv`) y códigos de país del INE; el libro `.xls` antiguo (dos filas de cabecera,
«Censo cerrado» / «a 1-12-2006», guiones por cero) y el `.xlsx` moderno (una fila, «Censo cerrado a 1-may-2026») se
normalizan al mismo formato. La serie nacional debe dar 245 observaciones entre enero de 2006 y mayo de 2026, y
1.160.209 en diciembre de 2006, 2.708.083 en mayo de 2026 y 2.762.829 en septiembre de 2026.

## D2. El cálculo: `mtpy/lib/studies/cera/`

Un paquete nuevo con funciones puras sobre `DataFrame`, sin base de datos (los resultados provinciales salen de los
XLSX de `data/`), con docstrings al estilo del repo. Lo que calcula, en el orden de la página:

**Serie y crecimiento (pregunta 1).**

- Serie mensual nacional y por provincia; marcas: Ley 20/2022 (2022-10-21), generales 2023 (2023-07-23), fin del
  plazo (2025-10-22), censo del 29-N (2026-08-01, el cierre del censo vigente).
- Altas netas en dos periodos de 43 meses: marzo de 2019 → octubre de 2022 y octubre de 2022 → mayo de 2026 (los
  del artículo, para la regresión), y los mismos periodos recalculados con el último cierre disponible como fin del
  segundo (lo que la página muestra, con la duración del primero igualada). Por provincia, por país y por origen
  (`latam` / resto) y por exposición. Ritmo = altas del segundo / altas del primero.
- Crecimiento desde las generales de 2023 hasta el censo de la convocatoria, por provincia y por país (tablas de
  la pregunta 1): el censo del 29-N es el de las tablas del INE de la convocatoria (`tab8`), no el cierre mensual.
- La regresión de ruptura de tendencia del artículo (altas netas mensuales sobre tendencia, ruptura en octubre de
  2022 e interacción, Newey-West con 12 retardos) se calcula y se publica (`p` del cambio de pendiente y del nivel),
  pero la página solo la cita en «Cómo leerlo».

**Participación (pregunta 2).**

- Participación por origen en cada proceso con datos por consulado (Europa, EE. UU. y Canadá, Latinoamérica, resto),
  agregando los consulados por país con `countries.csv`.
- Participación CERA en las generales desde 1993 (votantes CERA / censo CERA de los ficheros Infoelectoral), con el
  periodo: antes del voto rogado, voto rogado (2011–2019), sin voto rogado (2023 en adelante).
- Participación por proceso autonómico (anterior y último) de `autonomicas.csv`.

**Voto (pregunta 2).**

- Cuota de la izquierda sobre izquierda + derecha por proceso (generales 2019 y 2023; las tres comunidades, proceso
  anterior y último) con los bloques del apartado 3.2 del artículo; margen izquierda − derecha dentro del CERA;
  cambio de la cuota del PSOE y de las otras izquierdas (la canibalización); prima exterior PSOE − PP frente al
  total de cada elección.
- Ratios de representación en las generales de 2023: cuota de cada partido en el voto CERA / cuota en el total
  nacional (SUMAR 120 %, PP 99 %, PSOE 92 %, VOX 90 %).
- Inferencia ecológica: por provincia, peso latinoamericano del censo CERA (cruce provincia × país del cierre de
  julio de 2023) frente a cuota de izquierda del voto CERA provincial de 2023, sobre **todas las provincias con un
  mínimo de votantes CERA** (`params.ecological_min_voters` = 200, que solo deja fuera Ceuta y Melilla): `r`,
  pendiente y cuartiles, los que salgan. El artículo daba −0,77 sobre 43 provincias, pero las nueve excluidas
  (Ceuta, Melilla, La Rioja, Navarra, Baleares, Murcia, Las Palmas, A Coruña y Madrid) no responden a ningún umbral
  sino, con toda probabilidad, a un fallo de casado de nombres en el cálculo original (decisión de Luis: calcular
  sobre todas y publicar lo que salga; «Cómo leerlo» lo explica).

**Colocación (pregunta 2).**

- Cuota de las provincias sensibles (magnitud ≤ 5 escaños) en el censo de Argentina + Cuba + Venezuela antes de
  la ley y en las altas posteriores (24,0 % → 17,3 % en el artículo; en la página, con el censo del 29-N como fin).
- Aceleración de las altas por provincia (ritmo de los dos periodos) frente a magnitud, con el volumen absoluto;
  correlación del crecimiento con el tamaño de la circunscripción y con la brecha del último escaño.
- El «imán de Madrid»: cuota de Madrid en el stock de 2022 y en las altas de cada periodo para los seis países
  (Argentina, México, Cuba, Venezuela, Uruguay, Chile) y para México y Venezuela por separado.

**Reparto D'Hondt.** Sobre los resultados provinciales de 2023 con todas las candidaturas, umbral del 3 % de los
votos válidos (candidaturas más blancos) por circunscripción, con `Simulator.alloc_seats` o un equivalente propio
si aquel no admite el formato; el constructor elige, pero la validación es obligatoria: **350 de 350 escaños
iguales a los oficiales por provincia y candidatura**, como test.

**El modelo de incremento (pregunta 3).** Igual que el apartado 3.3 del artículo, con el censo real:

1. Electores nuevos por provincia y origen = censo de la convocatoria (`tab8`, provincia × país) menos censo de las
   generales de 2023 (cierre de julio de 2023 del libro mensual, o el fichero de escrutinio de 2023, que trae los
   electores CERA por provincia; el constructor usa el que cuadre con 2.328.261 y lo documenta).
2. Votos nuevos = electores nuevos × participación por origen (`params.json`: 5 % latinoamericano, 10 % resto, con
   su derivación en D8) × fracción de voto válido (99 %).
3. Reparto por candidatura: el del escrutinio CERA de 2023 de esa provincia (votos a candidaturas).
4. Resultado = resultados de 2023 + votos nuevos; D'Hondt provincia a provincia; cambios por candidatura y por
   bloque frente a 2023.

Dos escenarios de censo, los dos publicados:

- **Censo de la convocatoria**: el de `tab8` tal cual.
- **Sin las inscripciones en litigio**: se restan `params.litigation.count` electores, repartidos por provincia en
  proporción al crecimiento del censo de origen latinoamericano entre octubre de 2022 y la convocatoria (decisión
  mía: es el segmento al que afecta la cautelar y nadie publica el reparto real). Cuando la Oficina del Censo
  publique la cifra o el reparto, se cambia el parámetro.

**El último escaño.** Para cada circunscripción, con el reparto de 2023: quién obtuvo el último escaño, quién
quedó primero excluido y la brecha (votos que le faltaron al excluido para arrebatarlo, con los cocientes D'Hondt
exactos). Efecto neto del CERA = votos nuevos del aspirante − votos nuevos de quien defiende el escaño (la
definición del artículo, tabla 10). Y la prueba real: si el reparto con el incremento cambia en esa provincia. Se
publican las 52 filas y la participación que haría falta para cerrar la brecha (la de Cantabria, «duplicarse»).

**Sensibilidad (pregunta 3).** Tres rejillas, calculadas, no copiadas:

- *Participación × voto a la izquierda* (tabla 11): se recalcula el voto exterior completo sobre el voto peninsular
  de 2023 (resultados de 2023 menos el voto CERA de 2023 por candidatura, para no contar dos veces): censo de la
  convocatoria × participación {9 %, 20 %, 30 %, 50 %, 70 %} × fracción al bloque de izquierda {50, 60, 70, 80 %},
  repartida dentro de cada bloque como el CERA de 2023 de la provincia; cambio neto de escaños de la izquierda
  (PSOE + SUMAR). La celda 9 % / 50 % reproduce el «+1» del artículo, artefacto del reparto uniforme, y la nota lo
  dice.
- *Censo × participación × sesgo* (tabla 9): el modelo de incremento con censo {real (×1,19), ×1,5, ×2,0 sobre
  2023}, participación {observada, ×2, ×3} y sesgo del voto nuevo {empírico, 70 %, 90 % a la izquierda}; cambio de
  escaños de izquierda y derecha. Las filas de la tabla del artículo son el subconjunto que se muestra; la parte lleva
  la rejilla completa.
- *Colocación óptima* (tabla 14): el límite superior. Pool = altas netas de Cuba, Venezuela, Argentina y México
  entre octubre de 2022 y el censo de la convocatoria (**decisión mía**: en el artículo el pool eran las altas
  proyectadas de 2026 a 2027, unas 62.800; con el censo ya cerrado, lo que se puede colocar es lo que entró con la
  ley, y la pregunta pasa a ser «¿y si esas altas se hubieran colocado a dedo?»). Las altas se retiran de sus
  provincias reales y se asignan una a una a la circunscripción donde el PSOE necesita menos votos para el
  siguiente escaño (coste marginal D'Hondt recalculado tras cada asignación, en bloques del tamaño que el
  constructor decida para que el cálculo sea razonable), con participación {9, 20, 50, 100 %} y voto al PSOE
  {50, 60, 70, 80, 100 %}; escaños que arrancaría el PSOE. Y la celda de contraste: el mismo pool sin colocación
  dirigida (en sus provincias reales) a participación y reparto observados, que debe dar cero.

**Comprobación (con y sin CERA).** Para cada elección general con fichero de escrutinio CERA por candidatura (2019,
2023 y, cuando exista, 2026-11-29): D'Hondt con el resultado oficial y con el oficial menos el voto CERA por
candidatura; provincias donde cambia algún escaño (quién lo gana con CERA y sin CERA, y si el cambio cruza de
bloque); **saldo neto de escaños que el voto CERA da a cada partido y a cada bloque** (con CERA menos sin CERA);
provincias donde los votantes CERA superan la brecha del último escaño del reparto oficial (el criterio del informe
de los 23 escaños: en 2023, nueve); votos CERA y participación. Es la tabla de la tarjeta «Comprobación» y la de
«2023 ya pasó la prueba» del debate. Los precedentes, recalculados por el revisor con los ficheros reales: en 2019
el voto CERA movió el último escaño de Bizkaia (PP frente a PNV) y en 2023 el de Madrid (con CERA, PP 16 y PSOE 10;
sin CERA, PP 15 y PSOE 11): saldo del PSOE −1, de la izquierda −1. Son tests.

**El criterio de acierto (decisión de Luis, 2026-10-19).** La hipótesis acierta si, tras el recuento del voto
CERA del 29-N, **no hay un movimiento de escaños significativo a favor del PSOE**. Se mide con la prueba con y sin
CERA: el saldo neto del PSOE y el del bloque de izquierda. «Significativo» es más de 1 escaño (decisión de Luis,
2026-10-19); el umbral es un parámetro (`params.check.max_gain`, 1). Un saldo de 0, negativo (como en 2023) o de
1 escaño es acierto; de 2 o más a favor del PSOE o de la izquierda, fallo. La tarjeta dice siempre el saldo exacto y en qué provincias, con el
precedente de 2019 y 2023 al lado.

**La réplica de los 23 escaños.** Se calcula lo que el informe calcula, para enseñarlo: censo de la convocatoria ×
participación media de cada provincia desde 1986 y × máxima provincial, contra la brecha de 2023; número de
provincias que «cuentan» con cada criterio (el hilo obtuvo 19 y 27 con el censo de julio).

## D3. La publicación: los estudios

Una clase nueva de publicación en `mtpy/lib/bundle.py`, al lado de `analysis`, `backtest` y `events`:

- Prefijo `site/v1/studies/{study}/`, con `{study}` un slug (`ROUTE_ALIAS_RE`); `STUDIES = ('voto-cera',)`.
- `STUDY_PARTS = ('meta', 'census', 'participation', 'vote', 'placement', 'prediction', 'sensitivity', 'check')`,
  schemas `study-{part}` en `SCHEMAS`, y gemelos CSV (`STUDY_CSV`) para lo tabular: `census-series`,
  `census-provinces`, `census-countries`, `participation`, `vote`, `ecological`, `last-seat`, `sensitivity-grid`,
  `sensitivity-scenarios`, `placement-grid`, `check`.
- `path_study(study, part)` y `path_study_csv(study, name)`, como `path_analysis`.
- Sobrescritura en cada publicación (como `analysis`), `meta` el último. El manifiesto gana `studies:
  {'voto-cera': {'published_at', 'registered_at', 'checked': bool}}`.

Contenido de las partes (los nombres exactos los fija el constructor; el contrato los documenta):

| Parte | Contenido |
|---|---|
| `meta` | `study`, `title`, `subtitle`, `election` (`scope`, `date`, `census_closed`, `census_total`, `census_source`), `registered_at`, `registered_commit`, `checked_at` (null hasta que exista el escrutinio del 29-N), `params` (los de D8 con su fecha y fuente), `sources` (lista de fuentes con URL para «Datos y método»), `series_last` (último cierre mensual), `regression` (`p_slope`, `p_level`), `commit`, `dirty`, `versions`, `seconds` |
| `census` | serie nacional (`dates`, `values`) y marcas; dos periodos por provincia, por país, por origen y por exposición (ambas versiones: la del artículo y la actualizada); crecimiento 2023 → convocatoria por provincia y por país |
| `participation` | por origen y proceso; generales desde 1993 con periodo; autonómicas anterior/último |
| `vote` | cuotas por proceso, márgenes, canibalización, prima exterior, ratios 2023, filas de la ecológica con `r`, pendiente y cuartiles |
| `placement` | sensibles stock/altas, aceleración por provincia con magnitud y volumen, correlaciones, imán de Madrid |
| `prediction` | por escenario (`convocatoria`, `sin_litigio`): censo, electores nuevos, votos nuevos, escaños por candidatura y bloque (2023 y proyección), cambios; las 52 filas del último escaño; la réplica de los 23 escaños |
| `sensitivity` | las tres rejillas con sus ejes y la celda de contraste |
| `check` | por elección con escrutinio: votos CERA, participación, provincias sobre la brecha, escaños decididos (lista) y cambios de bloque; las filas por provincia de la última elección disponible |

**Registro de la predicción.** Lo que se registra es la afirmación comprobable: «el voto CERA del 29-N no dará al
PSOE ni a la izquierda un saldo neto de más de `max_gain` escaños» (D2, criterio de acierto), junto con los
escaños proyectados por partido y bloque de los dos escenarios del modelo de incremento (cero cambios), que son su
fundamento. `Publish.run(what='study', study='voto-cera', register=True)` escribe `registered_at` (fecha ISO),
`registered_commit` y `registered` (la afirmación con su umbral y los escaños proyectados); exige el árbol limpio
(`dirty` falso) y falla si ya había registro, salvo `force`, que escribe `registered_changed_at`. Las publicaciones
sin `register` conservan el registro anterior leyendo el `meta` previo. La página muestra «Predicción registrada el
{fecha}» en la cabecera y en la tarjeta de comprobación; sin registro, la predicción se presenta como borrador y
no hay píldora.

**Job.** `Publish.run(what='study', study='voto-cera', ...)`: calcula, escribe las partes y actualiza el
manifiesto, en `mtpy/lib/publish/studies.py` (`publish_study`), con las mismas convenciones de `publish_analysis`
(tiempos por paso, `NothingToPublish`, `dry_run`). Queda fuera de la guardia LOREG: no publica sondeos (decisión
mía; el aviso de congelación del manifiesto se muestra igual en la página, como en todas).

`docs/web/contrato.md` gana la sección «Estudios» (árbol, partes, CSV, rutas) y la lista de kinds de `api_url`.

## D4. La API

En `mtpy/lib/webapi/__init__.py`, dos rutas antes de las de `forecast`:

- `/api/v1/studies` → lista de estudios publicados (`meta` resumido de cada uno).
- `/api/v1/studies/{study}/{part}` → la parte en JSON o, con `?format=csv` y `table=` cuando la parte tiene más
  de un gemelo, el CSV, como `Analysis.part_action`. Controlador `Studies`, `check_study` en `checks.py`
  (`STUDIES` de `bundle`), `Site.study(study, part)` y `Site.study_csv(study, name)` en `published.py` (mutables,
  como `analysis`). `api_url` gana `kind='study'`.

## D5. Las páginas

### Rutas, menú, Inicio

- `pages.ROUTES` gana `('/analisis', 'analisis', 'index')` y `('/analisis/{study}', 'analisis', 'study')`;
  controlador `Analisis` (`mtpy/controllers/Analisis.py`) sobre `Page`, contextos en `mtpy/lib/pages/analisis.py`.
  Un estudio desconocido o no publicado es 404 por la vía de siempre.
- `PAGES` (menú) gana `{'href': '/analisis', 'label': 'Análisis'}` entre «Ranking Encuestadoras» y «Método». La
  barra de pestañas de móvil (`TABBAR`, cinco entradas) **no cambia** (decisión mía: seis no caben; en móvil la
  sección se alcanza desde Inicio y desde los enlaces internos). `active` de la página del estudio es `/analisis`.
- Inicio gana una banda «Análisis» (`.sec-h` con enlace «Ver los análisis ›») debajo de «Elecciones pasadas», con
  una tarjeta-enlace por estudio publicado: título, subtítulo y los tres veredictos como píldoras de color. Sin
  estudios publicados, la banda no sale.

### `/analisis` (índice)

Titular «Análisis» y una línea («Preguntas del debate electoral contestadas con datos oficiales, comprobables
después de cada elección»); una tarjeta por estudio con lo mismo que Inicio más las fechas (registrada el,
comprobada el). Hoy, una.

### `/analisis/voto-cera`

La maqueta v4, de arriba abajo. Todo lo tabular y los textos salen del servidor; la isla `#initial-data` lleva solo
lo que los gráficos usan. Las tarjetas usan `m.card_start` y el «Cómo leerlo» (`m.how()`) de siempre; las tres
tarjetas de pregunta llevan además una clase `qcard` con la regla de color, el número y la píldora del veredicto
(estilos nuevos en `site.css`, con los colores de la infografía: verde `#2e9e5b`, naranja `#e8912b`, rojo
`#ce3b3b`, solo en estas tarjetas).

1. **Cabecera del estudio.** Píldoras: «Análisis», «Generales del 29 de noviembre de 2026» y el estado
   («Predicción registrada · se comprueba tras el escrutinio CERA» / «Comprobada con el escrutinio del {fecha}»).
   Titular `h1` grande (Outfit 800, 36 px en escritorio, 26 en móvil) y subtítulo (Outfit 500, 18 px): los de la
   decisión 4, sin más texto.
2. **Índice de secciones** (`nav.section-nav.toc`): Resumen · 1 · 2 · 3 · Lo que dice el debate · Comprobación ·
   Datos y método; las tres preguntas con el color de su veredicto.
3. **Tres cifras** (`m.kpis`): censo CERA del 29-N (con la variación frente a 2023 en flecha `.dlt`), inscripciones
   en litigio (con la fecha de la cautelar y el censo vigente en el pie) y votos CERA esperados (con las tasas por
   origen y los votos de 2023).
4. **Tres preguntas, tres respuestas** (`#resumen`): las tres tarjetas-enlace con pregunta, veredicto y una frase,
   que llevan a `#p1`, `#p2`, `#p3`; debajo, la nota de cierre del hilo («El censo exterior crece y merece
   escrutinio… los datos por encima de bandos y consignas»).
5. **Pregunta 1** (`#p1`, verde, SÍ): serie mensual con marcas; altas en dos periodos apiladas por origen con el
   ritmo rotulado; tabla por país (altas antes, después, ritmo) y tabla por provincia (censo de octubre de 2022,
   censo del 29-N, altas, %) con «Ver las 52 provincias»; frase-resumen.
6. **Pregunta 2** (`#p2`, naranja, SIN EVIDENCIA): participación por origen (barras por proceso); ecológica
   (dispersión con recta y `r`); banda de texto («¿Y la ventaja de la izquierda fuera no ha crecido…? No.»);
   margen de bloque, comicio anterior → 2026 (pendientes); canibalización (barras PSOE / otras izquierdas, el
   cambio del bloque en el subtítulo); provincias sensibles (apilado 100 %); tabla del imán de Madrid con su nota;
   frase-resumen.
7. **Pregunta 3** (`#p3`, rojo, NO): selector de escenario (`fieldset.seg`: «Censo de la convocatoria» / «Sin las
   inscripciones en litigio»; cambia la tabla y el gráfico de brecha desde la isla, sin recarga; sin JavaScript se
   ve el primero) y tabla 2023 → 29-N (censo, votos, escaños por bloque y otros, cambios) con la nota del escenario
   alternativo; brecha frente a efecto neto (las 10 más ajustadas, con «Ver las 52»); subtítulo «¿Qué haría falta
   para que importara?» con el radar (realidad hoy frente a todo al máximo) y la rejilla participación × voto a la
   izquierda; debajo, las tres dimensiones y la colocación óptima; frase-resumen.
8. **Lo que dice el debate** (`#debate`): ficha de los 23 escaños (qué calcula, replicado con cifras vivas, tres
   problemas y veredicto) junto a la tabla «2023 ya pasó la prueba» (las provincias donde el voto CERA real superó la
   brecha, con la columna «¿Cambió?» que sale del cálculo con y sin CERA: en 2023, «Sí, al PP» en Madrid y «No» en
   las otras ocho); el veredicto de la ficha lo dice tal cual (decisión de Luis): **«con el criterio del informe, en
   2023 había 9 escaños en juego; cambió uno, en Madrid, y en contra de la hipótesis»**. Debajo, tres fichas: 16
   provincias, 162 pueblos y «censo fabricado», esta última cerrando con el punto legítimo de la adscripción. Las
   fichas son `article.claim` con `dl`; sus textos son los del hilo de réplica, con las cifras vivas donde las hay
   (número de provincias con cada criterio, votos con cada participación, el escaño de Madrid).
9. **Comprobación** (`#comprobacion`): la pregunta es «¿decidió el voto CERA algún escaño, y a favor de quién?».
   Antes del escrutinio, la tabla por elección (2019, 2023 y el 29-N) con votos CERA, participación, provincias sobre
   la brecha, escaños que decidió el CERA, dónde, y el saldo del PSOE y de la izquierda; la fila del 29-N lleva la
   afirmación registrada («saldo del PSOE y de la izquierda no mayor que {max_gain}») y la fecha del registro, o
   «borrador» si no lo hay. Después del escrutinio, la fila se rellena, la tabla por provincia del 29-N (último
   escaño sin CERA / con CERA, brecha, votos CERA, ¿decidió?; 20 filas y «Ver más») aparece debajo, y la nota dice
   acierto o fallo según el criterio de D2, con el saldo exacto y las provincias, siempre con 2019 y 2023 como
   precedente (texto condicionado por los datos, no por una bandera a mano). La tabla de la maqueta («predicción:
   8–10 provincias sobre la brecha») queda sustituida por esta.
10. **Datos y método** (`#datos`): fuentes con enlace (las de `meta.sources`), las descargas CSV
    (`api_url(..., kind='study', fmt='csv')`) y, al final, el enlace al artículo de Medium y la fecha de publicación,
    único lugar donde se menciona.

El pie de la página es el del sitio. En móvil, cada rejilla de dos o tres columnas pasa a una; tablas con
desplazamiento horizontal y primera columna fija, como las de la fase 8.

## D6. Los gráficos

Módulos sobre `charts/base.js`, con `THEME`, tooltips y colores del sitio; los nuevos exportan una función
`*Option(data, opts)` comprobable con el arnés de `tests/test_js_charts.py` y un `render*`:

| Gráfico | Módulo |
|---|---|
| Serie mensual con marcas verticales rotuladas | `charts/line.js` si admite `markLine` con etiqueta; si no, `charts/census.js` nuevo |
| Altas en dos periodos apiladas por origen, con el ritmo rotulado | `charts/stacked.js` (existente) |
| Participación por origen y proceso; canibalización | `charts/bars.js` (existente, barras agrupadas con etiqueta) |
| Ecológica: dispersión con recta de regresión y etiquetas de algunas provincias | `charts/scatter.js` nuevo |
| Margen de bloque anterior → 2026 (pendientes con etiqueta y zona negativa sombreada) | `charts/slope.js` nuevo |
| Provincias sensibles (apilado al 100 %) | `charts/stacked.js` |
| Brecha frente a efecto neto (barras horizontales, dos series, neto en azul o rojo según el signo) | `charts/diverging.js` o `bars.js`, lo que el constructor vea; leyenda abajo y altura suficiente para diez filas |
| Radar de tres ejes (realidad hoy / todo al máximo) | `charts/radar.js` nuevo |

Las rejillas de sensibilidad son tablas HTML con sombreado por valor (el filtro `shade_around` del lote del 17 de
octubre o uno nuevo `shade_scale` de 0 al máximo, en verde); no son mapas de calor ECharts.

`pages/analisis-cera.js` monta los gráficos desde la isla y cablea el selector de escenario y los «Ver más».

## D7. Los textos

- El tono es el de la infografía y del hilo: afirmativo, corto, cada frase con un dato. Se escriben en las
  plantillas con las cifras interpoladas desde las partes (`{{ figures.growth_pct|pct }}`); las frases-resumen de
  cada pregunta y las fichas del debate parten literalmente de la infografía (`infografia/prompt-infografia.md`,
  apartado 6) y de los hilos, adaptadas al 29-N.
- Los hechos del contexto que no salen de los datos (fechas y contenido de los autos del Supremo, nombres y fechas
  de los informes, la cifra de EFE) van en los textos o en `params.json` con su fuente.
- Veredictos: «SÍ», «SIN EVIDENCIA», «NO» (la infografía dice «N/D»; en la web se lee solo y Luis no objetó).
- «Cómo leerlo» en cada tarjeta, con lo que la figura correspondiente del artículo explica (periodos de 43 meses,
  falacia ecológica, qué es la brecha y el efecto neto, por qué la celda observada de la rejilla da +1, qué es la
  colocación óptima y por qué es un límite, qué mide la prueba con y sin CERA).
- Glosario (`_glosario.html`): CERA, voto rogado, Ley de Nietos, censo de la convocatoria, brecha del último
  escaño, efecto neto, inferencia ecológica, provincias sensibles, colocación óptima.
- La frase de cierre «Los datos por encima de bandos y consignas» va en la nota del resumen; el pie del sitio no
  cambia.

## D8. Parámetros (`data/cera/params.json`)

```
{
  "election": {"scope": "es", "date": "2026-11-29", "census_closed": "2026-08-01",
               "census_tables": ["ine/elecgen2026_tab6.xlsx", "ine/elecgen2026_tab7.xlsx", "ine/elecgen2026_tab8.xlsx"],
               "baseline": "2023-07-23"},
  "litigation": {"count": 163020, "as_of": "2026-10-12", "source": "EFE, vía Maldita.es 13-10-2026",
                 "note": "Inscripciones con efectos electorales suspendidos por los autos del TS de 10-09 y 12-10-2026"},
  "participation": {"latam": 0.05, "rest": 0.10, "valid": 0.99,
                    "derived_from": ["consulados/cyl-2026.csv", "consulados/extremadura-2025.csv", "consulados/aragon-2026.csv"]},
  "periods": {"article": [["2019-03-01", "2022-10-01"], ["2022-10-01", "2026-05-01"]]},
  "sensitive_magnitude": 5,
  "ecological_min_voters": 200,
  "check": {"max_gain": 1},
  "placement_countries": ["ARGENTINA", "MÉXICO", "CUBA", "VENEZUELA"],
  "madrid_countries": ["ARGENTINA", "MÉXICO", "CUBA", "VENEZUELA", "URUGUAY", "CHILE"]
}
```

Las tasas de participación por origen se publican junto a las observadas (la página enseña las observadas y «Cómo
leerlo» dice cuáles usa el modelo). Cuando Interior publique el escrutinio del 29-N, Luis añade
`escrutinio/es-2026-11-29.csv` y carga el Infoelectoral de 2026-11 como hace con cada elección; la publicación
siguiente rellena la comprobación y `checked_at`.

## Comprobación

Tests nuevos (hoy 817), con fixtures pequeños en `tests/fixtures/cera/` (recortes de los libros del INE, de los
ficheros de escrutinio y de Infoelectoral 2023, y las tablas derivadas del artículo como valores esperados):

- `test_cera_data.py`: lectura de los dos formatos del INE, de las tablas de la convocatoria, del escrutinio y de
  los consulados; la serie nacional de prueba cuadra con los totales del libro.
- `test_cera_compute.py`: 350/350 escaños de 2023; crecimiento 422.253 y ritmo ×2,2 con los periodos del artículo;
  las diez brechas de la tabla 10; cero cambios en el escenario central con el censo proyectado del artículo (×1,23,
  540.000 electores, ~37.500 votos) y con el real; la ecológica sobre todas las provincias salvo Ceuta y Melilla,
  con `r` negativa y los valores que fije la implementación (y −0,77 si se le pasa la lista de 43 del artículo); las
  rejillas con los valores de la implementación nueva y sus propiedades (decisión 4 de la revisión); la colocación
  óptima da 0 sin colocación dirigida y crece monótonamente con participación y voto; la comprobación con y sin CERA
  de 2019 y 2023 da los precedentes (Bizkaia PP/PNV; Madrid PP 16/PSOE 10 con CERA y 15/11 sin CERA, saldo del PSOE
  −1) y nueve provincias sobre la brecha en 2023; el criterio de acierto con `max_gain` sobre saldos de −1, 0, 1 y 2.
- `test_publish_unit.py`: `publish_study` escribe las partes en orden con `meta` al final, valida schemas, escribe
  los CSV, registra solo con `register=True` y árbol limpio, conserva el registro en publicaciones posteriores,
  falla al registrar dos veces sin `force`, no toca runs ni análisis.
- `test_bundle_unit.py`, `test_web_api.py`: partes y rutas de estudios en JSON y CSV; 404 de estudio desconocido.
- `test_pages.py`: `/analisis` y `/analisis/voto-cera` con un fixture de estudio: titular y subtítulo exactos,
  tres cifras, tres veredictos, las secciones por `id`, el selector de escenario, la ficha de los 23 escaños con
  cifras del fixture, la tarjeta de comprobación en los dos estados (fixture con y sin `check` del 29-N), el
  artículo solo dentro de `#datos`, el menú con «Análisis» activo; Inicio con la banda cuando hay estudio y sin ella
  cuando no; sin estudio publicado, `/analisis` dice que no hay ninguno y `/analisis/voto-cera` es 404.
- `test_js_charts.py`: las opciones de `scatter`, `slope`, `radar` y la serie con marcas.
- `docs/web/contrato.md` y `data/README.md` actualizados; el fixture de la web (`tests/fixtures/bundle`) gana el
  estudio.

Comprobación manual de Luis: `load --what cera`, `publish study`, pasear `/analisis/voto-cera` con datos reales en
escritorio y móvil, y comparar las cifras de la pregunta 1 con las del INE y las de la pregunta 3 con el artículo.

## Lo que queda al constructor

- Nombres internos, reparto en módulos dentro de `mtpy/lib/studies/cera/`, orden de los pasos del plan.
- Cómo casa las candidaturas del fichero de escrutinio CERA con las del Infoelectoral provincial (los nombres
  difieren en mayúsculas, acentos y siglas compuestas); un mapa en `data/cera/` si hace falta, documentado.
- El umbral de censo mínimo de la ecológica, el tamaño de los bloques de la colocación óptima y las participaciones
  «media desde 1986» y «máxima» de la réplica.
- La redacción final de los textos, «Cómo leerlo» y glosario, en el tono descrito, para revisión de Luis en el
  paseo.
- Colores y tamaños exactos de los gráficos y de las tarjetas-pregunta; el `shade` de las rejillas.
- El fixture sintético del estudio y los recortes de los ficheros reales.

## Decisiones tomadas tras la revisión del constructor (2026-10-19)

Un revisor leyó el documento como constructor y devolvió siete preguntas. Tres son de Luis (1 a 3, abajo, con la
respuesta cuando la dé); las otras cuatro se resuelven aquí como decisiones mías.

1. **Qué se registra y cómo se decide el acierto.** Luis: «la hipótesis acierta si no hay movimiento de escaños
   significativo a favor del PSOE tras el recuento del voto CERA». Se mide con la prueba con y sin CERA (saldo
   neto del PSOE y de la izquierda); el umbral de «significativo» es más de 1 escaño (2 o más es fallo), decisión
   de Luis, parámetro `check.max_gain` = 1 (D2, D3, D5).
2. **La ficha «2023 ya pasó la prueba»** dice lo que dicen los datos: nueve escaños en juego con el criterio del
   informe, uno cambió (Madrid) y en contra de la hipótesis (D5, punto 8).
3. **La ecológica** se calcula sobre todas las provincias salvo Ceuta y Melilla y publica la correlación que
   salga (D2).
4. **Reproducción de las tablas 9, 11 y 14: exacta donde el método está definido, documentada donde no.** Son
   exactos y bloqueantes: los 350 escaños de 2023, las brechas de la tabla 10, los crecimientos de la pregunta 1 y el
   cero del escenario central (con la proyección del artículo y con el censo real). Las tres rejillas se
   reimplementan con las definiciones de D2 y el constructor anota en esta sección cada celda que difiera del
   artículo y por qué; los tests fijan los valores de la implementación nueva y sus propiedades (cero sin
   colocación dirigida, monotonía con participación y sesgo, el «+1» de la celda observada solo si el reparto
   uniforme lo produce). Para la tabla 11 queda definido lo que el artículo no decía: la fracción de columna se
   aplica al voto de izquierda + derecha de cada provincia, y los separatistas, regionalistas y otros conservan su
   cuota del CERA de 2023.
5. **El registro es explícito, no automático.** `Publish.run(what='study', study='voto-cera', register=True)`
   registra (`registered_at`, `registered_commit`), exige el árbol limpio (`dirty` falso) y falla si ya había
   registro, salvo `force`, que escribe `registered_changed_at`. Sin registro, la página no muestra la píldora
   «Predicción registrada» ni la fecha, y la tarjeta de comprobación dice que la predicción está en borrador. Así
   el paseo de Luis con datos reales no registra nada por accidente.
6. **La cifra de cabecera «Votos CERA esperados» se expresa en votantes**, la magnitud de la participación oficial
   (233.688 en 2023): votantes de 2023 más electores nuevos por participación por origen. La tabla de la pregunta 3
   lleva las dos filas, votantes y votos a candidaturas (203.115 en 2023, la que usa el modelo), con sus
   proyecciones. «Cómo leerlo» explica la diferencia (papeletas recibidas frente a votos escrutados a candidaturas).
7. **El grupo de exposición «media» se deduce por ajuste y se documenta como curación propia.** El constructor
   clasifica los países con la regla del artículo (alta: Argentina, Cuba, México, Venezuela, Uruguay, Chile; baja:
   Europa Occidental; media: el resto de América, Estados Unidos y el resto del mundo), ajusta la lista hasta
   reproducir los tres totales de la tabla 2 si es posible y, si no, deja la regla y anota la diferencia en
   `countries.csv` y aquí.

## Cierre del constructor (2026-10-19)

Construida el 19 de octubre por desarrollo dirigido por subagentes, en 19 tareas: las once primeras en `dev`
(`2f3c91f`..`fefb3a9`) y las otras ocho en la rama `analisis-voto-cera`, que sale de `fefb3a9` porque Luis tenía
trabajo sin confirmar en los mismos ficheros. La suite y el paseo local están en el «Estado» de la cabecera. Aquí
queda lo que el constructor decidió o encontró y la página publica.

### Cifras publicadas que difieren del artículo, y por qué

Donde la cifra del artículo no sale de los datos correctos, mandan los datos: los tests y la página usan el valor
calculado.

- **Electores y votantes CERA de 2023.** Salen de Infoelectoral (`Censo CERA` y `Total votantes CERA`), la
  fuente que usa el reparto, no del fichero de Interior, que trae erratas (electores de Albacete, 862 por 8.064;
  votantes de Barcelona, 19.315 por 21.434; electores de 2019 truncados). Son 2.328.260 electores (la OCE da
  2.328.261) y 208.431 votantes. La OCE habla de 233.688 papeletas: la página cita 208.431 y «Cómo leerlo»
  explica la diferencia. Esto corrige la decisión 6 de la revisión, que daba 233.688 como cifra de la página.
- **Votos a candidaturas de 2023: 203.152** (el artículo da 203.115). Es la suma de las columnas de candidatura
  según el mapa `escrutinio/candidatures.json`.
- **Censo de Madrid en la convocatoria: 498.874** (la prensa da 494.799). Sale de `elecgen2026_tab8.xlsx`.
- **Participación por origen.**
  - Castilla y León 2026 da 14,4 / 5,0 / 4,8 (el artículo, 11,5 / 7,6 / 5,0): el artículo agrupó los consulados
    de Estados Unidos con Europa.
  - Aragón 2026 da 15,8 / 4,0 / 5,1.
- **Cuota de la izquierda en 2023: 49,7 %** sobre provincias (el artículo, 50,0 %).
- **Inferencia ecológica: r = −0,743 sobre 50 provincias.** Con la lista de 43 del artículo da −0,767. Las siete
  provincias de menos que excluía el artículo fueron un fallo al cruzar los nombres, no un criterio. Cuartiles
  59,8 / 44,3.
- **Brecha del último escaño.** Son los votos que el segundo necesita para **igualar** el último cociente del
  ganador, redondeados al entero más cercano, mínimo 1. Es la regla del artículo y reproduce la tabla 10. La regla
  estricta (superarlo) da un voto más en nueve de las diez provincias.
- **Colocación óptima.** La bolsa son las altas observadas en los cuatro países desde la ley: 181.032
  electores. Con ella, la celda observada da 5; con la bolsa de 62.800 del artículo daría 2. La página cita 5.
- **Rejilla exterior.** Queda hasta 4 escaños por debajo de la tabla 11 en la columna del 50 %. El escenario
  «censo ×2, participación ×3, sesgo 0,9» da 6 / −6; el artículo daba 7 / −6.
- **Ruptura de tendencia del censo.** p de la pendiente ≈ 0,0004 y p del nivel ≈ 0,61, sobre la serie completa.
  La página afirma solo el cambio de pendiente. El p ≈ 0,01 del nivel en el artículo venía de otra
  especificación.
- **Provincias sensibles.**
  - La cuota con todos los países pasa de 24,1 % a 17,4 %. Va en el gráfico, como en la infografía.
  - Con los tres países (Argentina, Cuba, Venezuela), de 30,0 % a 26,7 %. Va en el texto de la pregunta 2.
- **Exposición a la Ley de Nietos.** El grupo de cada país se ajustó a la tabla 2 (decisión 7):
  - baja: Europa Occidental y Canadá, África, Asia y Oceanía;
  - totales 209.591 / 206.703 / 160.691, a ±3 de los de la tabla 2;
  - la pertenencia es curación propia, documentada en `countries.csv`, no la regla literal.
- **Columna de España Vaciada.** Lleva las dos candidaturas con la sigla EV: 70 votos de Asturias son de
  «Asturias Existe-España Vaciada». El PSC va con el PSOE, como en Infoelectoral (no tiene fila propia).
- **Litigio.** Se descuentan exactamente las 163.020 inscripciones: 279.237 electores nuevos y 22.412 votos, cero
  cambios. Con el censo de la convocatoria salen 442.257 electores nuevos y 30.482 votos, cero cambios. Con la
  proyección del artículo, 539.260 y 37.006, cero cambios.

### Datos y parámetros

- **`ecological_min_voters` = 200 sustituye a `ecological_min_census` (D8).** La exclusión de la ecológica va
  por votantes CERA: Ceuta tuvo 89 en 2023 y Melilla 177. Por censo no se puede, porque el de Melilla (5.971)
  supera al de Cuenca.
- **Reparto del litigio por provincia.** Usa el crecimiento latinoamericano desde el 1-10-2022 hasta el último
  cierre mensual, no hasta el 1 de agosto. La serie mensual es la única fuente provincia × país, y la diferencia
  es de un mes.
- **Periodos actualizados.** Tienen la misma longitud (47 + 47 meses: 2018-11 → 2022-10 → 2026-09). Avanzan con
  cada cierre: el segundo acaba en el último y el primero gana meses por el principio. La página enseña estos
  (`now`) en las preguntas 1 y 2 y en la correlación del debate; la réplica del artículo con su ventana (marzo 2019
  – octubre 2022 – mayo 2026: ×2,2, 422.253, 24,1 → 17,4) queda en «Cómo leerlo». Una parte sin `now` cae a la
  del artículo y «Cómo leerlo» lo dice.
- **Siglas.** Las partes llevan el nombre completo de Infoelectoral y su sigla (`winner_sigla`,
  `runner_up_sigla`, `winner_with_sigla`, `winner_without_sigla`); la página y la isla solo imprimen la sigla.
- **Censo de 2023 del incremento.** Es el cierre mensual del 1-6-2023 (`meta.election.baseline_date`), 2.327.388,
  no el censo oficial de Infoelectoral (2.328.260): «Cómo leerlo» de la pregunta 3 lo dice. La cifra de sobres de
  la OCE (233.688) no la lleva ninguna parte: es un hecho de contexto en `OCE_RECEIVED` (`pages/cera.py`).
- **Predicción registrada.** Texto de `registered.claim`: «El voto CERA de las generales del {fecha} no dará al
  PSOE ni a la izquierda un saldo neto de más de {max_gain} escaño(s)». Va con los escaños por bloque de los dos
  escenarios.
- **`check.max_gain` = 1**, el umbral que decidió Luis.
- **Réplica del informe «hasta 23 escaños».**
  - Lee la participación CERA de todas las generales de Infoelectoral con columnas CERA: 13 ficheros, de 1986 a
    2023. Los de 1977–1982 no tienen esas columnas.
  - Media 18,9 %, máxima 40,2 %.
  - Con ellas, 523 k y 1,114 M votos, y 20 y 27 provincias.
- **Fechas de las elecciones.** Salen de `data/wikipedia/wp-urls.json`.
- **Constantes de presentación en `study.py`, no en `params.json`.** Son los ejes de las rejillas, los seis
  escenarios, los tres países sensibles, el nombre del PSOE y las siglas de los ratios: definen qué enseña la
  página, no el modelo.
- **Lectura del INE.**
  - Las filas de un elector mal archivadas se quedan con su código de provincia (±1 en los totales provinciales
    de 2010–2011).
  - Los países que cambiaron de nombre se cruzan por `country_code`.
  - El duplicado de febrero de 2011 conserva la lectura «(nuevo)», con un salto de método de +7.344.
- **Desempate del reparto.** `allocate_results` ordena las candidaturas por votos, de más a menos, para que un
  empate exacto de cocientes vaya a la de más votos, como dice la LOREG. En los datos oficiales no cambia nada.

### Decisiones de estructura

- **Un tipo nuevo de publicación: los estudios.**
  - En el paquete viven en `studies/{study}/`, con ocho partes y sus CSV, `meta.json` el último y la entrada
    `manifest.studies`.
  - En la API, `/api/v1/studies` y `/api/v1/studies/{study}/{part}`.
  - Los publica `publish.publish_study`, desde `Publish.run(what='study', study=…)`.
  - Están documentados en `docs/web/contrato.md` («Estudios», «Rutas de estudios»).
- **Reparto en módulos.**
  - `seats.py` (reparto, brechas, efecto neto) queda solo.
  - `stress.py` lleva la rejilla exterior, los escenarios, la colocación óptima y la réplica.
  - `check.py` lleva la prueba con y sin CERA y el veredicto.
  - En la web, `pages/analisis.py` tiene la sección (índice, tarjetas, contexto, isla) y `pages/cera.py` los
    modelos de vista de cada pregunta, del debate y de la comprobación.
- **`study_meta`.** La plantilla recibe la `meta` del estudio con ese nombre, porque `meta` es la del run, que lee
  el pie.
- **Ruta.** Es `/analisis/{study}`, con el alias por defecto: `:str` es `[a-z]+` y no admite el guion de
  `voto-cera`.
- **Importaciones perezosas.** `publish/studies.py` importa el estudio dentro de `publish_study`, y `census.py`
  importa statsmodels dentro de `trend_break`. Así, arrancar la web no carga ni el estudio ni `statsmodels.api`.
- **Registro explícito.** Solo con `register=True` y el árbol limpio. `force` reescribe y deja
  `registered_changed_at`.
- **Gráficos.** `web/dist/js/charts/grouped.js` (barras agrupadas, apiladas y horizontales) se suma a `census`,
  `scatter`, `slope` y `radar`. El selector de escenario funciona sin JS para las tablas (con `:has()` en CSS);
  el gráfico de brechas lo redibuja `web/dist/js/pages/analisis-cera.js`.

### Lo que hace Luis ahora

1. **Cargar y publicar contra el RDS/S3,** como con lo demás:
   `python load/run_load.py --scopes es --what cera`, revisar el diff de `data/cera/ine/` y confirmarlo, y
   después `python job.py publish '{"what": "study", "study": "voto-cera"}'`. El constructor no bajó nada del
   INE.
2. **Pasear las páginas en un navegador,** en escritorio y a 390 px: `/`, `/analisis` y `/analisis/voto-cera`.
   El constructor solo pudo comprobar el HTML servido, no los gráficos.
   - Los enlaces de fuente de Interior y de los boletines autonómicos apuntan a portadas
     (`INFOELECTORAL_URL`, `REGIONAL_SOURCES` en `publish/studies.py`).
   - En el móvil, la etiqueta de un saldo negativo grande pisa los nombres de provincia en el gráfico de
     brechas.
   - Un fichero nuevo en `consulados/` sale con su nombre crudo hasta que se añada a `COMMUNITY_NAMES` en
     `web/dist/js/pages/analisis-cera.js`.
3. **Registrar cuando la dé por buena,** con el árbol limpio:
   `python job.py publish '{"what": "study", "study": "voto-cera", "register": true}'`.
4. **Tras el escrutinio del 29-N:**
   - añadir `data/cera/escrutinio/es-2026-11-29.csv` y su entrada en `candidatures.json` (sin la entrada, el
     cálculo para con «cera: no candidatures map for 2026-11-29 in cera/escrutinio/candidatures.json»);
   - cargar `PROV_02_202611_1.xlsx` y su `.json` a la vez, como en cada elección: el `.json` lleva las siglas, y
     sin él no hay bloques ni siglas en la página;
   - antes de publicar la comprobación, cotejar `params.blocks` con las siglas de `PROV_02_202611_1.json`: una
     sigla de izquierda que no esté en los bloques cae en «Otros» y rebaja `net_left`, que es lo que decide el
     acierto;
   - publicar otra vez: la comprobación y `checked_at` se rellenan solos.

   Mientras falte el XLSX, la publicación deja fuera el escrutinio del 29-N y lo anota en
   `meta.skipped_scrutinies`.

   Estos tests fijan el estado de antes de la elección y habrá que actualizarlos cuando lleguen los ficheros del
   29-N (y, el de los libros mensuales, también en enero de 2027, con `cerapropais_27`):
   - `tests/test_cera_behaviour.py`: los recuentos de ficheros (`test_available_results_lists_every_general_election`,
     16 y el último 2023-07-23; `test_participation_generales_periods`, 13);
   - `tests/test_cera_study.py`: `test_compute_cera_runs_end_to_end` (las dos comprobaciones y `verdict_input`
     nulo), `test_compute_cera_behaviour_block` (13 resultados y las dos generales con escrutinio),
     `test_compute_cera_seats_block` (la réplica del informe) y
     `test_read_scrutinies_skips_the_elections_without_results_and_needs_their_candidatures`;
   - `tests/test_publish_studies.py`: `test_export_study_figures` (las dos fechas de la comprobación y el veredicto
     nulo) y `test_publish_study_sets_checked_at_when_the_election_has_a_scrutiny` (`checked_at` nulo antes);
   - `tests/test_cera_data.py`: `test_cera_data_folder_is_complete` (21 libros mensuales; 22 en enero de 2027).
5. **Cada mes,** `--what cera` y `publish study` para refrescar la serie.
6. **Integrar.** Fusionar la rama `analisis-voto-cera` en `dev` después de confirmar su trabajo pendiente. Habrá
   conflictos previsibles en `pages/common.py` y en los ficheros de test.
7. **Revisión final.** La lista de menores aplazados está en el registro del desarrollo
   (`.superpowers/sdd/2026-10-19-web-analisis-voto-cera/`).
