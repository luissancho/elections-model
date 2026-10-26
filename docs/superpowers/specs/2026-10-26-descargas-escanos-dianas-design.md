# Descargas: Reparto de escaños, Calculadora de mayorías y las dos dianas

Fecha: 2026-10-26. Estado: pedida por Luis y CONSTRUIDA el 2026-10-26 en `dev` (sin commits: los hace Luis).

## Intención

Luis quiere descargar como imagen, con el mismo botón «Descargar» de Promedios (`share.js`), cuatro gráficos más:

- «Reparto de escaños»: la barra y la tabla de la vista (Partidos o Bloques), sin las curvas de densidad.
- «Calculadora de mayorías»: completa, con el chip, la barra y el histograma de la selección del momento.
- La diana del histórico de errores (`/elecciones`, «Hacia dónde se equivocaron») y la del perfil de encuestadora
  (`/casas/{slug}`, «Diana de errores»), con los filtros aplicados.

Y quitar el borde que rodeaba a los selectores de partidos de la calculadora.

## Qué había

`share.js` dibujaba una sola opción de ECharts bajo título y subtítulo, con un panel opcional a la derecha (la tabla
del hemiciclo). El histograma sólo tenía `renderHistogram` (sin la opción suelta). El fieldset de las píldoras de la
calculadora llevaba la clase `lg`, que es la de los botones de la leyenda (`legend-dom.js`): borde, radio, relleno y
fondo alrededor de todas las píldoras. Eso era el «borde».

## Enfoques

- **A.** Una opción combinada de ECharts con dos rejillas para la calculadora. Descartado: remapear ejes, marcas y la
  leyenda del histograma es frágil.
- **B.** Generalizar `imageLayout`/`composeImage`/`wireDownload` a una columna de gráficos apilados, un chip y un panel
  debajo, reutilizando las opciones de cada gráfico tal cual. Elegido: cada gráfico se dibuja fuera de pantalla con su
  propia opción y la composición es sólo geometría.

## Decisiones (mías, con su razón)

1. **Reparto de escaños**: imagen de 800 px con la barra (120 px) y la tabla de la vista mostrada debajo, centrada
   (`bottom`); la imagen se ensancha si la tabla lo pide. La tabla de partidos (siete columnas) no cabe al lado de una
   barra legible, y la lectura de la página es barra-encima-tabla.
2. **Calculadora**: chip como en la página (`CHIP`, verde con `ok`), barra (120 px) e histograma (300 px) con su
   leyenda; el subtítulo intercala los partidos marcados tras el nombre de la tarjeta («Calculadora de mayorías · PP +
   VOX · Pronóstico para …») y el fichero lleva su slug. Sin partidos marcados el botón queda deshabilitado
   (`.tool-btn:disabled`).
3. **Dianas**: gráfico cuadrado (744 × 744 CSS px, para que los anillos sigan redondos). Título fijo del servidor (la
   encuestadora en el perfil; «Histórico de errores» en /elecciones) y el JS añade al subtítulo los filtros elegidos
   (etiquetas de los selectores, años como «desde 2019»/«hasta 2023»/«2019–2023») y el recuento del contador
   (`errors.js: yearsText`, `imageTexts`); el fichero lleva los filtros como slug. Filtro Jinja `slug` (`slugify`)
   para el nombre del fichero del perfil. Sin diana (sin análisis o casa sin elecciones) no hay botón.
4. **Borde**: el fieldset pasa a `class="pills"`; no había ninguna regla `.pills.lg`, así que nada más cambia.
5. `fileSlug` se muda a `format.js` (re-exportado por `share.js`) para que `errors.js` siga sin depender de ECharts.
6. La diana del modelo en /metodo/backtest usa el mismo cableado, pero no se le añade botón: no se pidió.

## Comprobación

1409 tests en verde, 76 saltados (unitarios de `share.js`, `histogram.js`, `errors.js` y de las plantillas). Comprobación en navegador con
Chrome sin cabeza por CDP sobre el paquete de fixtures: los siete botones generan su PNG, sin errores de consola.
