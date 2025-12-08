# CSIRO Image2Biomass Project – Contexto Completo

Este documento resume toda la información relevante que se ha discutido sobre la
competencia **CSIRO – Image2Biomass Prediction** de Kaggle. Está diseñado
como un recurso de referencia para grandes modelos de lenguaje (LLMs) y
personas que trabajen en el proyecto, de modo que se mantenga siempre el
contexto y los objetivos claros. Los apartados están organizados de forma
jerárquica para facilitar su lectura y la extracción de segmentos concretos.

## 🧭 Resumen ejecutivo

* **Objetivo del concurso:** predecir diversos componentes de biomasa seca en
  pasturas (vegetación verde, vegetación muerta, trébol, “green dry matter” y
  biomasa total) a partir de imágenes de vista cenital de parcelas de pastos
  (70 cm × 30 cm) y algunas mediciones de campo.  El objetivo final es
  ofrecer una alternativa rápida y escalable a las mediciones manuales,
  permitiendo a ganaderos y científicos planificar mejor el pastoreo y
  monitorizar la salud del suelo.
* **Naturaleza de la competición:** es una *research code competition* de
  Kaggle patrocinada por CSIRO, MLA y Google.  Los participantes deben
  publicar los notebooks con el código que reproduce sus predicciones.  Se
  evalúa mediante un coeficiente de determinación ponderado (R²) que da
  mayor peso a la biomasa total y a la biomasa del material verde seco.
* **Datos disponibles:** las carpetas `train/` y `test/` contienen las
  imágenes; `train.csv` proporciona `sample_id`, `image_path`, fecha,
  estado, especie dominante, NDVI promedio (`Pre_GSHH_NDVI`), altura media
  (`Height_Ave_cm`), nombre del objetivo (`target_name`) y valor de
  biomasa (`target`)【260648875389173†screenshot】.  `test.csv` sólo incluye
  `sample_id`, `image_path` y `target_name`【22011555562683†screenshot】.
* **Restricciones clave:** las entregas **deben** realizarse a través de
  notebooks en Kaggle, sin acceso a internet y con un tiempo de ejecución
  inferior a 9 horas【205478836664948†screenshot】.  Se permiten datos
  externos y modelos preentrenados siempre que sean de acceso público.
* **Estrategia actual del proyecto:** se ha implementado un MVP que extrae
  características simples de las imágenes (medias y desviaciones de los
  canales RGB y histogramas de color), combina metadatos (NDVI, altura,
  estado, especie, fecha) y entrena un regressor de múltiples salidas
  basado en XGBoost.  Este modelo ofrece una línea base robusta y
  reproducible en CPU.

---

## 📖 Descripción de la competencia

### Objetivo general

La competencia **CSIRO – Image2Biomass Prediction** reta a los
participantes a desarrollar modelos capaces de estimar con precisión la
biomasa de pasturas a partir de imágenes de vista cenital y mediciones de
campo.  La capacidad de estimar la biomasa en tiempo real ayuda a los
ganaderos a determinar si una parcela tiene suficiente forraje, cuándo
dejarla descansar y cómo optimizar la producción de carne y leche.

### Sponsors y organización

El concurso está organizado por la agencia nacional de ciencias de
Australia (CSIRO) con apoyo de Meat & Livestock Australia (MLA) y
Google.  Es parte de las **research code competitions** de Kaggle: los
participantes no solo envían predicciones, sino también el código que
permite reproducirlas.  Las mejores soluciones pueden adoptarse en
herramientas de agricultura de precisión de próxima generación.

### Normas importantes

* **Submissions vía Notebooks:** según el requisito del concurso, las
  soluciones deben enviarse como notebooks que se ejecuten en la
  infraestructura de Kaggle.  Las notebooks no pueden conectarse a
  internet, deben ejecutarse en menos de 9 horas y pueden usar tanto CPU
  como GPU【205478836664948†screenshot】.
* **Datos y modelos externos:** se permite utilizar pesos preentrenados y
  datos externos siempre que estén disponibles públicamente【205478836664948†screenshot】.
* **Límite de envíos:** los equipos pueden realizar hasta 5 envíos al día
  y deben seleccionar sus dos mejores notebooks para la evaluación final.
* **Equidad y reproducibilidad:** Kaggle puede solicitar la revisión del
  código para verificar que los resultados sean reproducibles y que se
  respeten los límites de tiempo.

---

## 🗂️ Dataset

### Estructura de los archivos

* **`train.csv`**: contiene casi 1 162 imágenes únicas y, para cada
  combinación de imagen y tipo de biomasa, incluye las columnas
  `sample_id`, `image_path`, fecha de muestreo (`Sampling_Date`), estado de
  Australia (`State`), especie de pasto dominante (`Species`), NDVI
  promedio (`Pre_GSHH_NDVI`), altura media (`Height_Ave_cm`), nombre del
  objetivo (`target_name`) y el valor de biomasa (`target`)【260648875389173†screenshot】.
  Cada imagen se repite cinco veces con distintos `target_name` y
  `target` (una por cada componente de biomasa), lo que permite
  reconstruir un vector de cinco salidas por imagen.
* **`test.csv`**: incluye `sample_id`, `image_path` y `target_name` para
  cada par imagen–objetivo【22011555562683†screenshot】.  No contiene
  metadatos, por lo que al preparar las características para el modelo
  debemos asignar valores por defecto o estimados para NDVI y altura.
* **`sample_submission.csv`**: formato de ejemplo de la entrega.  La
  columna `sample_id` identifica unívocamente cada fila (imagen +
  objetivo) y `target` debe contener las predicciones.
* **Carpetas de imágenes**: `train/` y `test/` contienen las fotos en
  formatos JPEG/PNG.  Las rutas indicadas en `image_path` son relativas
  a estas carpetas.

### Variables disponibles

1. **`image_path`** – ruta relativa a la imagen en la carpeta
   correspondiente.  Cada imagen representa un rectángulo de 70 cm × 30 cm
   de la parcela de pasto.  El encuadre y la iluminación varían según
   condiciones de campo.
2. **`target_name`** – nombre del componente de biomasa.  Puede ser:
   - `Dry_Green_g`: peso seco de vegetación verde (excluyendo trébol)
   - `Dry_Dead_g`: biomasa seca de material muerto (hojarasca)
   - `Dry_Clover_g`: biomasa seca de trébol
   - `GDM_g`: “green dry matter”, suma de vegetación verde más trébol
   - `Dry_Total_g`: biomasa seca total (verde + muerta + trébol)
3. **`target`** – valor numérico del objetivo correspondiente.  En
   `train.csv` sirve de etiqueta; en `test.csv` es la variable a
   predecir.
4. **`Sampling_Date`** – fecha en que se recolectó la muestra.
5. **`State`** – estado australiano donde se tomó la muestra (ej.
   Victoria, Nueva Gales del Sur).
6. **`Species`** – especie de pasto dominante (e.g. ryegrass, clover).
7. **`Pre_GSHH_NDVI`** – índice NDVI promedio calculado previamente sobre
   la parcela.
8. **`Height_Ave_cm`** – altura media de la vegetación en centímetros.

### Consideraciones de calidad de datos

Algunas discusiones en Kaggle indican que ciertas anotaciones del valor
de biomasa pueden contener errores o ruido (por ejemplo, imágenes sin
trébol con `Dry_Clover_g` alto).  También se observa que la altura
media tiene baja correlación con la biomasa muerta; proporciones como
`Dead/Total` y `Dead/GDM` se correlacionan mejor.  Por ello, es
importante inspeccionar las distribuciones y, cuando sea necesario,
aplicar transformaciones (como trabajar con fracciones o logaritmos) o
implementar pérdidas robustas durante el entrenamiento.

---

## 📏 Métrica de evaluación

Kaggle evalúa las soluciones utilizando un **coeficiente de
determinación ponderado (R²)**.  Para cada componente de biomasa se
calcula el R² entre los valores verdaderos y las predicciones, y luego
se promedia con un peso mayor para la biomasa total y para la materia
seca verde.  El peso exacto de cada variable está especificado en las
reglas del concurso (por ejemplo, `Dry_Total_g` recibe alrededor del
50 % del peso total).  Este enfoque obliga a los modelos a priorizar un
buen ajuste de la biomasa total, mientras que un error más alto en
`Dry_Clover_g` impacta menos el marcador.

En la práctica, al entrenar un modelo de referencia se puede utilizar
una pérdida estándar (MSE o Huber), pero conviene medir el rendimiento
en validación con el R² ponderado para alinear las mejoras con la
métrica de la competencia.

---

## 🛠️ Estrategia y línea base

### Extracción de características

**Imágenes**.  Para un MVP sin librerías de deep learning se extraen
características numéricas simples:

1. **Estadísticas básicas** – media y desviación estándar de los
   valores de los canales rojo, verde y azul.  Capturan brillo y color
   globales.
2. **Histogramas de color** – histogramas normalizados de 16 bins para
   cada canal.  Representan la distribución de colores de manera
   compacta.

Esta vectorización produce un vector de 54 dimensiones por imagen.  En
entornos con PyTorch/TensorFlow se pueden sustituir estos vectores por
embeddings extraídos de redes preentrenadas (p. ej. EfficientNet o
ResNet), lo que mejora notablemente la capacidad predictiva.

**Metadatos**.  Se usan directamente `Pre_GSHH_NDVI` y `Height_Ave_cm`.
La fecha se transforma a un número (días desde la fecha mínima), y las
variables categóricas `State` y `Species` se convierten en vectores
“one‑hot”.  Todas las variables numéricas se estandarizan para que
tengan varianza comparable.

### Modelo de referencia

Para manejar las cinco salidas simultáneamente se emplea un
**`MultiOutputRegressor`** de scikit‑learn con un `XGBRegressor` como
estimador base.  XGBoost es robusto con estructuras de datos mixtas y
no necesita GPU, lo que facilita su ejecución en entornos limitados.

Pasos de entrenamiento:

1. **Pivotar datos** para que cada imagen sea una fila con cinco
   columnas de objetivos.
2. **Extraer las características** de imagen y metadatos.
3. **Dividir en entrenamiento y validación** (por ejemplo, 80 %/20 %).
4. **Preprocesar** con `ColumnTransformer` (one‑hot para categóricos y
   normalización para numéricos).  Las características de imagen se
   mantienen sin escala adicional.
5. **Entrenar el modelo** XGBoost (ajustar número de árboles,
   profundidad y tasa de aprendizaje).  Se evalúa con MSE pero también
   se calcula el R² ponderado en la validación para guiar la
   sintonización.
6. **Inferir en el conjunto de test** generando las predicciones para
   cada imagen.  Como `test.csv` no incluye metadatos, se usan
   valores por defecto o estimados; se emparejan las predicciones con
   cada `sample_id` y se guarda `submission.csv`.

### Mejoras posibles

* **Extracción de features con CNN/ViTs:** usar embeddings de modelos
  preentrenados (EfficientNet, ViT, DINO) para captar patrones espaciales
  complejos.  Los embeddings pueden combinarse con los metadatos y
  alimentarse a un modelo más simple (XGBoost, Ridge, NN).  Se requiere
  ejecutar PyTorch o TensorFlow en Kaggle.
* **Modelos multi‑tarea o en dos fases:** predecir primero la biomasa
  total y luego la fracción muerta o las proporciones entre
  componentes, como sugiere la comunidad.  Esto puede mejorar la
  estabilidad de la predicción de `Dry_Dead_g`.
* **Pérdidas robustas:** usar pérdidas tipo Huber o Tukey para reducir el
  impacto de valores atípicos y etiquetados erróneos.
* **Cross‑validation estratificada:** dividir las imágenes en
  foldings que respeten la distribución de estados/estaciones y la
  biomasa total, evitando que un fold contenga todas las muestras de
  cierto estado.

---

## 🤖 Reglas de uso para el LLM

Cuando interactúes con un modelo de lenguaje basándote en este
proyecto, utiliza las siguientes reglas para asegurar coherencia y
relevancia:

1. **Enfoque en el objetivo:** recuerda que la meta es estimar biomasa de
   pasturas; evita divagar hacia temas no relacionados.
2. **Respeta las restricciones de Kaggle:** no propones soluciones que
   requieran internet durante la ejecución o superen 9 horas de
   computación.  Recuerda que las submissions deben ser notebooks【205478836664948†screenshot】.
3. **Sé reproducible:** cualquier código o método sugerido debe
   ejecutarse con las herramientas disponibles en el entorno de Kaggle.
4. **Comenta supuestos:** si falta información (por ejemplo,
   versiones de librerías), indica los supuestos que realizas.
5. **Formato de salida:** al solicitar código, entrega siempre
   pseudocódigo claro y luego explica qué hace cada parte; cuando sea
   posible, proporciona referencias o comentarios para guiar al lector.

---

## 🧪 Backlog / Próximos pasos

* **Análisis exploratorio profundo:** explorar distribuciones de los
  objetivos y metadatos, detectar outliers y errores de etiquetado.  Usar
  gráficos y estadísticas descriptivas para orientar nuevas
  transformaciones.
* **Pruebas con embeddings de imagen:** implementar un pipeline en
  Kaggle que cargue un modelo preentrenado (EfficientNet, ViT o DINO),
  obtenga un vector de características por imagen y entrene un regressor
  simple.  Comparar su rendimiento con la línea base.
* **Ajuste de hiperparámetros de XGBoost:** explorar más árboles,
  regularización, métodos de muestreo y pérdida Huber para mejorar el
  ajuste.
* **Ensamblados y promediado:** combinar varios modelos (árboles,
  redes, regresores lineales) para obtener predicciones más robustas.
* **Predicción en dos etapas:** estudiar la sugerencia de la comunidad de
  predecir la biomasa total y luego derivar la biomasa muerta como una
  fracción; evaluar si mejora el R² ponderado.
