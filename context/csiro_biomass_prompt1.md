# Mini‑prompt: Contexto resumido del proyecto CSIRO – Image2Biomass

Este breve resumen contiene la información esencial del proyecto para
alimentarla como contexto inicial a un modelo de lenguaje.  Se centra en
objetivo, datos, restricciones del concurso y la estrategia de línea
base.

## 📌 Puntos clave

* **Objetivo:** predecir la biomasa seca (verde, muerta, trébol, GDM y
  total) de pasturas a partir de imágenes cenitales y metadatos de
  muestreo.  Esto ayuda a agricultores y científicos a gestionar mejor
  el pastoreo.
* **Dataset:** `train.csv` contiene, para cada imagen, fecha, estado,
  especie, NDVI, altura y cinco filas con distintos componentes de
  biomasa y su valor【260648875389173†screenshot】.  `test.csv` incluye
  `sample_id`, `image_path` y `target_name`【22011555562683†screenshot】;
  las predicciones deben devolverse en un `submission.csv`.
* **Métrica:** el desempeño se mide mediante un R² ponderado que da
  mayor peso a la biomasa total y a la materia verde; los pesos están
  definidos en las reglas de Kaggle.
* **Restricciones del concurso:** las entregas deben ser notebooks de
  Kaggle sin acceso a internet y ejecutarse en ≤ 9 h【205478836664948†screenshot】.
  Se permiten modelos preentrenados y datos externos siempre que sean de
  libre acceso【205478836664948†screenshot】.
* **Línea base actual:** se ha implementado un MVP que extrae
  características sencillas de imagen (medias RGB y histogramas) y
  combina NDVI, altura, estado, especie y fecha transformada.  Se
  entrena un `MultiOutputRegressor` con XGBoost para predecir las cinco
  salidas simultáneamente, evaluando con R² ponderado.  Este enfoque no
  requiere librerías de deep learning y sirve como punto de partida
  reproducible.

## 🧠 Cómo usar este contexto

* Al formular preguntas o pedir código, recuerda que la solución debe
  ajustarse a las normas del concurso (notebook‑only, < 9 h de
  ejecución).  No propongas soluciones que requieran internet en tiempo
  de scoring.
* Si falta información (p.ej. versiones de librerías), explica los
  supuestos razonables que adoptas.
* Para desarrollar mejoras, sugiere explorar embeddings de imagen
  preentrenados (EfficientNet, ViT), modelos multi‑tarea y pérdidas
  robustas.
