# Resultados y justificación

## Métrica y validación

El modelo se evalúa con tres métricas de regresión sobre un conjunto de test separado del de entrenamiento:

| Métrica | Valor (random_forest) | Valor (baseline lineal) |
|---|---|---|
| MAE (meses) | 11.41 | 17.48 |
| RMSE | 15.34 | 21.88 |
| R² | 0.533 | 0.051 |

Se usa **MAE** como métrica principal porque se interpreta directamente en las unidades del negocio ("la predicción falla, en promedio, por X meses"), algo que un responsable de riesgo puede leer sin traducir. **RMSE** se reporta junto a MAE porque penaliza más los errores grandes: para gestión de libro de riesgo, unas pocas predicciones muy equivocadas importan más que muchos errores pequeños, y RMSE > MAE ya indica que existen esos casos de cola. **R²** da el marco de referencia: cuánta varianza de `avg_duration_months` explica el modelo frente a predecir siempre la media.

La separación train/test es **cronológica por `requested_date`** (80% más antiguo entrena, 20% más reciente evalúa), no aleatoria. Esto replica cómo se usará el modelo en producción — prediciendo sobre RFQs nuevas, nunca vistas — y evita la fuga de información que se daría si RFQs futuras entrenaran el modelo que luego predice sobre RFQs pasadas.

Como referencia de sanidad se entrena también una regresión lineal simple (mismo preprocesado, sin ajuste). El random_forest la mejora en MAE (-35%) y en R² (10x), confirmando que el modelo capta señal real y no es puro ruido con parámetros ajustados.

## Qué variables importan

Las variables más influyentes del modelo (importancia de features del random_forest) son, en orden:

1. **`product_type`** (en particular la categoría "Wretched Hive Digital"): la variable individual más importante, por delante de cualquier estadístico de volatilidad.
2. **`structural_base_vol_max`**: la volatilidad estructural del subyacente más volátil de la cesta.
3. **`protection_barrier_pct`** y **`autocall_barrier_pct`**: los niveles contractuales de barrera.
4. **`observation_frequency_days`**: cada cuánto se comprueba la condición de autocall.
5. **`realized_vol_63d_min/mean/max`**: la volatilidad de mercado reciente de la cesta.

Tiene sentido de negocio: `avg_duration_months` depende de cuándo (si acaso) se toca la barrera de autocall, y eso depende directamente de los términos contractuales del producto (qué tipo es, dónde están sus barreras, con qué frecuencia se observa) y de la volatilidad del subyacente (más volatilidad, más probabilidad de tocar la barrera antes). Que `product_type` domine por encima de la volatilidad de mercado sugiere que la mecánica propia de cada estructura (p. ej. cuán agresiva es su barrera típica) explica más varianza en la duración que las condiciones de mercado del momento — información útil para la mesa: el diseño del producto pesa más que el timing de mercado a la hora de anticipar cuánto va a durar.

## Limitaciones

- **Techo de R² estructural, no solo de modelado.** `avg_duration_months` depende del camino futuro del precio del subyacente después de cotizada la RFQ — información que, por definición, no existe en el momento de la predicción. Es estructuralmente parecido a predecir un tiempo de primer paso de un proceso estocástico: ningún modelo, por sofisticado que sea, puede eliminar esa incertidumbre con features estáticas de la fecha de cotización. Un R²≈0.53 puede estar razonablemente cerca del techo alcanzable con esta información.
- **Se probaron mejoras adicionales sin éxito claro.** Añadir el momentum de volatilidad realizada (cambio en los últimos ~63 días hábiles) y estacionalidad (mes de la solicitud) dio una mejora dentro del margen de ruido del test set (bootstrap con intervalo de confianza que cruza cero). Cambiar el modelo a gradient boosting (HistGradientBoostingRegressor) dio una mejora algo más consistente (R² 0.533→0.545) pero tampoco decisiva. Se optó por mantener random_forest por su relación simplicidad/rendimiento y por ser ya el modelo validado end-to-end con la API.
- **Referencia de subyacentes pequeña y poco diversa.** Solo 14 tickers, cada uno con un sector distinto — no hay dos subyacentes que compartan sector, lo que limita cualquier análisis de riesgo sectorial y hace que `sector` no aporte señal más allá de la que ya da `structural_base_vol` (descartada como feature por esta razón).
- **El dataset es sintético.** Las magnitudes y relaciones pueden no replicar completamente la dinámica de un libro de autocallables real; el modelo debería revalidarse si se aplica a datos de producción.
