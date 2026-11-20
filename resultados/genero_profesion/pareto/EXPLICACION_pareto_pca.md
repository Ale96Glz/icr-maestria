# Análisis de soluciones no dominadas (Pareto) + PCA

**Proyecto:** Mitigación de sesgo (género y profesión) en LLMs con LoRA.
**Configuraciones analizadas:** 30 = 5 modelos × 2 rutas × 3 % LoRA.
**Fuente de datos:** `resultados/dataset_mitigacion_tesis.csv` (prefijo `lrm` = modelo con LoRA activo = valores absolutos).
**Scripts:**
- `scripts/pareto_analysis.py` — vector completo (16-D): Pareto + capas + PCA.
- `scripts/pareto_reducido_diagnostico.py` — diagnóstico del "por qué" + variante reducida a 6 familias.

---

## 1. Objetivo del tutor

> *"Meter las variables para el frente de Pareto (SS, LMS, ICAT, STS, Regard, Regard Gap) para las soluciones no dominadas (1er frente) y sus vectores (análisis de soluciones no dominadas sobre el vector completo). Una vez que tenga los vectores y las N dimensiones para cada modelo, aplicarle PCA para bajarlos a 2D y hacer las gráficas de dispersión para ver si se presentan las características de dominancia. Si no se puede, tengo que explicarle por qué sucede esto."*

Se reemplazan los promedios ponderados anteriores (M_GP, M_stereo, MEI_GP) por:
1. **Análisis de soluciones no dominadas** (frente de Pareto multidimensional).
2. **PCA** para visualización 2D y comprobar si la dominancia se manifiesta geométricamente.

---

## 2. El "vector completo" = 16 dimensiones

Cada configuración se describe por un vector de **16 métricas** (valores absolutos del modelo con LoRA), agrupadas en las 6 familias que enumera el tutor:

| Familia | Sub-métricas (dimensiones) | nº |
|---|---|---|
| **SS** (StereoSet bias) | `SS_inter_gen`, `SS_intra_gen`, `SS_inter_prof`, `SS_intra_prof` | 4 |
| **LMS** (modelado de lenguaje) | `LMS_inter`, `LMS_intra` | 2 |
| **ICAT** (idealized CAT) | `ICAT_inter_gen`, `ICAT_intra_gen`, `ICAT_inter_prof`, `ICAT_intra_prof` | 4 |
| **STS** (CrowS-Pairs) | `STS_gen`, `STS_soc` | 2 |
| **Regard** (tono BOLD) | `regard_gen`, `regard_prof` | 2 |
| **Regard Gap** (equidad BOLD) | `rg_gap_gen`, `rg_gap_prof` | 2 |
| | **Total** | **16** |

### Normalización a "mayor = mejor" (todo en [0,1])

Para que la dominancia tenga sentido, todas las métricas deben apuntar en la misma dirección:

- **SS, STS** (ideal = 50; sin sesgo): `norm = 1 − |x − 50| / 50`
- **LMS, ICAT** (mayor mejor): `norm = x / 100`
- **Regard** (de [−1,1] a [0,1]): `norm = (x + 1) / 2`
- **Regard Gap** (menor = más equitativo → invertir): `norm = 1 − x`

Verificado: todos los valores caen en [0,1] **sin necesidad de clipping** (mín global 0,289; máx 1,000).

> **Nota metodológica (colinealidad):** ICAT integra LMS y SS por construcción (`ICAT = LMS × f(SS)`). Incluir SS, LMS e ICAT a la vez introduce dependencia lineal parcial. Se incluyen las 16 por petición expresa del tutor; esto afecta a la *interpretación* de los loadings del PCA, no a la validez del frente de Pareto. Queda documentado por si surge en la defensa.

---

## 3. Definición de dominancia

Una configuración **B domina a A** si:
- `B[i] ≥ A[i]` en **todas** las dimensiones, y
- `B[i] > A[i]` en **al menos una**.

El **primer frente de Pareto** son las configuraciones **no dominadas** por ninguna otra. El sorting por capas (NSGA-II, Deb et al. 2002) asigna rango 1 a las no dominadas, rango 2 a las que solo domina el rango 1, etc.

---

## 4. RESULTADO sobre el vector completo (16-D): la dominancia NO discrimina

| Métrica | Valor |
|---|---|
| Configuraciones en el **1er frente** | **29 de 30** |
| Configuraciones dominadas | **1** (Llama 3.2-1B, *solo dataset*, 3%) |
| Capas NSGA-II | **2** (tamaños 29 + 1) |
| Varianza PCA (PC1+PC2) | 66,8% (PC1 45,2% + PC2 21,7%) |

La única configuración dominada (Llama 3.2-1B, solo dataset, 3%) es batida en las 16 dimensiones por: GPT-2 Base D+p 1%, GPT-2 Medium D+p 1/2/3% y GPT-2 XL D+p 2%.

**El frente de 29/30 no sirve para seleccionar configuraciones**, que era el objetivo. El scatter PCA (`pareto_pca_scatter.png`) muestra estructura (PC1 separa ruta y familia) pero casi todos los puntos son "rango 1".

---

## 5. POR QUÉ sucede: maldición de la dimensionalidad

Esto **no es un error** del método ni de los datos; es una propiedad geométrica conocida de la optimización multiobjetivo. La evidencia cuantitativa (script de diagnóstico):

### 5.1 Comparabilidad por pares

- Pares totales de configuraciones: **435**.
- Pares **comparables** (uno domina al otro): **5 (1,1%)**.
- Pares **mutuamente no dominados**: **430 (98,9%)**.
- **Mediana de dimensiones que una configuración le gana a otra: 8 de 16** (justo la mitad).

Interpretación: para que A domine a B hay que ganar en las **16** dimensiones simultáneamente. Pero, en promedio, cada configuración gana solo en ~8 y pierde en ~8. La probabilidad de "barrer" las 16 es ínfima → casi nadie domina a nadie.

### 5.2 Tamaño del frente según el número de objetivos

Promediando sobre subconjuntos aleatorios de dimensiones (figura `diagnostico/figuras/frente_vs_dimensiones.png`):

| nº objetivos | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frente medio | 1,2 | 3,2 | 6,6 | 11,1 | 13,2 | 17,5 | 20,4 | 25,4 | 25,9 | 27,8 | **29,0** |

El tamaño del primer frente **crece monótonamente** con la dimensión hasta absorber casi toda la población. Es la firma típica de la maldición de la dimensionalidad: a partir de ~5–6 objetivos, el frente deja de discriminar.

**Frase para el tutor:** *Sobre el vector completo (16-D) la dominancia desaparece porque el 98,9% de los pares son incomparables (cada configuración gana en ~8 de 16 dimensiones); para dominar haría falta ganar en las 16 a la vez, lo que casi nunca ocurre. La curva tamaño-del-frente vs nº de objetivos lo demuestra empíricamente.*

---

## 6. SOLUCIÓN: reducir a las 6 familias (la dominancia reaparece)

Se colapsan las 16 sub-métricas a **6 dimensiones**, una por familia, promediando las variantes (género/profesión, inter/intra) ya normalizadas:

```
SS        = media(norm SS_inter_gen, SS_intra_gen, SS_inter_prof, SS_intra_prof)
LMS       = media(norm LMS_inter, LMS_intra)
ICAT      = media(norm ICAT_inter_gen, ICAT_intra_gen, ICAT_inter_prof, ICAT_intra_prof)
STS       = media(norm STS_gen, STS_soc)
Regard    = media(norm regard_gen, regard_prof)
RegardGap = media(norm rg_gap_gen, rg_gap_prof)
```

### Resultado en 6-D

| Métrica | Valor |
|---|---|
| Capas NSGA-II | **3** (tamaños **16 + 7 + 7**) |
| 1er frente (no dominadas) | **16 de 30** |
| Varianza PCA (PC1+PC2) | **81,2%** (PC1 49,4% + PC2 31,8%) |

Ahora **sí hay gradiente de dominancia** (3 capas). El scatter `reducido/figuras/pareto_pca_scatter_6d.png` muestra:
- GPT-2 *dataset+prompt* en rango 1 (no dominadas), a la derecha (PC1 alto).
- Las Llama caen mayoritariamente a rangos 2–3.

### Loadings del PCA reducido (qué significan los ejes)

| Familia | PC1 | PC2 |
|---|---|---|
| SS | 0,532 | −0,089 |
| LMS | 0,386 | 0,428 |
| ICAT | 0,540 | 0,235 |
| STS | 0,414 | −0,215 |
| Regard | 0,325 | −0,490 |
| RegardGap | 0,005 | 0,683 |

- **PC1** (49,4%): eje de "calidad global" — combina SS, ICAT, STS, LMS, Regard positivamente. Mayor PC1 ≈ mejor en casi todo.
- **PC2** (31,8%): contrasta **RegardGap/LMS** (positivo) frente a **Regard** (negativo); separa el perfil de equidad-de-tono frente a tono-medio.

---

## 7. Archivos generados

```
resultados/genero_profesion/pareto/
├── EXPLICACION_pareto_pca.md          <- este documento
├── datos/
│   ├── pareto_front.csv               (16-D, rango 1: 29 configs)
│   ├── pareto_ranking.csv             (16-D, las 30 con pareto_rank)
│   └── pareto_front_table.csv         (16-D, ordenado por PC1)
├── figuras/
│   ├── pca_loadings_heatmap.png       (16 métricas × 2 PC)
│   ├── pareto_pca_scatter.png         (16-D, familia/ruta + rango)
│   ├── pareto_layers_pca_scatter.png  (16-D, capas con barra de color)
│   └── pareto_parallel_coords.png     (perfil normalizado del frente)
├── diagnostico/
│   ├── figuras/frente_vs_dimensiones.png   <- el "por qué"
│   └── datos/frente_vs_dimensiones.csv
└── reducido/
    ├── figuras/pareto_pca_scatter_6d.png   <- dominancia visible (3 capas)
    └── datos/pareto_front_6d.csv

scripts/
├── pareto_analysis.py
└── pareto_reducido_diagnostico.py
```

---

## 8. Resumen ejecutivo (1 párrafo)

El análisis de soluciones no dominadas sobre el **vector completo de 16 dimensiones** (familias SS, LMS, ICAT, STS, Regard, RegardGap) **no discrimina**: 29 de 30 configuraciones quedan en el primer frente porque el 98,9% de los pares son mutuamente no comparables —cada configuración gana, en promedio, en solo 8 de las 16 dimensiones, y la dominancia exige ganar en las 16 a la vez—. Es la **maldición de la dimensionalidad** en optimización multiobjetivo, demostrada empíricamente con la curva tamaño-del-frente vs número de objetivos (crece de 1 a 29). El PCA reduce el espacio a 2D conservando el 66,8% de la varianza y revela estructura (ruta y familia), pero no resuelve la falta de dominancia, que es intrínseca al número de objetivos, no a la proyección. Al **reducir el problema a las 6 familias**, la dominancia **reaparece** con 3 capas claras (16/7/7) y un PCA que captura el 81,2% de la varianza, permitiendo por fin ordenar y seleccionar configuraciones.
