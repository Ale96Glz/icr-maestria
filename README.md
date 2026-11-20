# ICR — revisión de la tesis

Carpeta de trabajo para cerrar la tesis a partir del feedback del tutor. El texto que se entrega es un solo archivo LaTeX; las cifras del capítulo de resultados salen de `resultados/`.

## Documento final

| Qué | Dónde |
|---|---|
| Fuente | [`ICR Final.tex`](ICR%20Final.tex) |
| PDF | [`ICR Final.pdf`](ICR%20Final.pdf) |
| Bibliografía que usa hoy la tesis | [`bibliografia_icr.bib`](bibliografia_icr.bib) |
| Bibliografía con `year` y `journal` para IEEE | [`bibcorregida.bib`](bibcorregida.bib) |

`ICR Final.tex` es el documento completo (resumen, capítulos y bibliografía). No lo sustituyen las piezas sueltas de `picada/` ni la guía de `tutor/`.

Compilar desde esta carpeta:

```text
pdflatex "ICR Final.tex"
bibtex "ICR Final"
pdflatex "ICR Final.tex"
pdflatex "ICR Final.tex"
```

`bibcorregida.bib` todavía no está enlazada en la tesis. Para usarla, el comando de bibliografía de `ICR Final.tex` tiene que apuntar a `bibcorregida` en lugar de `bibliografia_icr`. El archivo original no se modificó. Con el estilo `IEEEtran`, la copia corregida compila sin los avisos de año o revista vacíos: el `.bib` de Zotero guardaba el año en `date` y la revista en `journaltitle`, y ese estilo no lee esos campos.

## Carpeta de resultados

Todo lo numérico que sostiene el capítulo 5 está en [`resultados/`](resultados/).

| Ruta | Para qué sirve |
|---|---|
| [`resultados/dataset_mitigacion_tesis.csv`](resultados/dataset_mitigacion_tesis.csv) | Tabla maestra: 60 filas, 30 configuraciones (cada una evaluada en StereoSet/CrowS-Pairs y en BOLD). El diccionario de columnas está en [`resultados/README.md`](resultados/README.md). |
| [`resultados/genero_profesion/`](resultados/genero_profesion/) | Análisis que cita el capítulo 5: género y profesión. CSV en `analisis/datos/`, tablas LaTeX en `analisis/tablas/`, figuras en `figuras/`. |
| `resultados/genero_profesion/analisis/datos/metricas_genero_profesion.csv` | Cifras de referencia del resumen (por ejemplo ΔSS inter género 8.68 p.p. y profesión 7.50 p.p. en GPT-2 Medium, dataset + prompt, 1 % LoRA; \(M_{GP} \approx 4.15\)). |
| `resultados/figuras/` y exports globales antiguos | Pipeline previo (la media tipo 9.40 p.p.). No mezclar esas cifras con el capítulo 5. |

Los experimentos no se entrenan dentro de esta carpeta. El entrenamiento está en `D:\Tesis\debias_lora` y la evaluación en `D:\Tesis\bias_eval`. Cada configuración es un modelo (GPT-2 Base, Medium o XL, o Llama 3.2 de 1B o 3B), un porcentaje de LoRA (1, 2 o 3 %) y una técnica (solo dataset, o dataset + prompt).

El detalle para repetir los 30 entrenamientos ($r$, $\alpha$, hiperparámetros, corpus, prompts y parámetros de BOLD) está en [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md).

## Qué detectó el tutor

El contraste con `ICR Final.tex` y con `resultados/genero_profesion/` confirma los señalamientos. El detalle, el borrador del resumen y el procedimiento de los experimentos están en [`tutor/correciones.tex`](tutor/correciones.tex) (PDF: [`tutor/correciones.pdf`](tutor/correciones.pdf)).

1. **El resumen no coincide con el capítulo 5.** Sigue cifras de un export global anterior (9.40 p.p., LMS −6.6, ICAT +11.0). El análisis vigente es el de género y profesión.
2. **BOLD en el resumen habla de raza.** El capítulo 5 y las figuras GP solo analizan género y profesión. Raza se evaluó en la infraestructura, pero no entra en ese análisis.
3. **Llama y el prompt.** Decir que «el patrón se invirtió» simplifica de más. En las medias del barrido, dataset + prompt sigue mejorando frente a solo dataset; el comportamiento por categoría sí es asimétrico.
4. **«Evaluación multimodal».** Los tres benchmarks son de texto. La redacción que corresponde es multi-benchmark o protocolo complementario.
5. **Referencias rotas en resultados.** En la sección 5.5.1 aparecen «Tabla 1», «Figura 1» y «Tabla 3» en lugar de `\ref` a las etiquetas reales.
6. **GPT-2.** El texto habla de cuatro escalas; los experimentos usan tres: Base, Medium y XL.
7. **Tiempo verbal.** El capítulo 5 ya narra experimentos hechos. El capítulo 4 y la sección 1.3 todavía están en futuro (*se empleará*, *se ejecutarán*).
8. **Reproducibilidad.** Falta en el capítulo 4 la ficha del procedimiento: qué script, qué corpus, hiperparámetros y cómo se pasa del adaptador al CSV. Esos datos están en los dos repositorios de arriba; falta anotar, por cada uno de los 30 entrenamientos, el script usado y el par \((r, \alpha)\).
9. **Bibliografía.** BibTeX avisa año y revista vacíos. La ficha ya estaba en `date` y `journaltitle`; la copia lista para IEEE es `bibcorregida.bib`.

Esos cambios todavía no están aplicados en `ICR Final.tex`. La guía de `tutor/` es la hoja de ruta para hacerlo.
