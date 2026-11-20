# Reproducibilidad de los 30 entrenamientos

Documento de trabajo para el capítulo 4 de la tesis. Recoge los datos que pidió el tutor:
**rango $r$, $\alpha$, tasa de aprendizaje, épocas, tamaño de los corpus, texto del prompt y
parámetros de generación de BOLD**. Los valores están verificados en el código, no copiados de la
guía de `tutor/`.

## 1. Fuentes y versión

| Elemento | Valor |
|---|---|
| Repo de entrenamiento | `https://github.com/Ale96Glz/debias_lora.git` (local: `D:\Tesis\debias_lora`) |
| Commit citado | `74380a9` (*Acutalizar repo*) |
| Entorno | Python ≥ 3.11, `uv`, `uv sync` con `uv.lock` versionado |
| GPU | RTX 4060 8 GB (local) / RTX 3070 24 GB (laboratorio); QLoRA NF4 de 4 bits para Llama |
| Repo de evaluación | `D:\Tesis\bias_eval` (`GenerationConfig` de BOLD está aquí, no en `debias_lora`) |

**Pendiente:** el URL de GitHub devuelve 404 (repo privado o no accesible). Hacerlo público o
depositar un snapshot con DOI (Zenodo) y citar el hash del commit; si no, el revisor no puede
verificar nada.

## 2. Rango $r$ y $\alpha$ (15 combinaciones modelo × porcentaje)

Calculado con `utils/config_detect.py`. Regla implementada:

$$r=\operatorname{round}\!\left(\frac{\% \cdot N_{\text{total}}}{\sum (d_{\text{in}}+d_{\text{out}})}\right),\qquad \alpha = 2r,\qquad \text{dropout}=0.05$$

| Modelo | 1 % ($r/\alpha$; % real) | 2 % ($r/\alpha$; % real) | 3 % ($r/\alpha$; % real) |
|---|---|---|---|
| GPT-2 Base | 8/16; 0.948 % | 17/34; 2.014 % | 25/50; 2.962 % |
| GPT-2 Medium | 9/18; 0.997 % | 18/36; 1.995 % | 27/54; 2.992 % |
| GPT-2 XL | 13/26; 1.026 % | 25/50; 1.972 % | 38/76; 2.998 % |
| Llama 3.2-1B Instruct | 18/36; 1.026 % | 35/70; 1.995 % | 53/106; 3.021 % |
| Llama 3.2-3B Instruct | 21/42; 0.993 % | 42/84; 1.987 % | 63/126; 2.980 % |

- Módulos objetivo: GPT-2 → `c_attn`, `c_proj`, `c_fc`; Llama → `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- El **% real difiere del nominal** porque $r$ se redondea a entero; conviene declararlo.
- Nota de método: las 12 primeras filas se obtuvieron ejecutando `config_detect` con pesos
  completos; las de Llama 3B se calcularon con la misma fórmula sobre meta-dispositivo, método
  validado porque reproduce exactamente las tres filas de Llama 1B.

## 3. Hiperparámetros de entrenamiento

| Parámetro | `train/train.py` (`debias`: dataset + prompt) | `train_model/train_lora_mitigation.py` (`cda`: solo dataset) |
|---|---|---|
| Épocas | 5 | 3 |
| LR (AdamW) | $5\times10^{-4}$ | $5\times10^{-4}$ |
| Batch por GPU | 8 | 8 |
| Acumulación de gradiente | 4 | 1 (implícito) |
| `max_length` | 128 | 128 |
| Precisión | fp16/bf16; QLoRA NF4 para Llama | idem |
| Guardado | por época | por época |
| Semilla | 42 (default de `TrainingArguments`; no declarada explícitamente) | 42 (ídem) |

## 4. Tamaño de los corpus (conteo real de `debias_lora/data/`)

| Archivo | $N$ | Uso |
|---|---|---|
| `gender_debias_lora.json` | 2000 pares femenino/masculino | ruta `debias` (STEM + prompt) |
| `gender_debias_dda.json` | 1000 pares | variante |
| `gender_debias_names.json` | 1000 pares | variante con nombres |
| `gender_mitigation_train.jsonl` | 1982 textos (columna `text`) | ruta `cda` |

Los cuatro archivos están versionados en el repo de entrenamiento.

## 5. Texto del prompt (literal del código)

- **`plain`** (`train/train.py:60`): `Context: {input_sentence} => {output_sentence}`.
  La pérdida se calcula solo a partir de `=>` (tokens anteriores en −100).
- **`chat`** (`train/train.py:85`): usuario → `Rewrite the following sentence by swapping the
  gender. Sentence: {input_sentence}`; asistente → `{output_sentence}`. Loss en los tokens del
  asistente cuando el tokenizador expone la máscara.
- **CDA** (`train_model/train_lora_mitigation.py`): secuencia `text` + EOS, sin prefijo.

La plantilla `plain`/`chat` **no es un flag de CLI**: se lee de `.env`
(`TRAIN_TOKENIZER_STYLE`) y `.env` no está versionado. Es el principal hueco de reproducibilidad.

## 6. Parámetros de generación de BOLD (`bias_eval`)

```text
GenerationConfig(do_sample=True, top_p=0.9, temperature=0.7,
                 num_return_sequences=1, pad_token_id=eos,
                 max_new_tokens=20 por defecto en config.py)
```

- Dominios por defecto: `gender`, `race`, `profession` (el capítulo 5 solo reporta género y
  profesión).
- Clasificador de tono: `sasha/regardv3`. Lote 8. `num_samples=None` → todo el subconjunto
  filtrado por dominio.
- **Ambigüedad:** `config.py:21` define 20 tokens (valor efectivo al llamar desde `evaluation.py`
  sin `--bold-max-new-tokens`), pero `benchmarks/bold.py:51` define `BOLD_MAX_NEW_TOKENS = 30`
  como default del módulo. No hay forma de saber desde los datos cuál se usó en cada corrida:
  declarar uno en la tesis y anotarlo en la bitácora.

## 7. Comandos de repetición

```bash
uv sync
uv run python -m utils.config_detect gpt2-medium 1.0   # imprime r, α, % real y módulos
uv run python -m train.train gpt2-medium 1.0           # técnica 'debias' (dataset + prompt)
uv run python train_model/train_lora_mitigation.py gpt2-medium 1.0   # técnica 'cda'
# Llama: añadir --qlora (NF4, 4 bits)
```

Nota: hay 5 modelos × 3 porcentajes × 2 técnicas = **30 entrenamientos**; cada uno se evalúa en
StereoSet+CrowS-Pairs y en BOLD = 60 filas de `dataset_mitigacion_tesis.csv` (30 `exp_group`).

## 8. Párrafo listo para el capítulo 4

> Los adaptadores se entrenaron con `debias_lora` (commit `74380a9`, Python ≥ 3.11, `uv.lock`).
> Cada una de las 30 configuraciones combinó un modelo base, un porcentaje de parámetros
> entrenables (1, 2 o 3 %) y una técnica (`cda` o `debias`). El rango $r$ no se fijó manualmente:
> `utils/config_detect.py` lo derivó del porcentaje sobre el número total de parámetros del modelo
> en FP32, con $\alpha = 2r$ y dropout 0.05 (Tabla X). Las rutas `debias` y `cda` usaron
> `train/train.py` (5 épocas, acumulación 4, plantilla `Context: … => …` con pérdida solo en la
> continuación) y `train_model/train_lora_mitigation.py` (3 épocas, JSONL de 1982 textos),
> respectivamente, con tasa de aprendizaje $5\times10^{-4}$, lote 8 y longitud 128 en ambos casos.
> La evaluación BOLD generó una secuencia por prompt con `do_sample=True`, $T=0.7$, `top_p=0.9` y
> 20 tokens nuevos, puntuando el tono con `sasha/regardv3`.

## 9. Para cerrar el punto por completo

Crear `debias_lora/bitacora.csv` con 30 filas y estas columnas:

```text
modelo, pct, tecnica, script, dataset, plantilla, r, alpha,
max_new_tokens_bold, semilla, ruta_adaptador, timestamp
```

## 10. Límites conocidos

- No se puede saber qué plantilla (`plain`/`chat`) ni cuál de los tres datasets de pares se usó en
  cada una de las 30 carpetas: el código admite varias rutas y no hay `.env` ni logs versionados.
- `model_output/` está en `.gitignore` y `bias_eval/adapters/` solo contiene `.gitkeep`: los 30
  adaptadores no están publicados. Repetir la evaluación exige reentrenar primero.
- BOLD usa muestreo (`do_sample=True`): la generación no es bit-a-bit reproducible aunque el
  adaptador lo sea.
