# Análisis focalizado: género y profesión

Carpeta dedicada a los **objetivos iniciales** de la tesis (sesgo de género y profesión).

## Estructura

```
genero_profesion/
├── analisis/
│   ├── analizar_genero_profesion.py   # Script principal
│   ├── datos/                         # CSV (incl. resumen_categorias.csv con config. del máximo)
│   └── tablas/                        # Tablas LaTeX
├── figuras/                           # PNG 300 DPI
└── README.md
```

## Regenerar resultados

```bash
python resultados/genero_profesion/analisis/analizar_genero_profesion.py
```

## Documento LaTeX

El capítulo reducido para el tutor está en la raíz del repo: **`results.tex`**.

Compilar:

```bash
pdflatex results.tex
```

## Métricas

- **M_GP**: media de 6 componentes — ΔSS inter/intra (género, profesión) + ΔSTS género + ΔSTS socioeconómico (proxy ocupación).
- **CrowS-Pairs**: género directo; **socioeconómico** como proxy de profesión/ocupación (no existe categoría homónima).
- **BOLD**: dominios género y profesión (longitud, Regard, brecha).

## Figuras generadas

| Archivo | Contenido |
|---------|-----------|
| `gp01_*` | StereoSet inter |
| `gp02_*` | StereoSet intra |
| `gp03_*` | CrowS-Pairs género + socioeconómico |
| `gp14_*` | CrowS-Pairs vs LoRA (género y socioeconómico) |
| `gp04_*` | M_GP vs LoRA |
| `gp05–06_*` | Líneas LoRA inter/intra |
| `gp07–08_*` | BOLD longitud y Regard |
| `gp11_*` | BOLD equidad (brecha) |
| `gp12–13_*` | Trade-off y ranking MEI |
| `gp_modelo_*` | Panel por modelo (5) |
