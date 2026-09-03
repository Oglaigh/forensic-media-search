# Forensic Media Search

Aplicación offline de consola para triage semántico de colecciones de imágenes. Recorre un directorio de evidencia de forma recursiva y devuelve candidatos para revisión humana; ningún resultado prueba que un objeto o atributo exista.

## Arquitectura

La búsqueda utiliza dos motores independientes:

1. SigLIP 2 `google/siglip2-base-patch16-224` como motor principal.
2. OpenAI CLIP `ViT-B/32` como motor secundario.

Los motores se ejecutan secuencialmente para reducir presión sobre la VRAM. Cada uno procesa el mismo manifiesto estable de archivos, pero utiliza su propio preprocessing y su propia escala de similitud.

`--top-k K` conserva K candidatos **por modelo y por query**. Con dos queries se mantienen cuatro listas independientes:

```text
SigLIP2 / query 1 -> Top K
SigLIP2 / query 2 -> Top K
CLIP    / query 1 -> Top K
CLIP    / query 2 -> Top K
```

No se aplica un `argmax` global entre queries. Una imagen puede aparecer para varias queries. Luego se unifican únicamente resultados con la misma clave `archivo + query`.

El orden final usa Reciprocal Rank Fusion:

```text
FusionScore = 1 / (60 + SigLIP2Rank) + 1 / (60 + CLIPRank)
```

Un motor que no recuperó esa combinación no aporta un término. Los scores de SigLIP2 y CLIP nunca se comparan ni combinan directamente.

## Requisitos

- Windows 10/11 con WSL2 y Docker Desktop.
- GPU NVIDIA y drivers compatibles con CUDA.
- Desarrollo validado para NVIDIA GeForce RTX 4080 de 16 GB.
- Al menos 16 GB de RAM; 32 GB recomendados.
- Espacio para la imagen Docker y los cachés de ambos modelos.

La CLI exige CUDA por defecto. Para un diagnóstico deliberado sin GPU puede indicarse `--device cpu`; nunca hay fallback silencioso.

Verificación básica:

```powershell
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
```

## Build

Desde la raíz del repositorio:

```powershell
docker build -t forensic-media-search:dev .
```

Las revisiones de Transformers, SigLIP2 y OpenAI CLIP están fijadas para mejorar la reproducibilidad.

## Ejecución

```powershell
docker run --rm --gpus all `
  --mount type=bind,source="D:\DiscoPeritado",target=/evidence,readonly `
  --mount type=bind,source="$PWD\output",target=/output `
  --mount type=bind,source="$PWD\models",target=/root/.cache/clip `
  --mount type=bind,source="$PWD\models\huggingface",target=/root/.cache/huggingface `
  forensic-media-search:dev `
  --directory /evidence `
  --display-root "D:\DiscoPeritado" `
  --query "Es un perro" `
  --query "Es un robot" `
  --top-k 1000 `
  --batch-size 64 `
  --output /output/report.csv
```

También puede utilizarse el wrapper PowerShell:

```powershell
.\run.ps1 -Directory "D:\DiscoPeritado" `
  -Query "Es un perro","Es un robot" `
  -TopK 1000 `
  -BatchSize 64
```

Para un smoke test determinista sobre las primeras 100 imágenes agregue `--max-images 100`. La consola lo identifica como un scan parcial.

## Cachés

- OpenAI CLIP: `models/` montado en `/root/.cache/clip`.
- Hugging Face/SigLIP2: `models/huggingface/` montado en `/root/.cache/huggingface`.

Los cachés persisten fuera del contenedor. Una segunda ejecución puede verificarse sin red, una vez descargados ambos modelos, agregando `--network none` y las variables `-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1` al comando Docker.

## Reporte

El CSV contiene una fila por `archivo + query`:

```csv
FilePath,OriginalQuery,MatchedQuery,SigLIP2Score,CLIPCos,SigLIP2Rank,CLIPRank,ModelsMatched,FinalRank,FusionScore
D:\Evidence\IMG001.jpg,Es un perro,Es un perro,0.312345,0.287654,12,19,SigLIP2+CLIP,1,0.027149321267
D:\Evidence\IMG001.jpg,Es un robot,Es un robot,0.221234,,820,,SigLIP2,145,0.001136363636
```

`ModelsMatched` vale `SigLIP2`, `CLIP` o `SigLIP2+CLIP`. Si un motor no incluyó la combinación dentro de su Top-K, sus campos de score y rank quedan vacíos; nunca se inventa un cero.

`SigLIP2Score` y `CLIPCos` son similitudes coseno propias de cada modelo. `FusionScore` es una medida de fusión de rankings. Ninguno representa probabilidad, confianza estadística ni porcentaje de certeza.

Junto al CSV se conservan:

- `report.csv.manifest.jsonl`: universo estable de archivos descubierto.
- `report.csv.errors.jsonl`: errores de filesystem y decoding por motor.

## Seguridad forense

- La evidencia se monta `readonly` y sólo se abre en modo lectura.
- No se modifican imágenes ni metadatos.
- No se escriben thumbnails ni temporales dentro de evidencia.
- Archivos corruptos se registran y el scan continúa.
- CSV, manifiesto, errores y cachés se escriben fuera de evidencia.
- La CLI rechaza `HF_HOME` o `CLIP_CACHE_DIR` si resuelven dentro de evidencia.
- `FilePath` conserva la ruta original mediante `--display-root` cuando se proporciona.

## Tests

Los unit tests usan adapters falsos y no descargan modelos:

```powershell
python -m pytest -q
docker compose config
```

Los tests cubren Top-K independiente por query, unión y RRF por `archivo + query`, candidatos recuperados por uno o ambos motores, empates deterministas, corruptos, campos CSV vacíos y apertura read-only de evidencia.

### Limitaciones conocidas

El discovery ordena las entradas de cada directorio para obtener un manifiesto determinista. El consumo general permanece acotado por batch, queries y Top-K, pero un único directorio plano con una cantidad extrema de archivos requiere memoria proporcional a las entradas de ese directorio durante su ordenamiento.
