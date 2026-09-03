# Forensic Media Search

Aplicación de consola para analizar recursivamente imágenes dentro de un directorio o disco y buscar coincidencias visuales a partir de una o más consultas de texto utilizando CLIP.

El sistema procesa las imágenes con GPU, muestra las coincidencias por consola y genera un archivo CSV.

Ejemplo de salida:

```text
FilePath | % | Cos | MatchedQuery
```

> El valor `%` es un score heurístico de coincidencia. No representa una probabilidad estadística.  
> `Cos` corresponde a la similitud coseno calculada por el modelo.

---

## 1. Requisitos de la computadora

### Recomendado

- Windows 10/11 de 64 bits
- CPU: 8 núcleos o superior
- RAM: 16 GB mínimo, 32 GB recomendado
- GPU NVIDIA con soporte CUDA
- VRAM:
  - 8 GB mínimo
  - 16 GB recomendado
- Docker Desktop
- WSL2 habilitado
- Driver NVIDIA actualizado
- Espacio libre:
  - al menos 10 GB para Docker, PyTorch y modelos
  - espacio adicional según el tamaño de la evidencia y los reportes

Configuración utilizada durante el desarrollo:

```text
GPU: NVIDIA GeForce RTX 4080
VRAM: 16 GB
PyTorch: 2.12.1
CUDA Runtime: 13.2
Modelo inicial: OpenAI CLIP ViT-B/32
```

### Verificar GPU

Desde CMD:

```cmd
nvidia-smi
```

Para comprobar que Docker puede utilizar la GPU:

```cmd
docker run --rm --gpus all ubuntu nvidia-smi
```

La GPU NVIDIA debe aparecer dentro del contenedor.

---

## 2. Instalación con Docker

### 2.1 Instalar Docker Desktop

Instalar Docker Desktop para Windows y habilitar:

```text
Use the WSL 2 based engine
```

Docker debe estar iniciado antes de ejecutar la aplicación.

### 2.2 Clonar o copiar el proyecto

Ejemplo:

```cmd
cd C:\Repos
git clone <URL_DEL_REPOSITORIO> forensic-media-search
cd forensic-media-search
```

Si el proyecto ya fue copiado manualmente:

```cmd
cd C:\Repos\forensic-media-search
```

### 2.3 Construir la imagen

Ejecutar:

```cmd
docker build -t forensic-media-search:dev .
```

La primera compilación puede tardar varios minutos porque descarga Python, PyTorch, CUDA y CLIP.

### 2.4 Crear directorios locales

Desde la raíz del proyecto:

```cmd
mkdir models
mkdir output
```

`models` se utiliza como caché de modelos.

`output` contiene los archivos CSV generados.

---

## 3. Ejecución desde CMD

La aplicación recibe:

```text
--directory       Directorio montado dentro del contenedor
--query           Consulta visual. Puede repetirse varias veces
--min-percent     Score mínimo de coincidencia
--batch-size      Cantidad de imágenes procesadas por lote
--model           Modelo CLIP
--output          Archivo CSV de salida
--display-root    Ruta original que debe mostrarse en el informe
```

### Ejemplo

Analizar:

```text
D:\DiscoPeritado
```

Buscando:

```text
a house with a red roof
a house with a black door
```

Ejecutar desde CMD:

```cmd
docker run --rm --gpus all ^
  --mount type=bind,source="D:\DiscoPeritado",target=/evidence,readonly ^
  --mount type=bind,source="%cd%\output",target=/output ^
  --mount type=bind,source="%cd%\models",target=/root/.cache/clip ^
  forensic-media-search:dev ^
  --directory /evidence ^
  --display-root "D:\DiscoPeritado" ^
  --query "a house with a red roof" ^
  --query "a house with a black door" ^
  --min-percent 75 ^
  --batch-size 64 ^
  --model "ViT-B/32" ^
  --output /output/report.csv
```

El directorio de evidencia se monta como:

```text
readonly
```

por lo que el contenedor no puede modificar los archivos analizados.

### Resultado por consola

Ejemplo:

```text
91.82% | cos=0.321386 | D:\DiscoPeritado\DCIM\IMG_4821.jpg
         | a house with a red roof

86.47% | cos=0.302653 | D:\DiscoPeritado\Pictures\IMG_1932.jpg
         | a house with a black door
```

### Archivo CSV

Al finalizar se genera:

```text
output\report.csv
```

Ejemplo:

```csv
FilePath,%,Cos,MatchedQuery
D:\DiscoPeritado\DCIM\IMG_4821.jpg,91.82,0.321386,a house with a red roof
D:\DiscoPeritado\Pictures\IMG_1932.jpg,86.47,0.302653,a house with a black door
```

---

## Consideraciones

- El análisis es recursivo.
- Las consultas múltiples se evalúan con semántica OR: se conserva la mejor coincidencia de cada imagen.
- Los archivos que no puedan abrirse se informan como error y el análisis continúa.
- La evidencia debe mantenerse en modo solo lectura.
- Los resultados generados deben guardarse fuera del directorio analizado.
- El score del modelo se utiliza para seleccionar candidatos que luego deben ser revisados por un analista/perito.
