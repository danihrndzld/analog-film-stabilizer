PERFORATION STABILIZER PARA MAC
===============================

Estabilizador automático de perforaciones para secuencias de película analógica escaneada.

---

## Cómo descargar e instalar

### Opción recomendada: descargar desde GitHub Releases

1. Ve a la página del repositorio en GitHub
2. En la columna derecha, haz clic en **Releases**
3. Descarga el archivo `Perforation_Stabilizer_macOS.zip` de la versión más reciente
4. Descomprime el ZIP — obtendrás `Perforation Stabilizer.app`
5. Arrastra la app a tu carpeta `/Aplicaciones`

### Primera vez que abres la app (macOS Gatekeeper)

macOS bloquea apps que no vienen de la App Store. Esto es normal y se resuelve una sola vez:

**Opción A (más fácil):**
1. Clic derecho sobre "Perforation Stabilizer.app"
2. Seleccionar "Abrir" (Open)
3. En el aviso que aparece, hacer clic en "Abrir" de nuevo
→ A partir de ahí se abre normal con doble clic

**Opción B (si la Opción A no funciona):**
1. Ir a Ajustes del Sistema → Privacidad y Seguridad
2. Bajar hasta ver el mensaje sobre la app bloqueada
3. Hacer clic en "Abrir de todas formas"

**Opción C (Terminal):**
```
xattr -cr "/Applications/Perforation Stabilizer.app"
```
→ Esto elimina los atributos de cuarentena de macOS

---

## Qué hace la app

- Arrastras la carpeta de frames o la eliges con el botón
- El programa detecta automáticamente la perforación izquierda en cada frame
- Fija la perforación en una posición constante a lo largo de toda la secuencia
- Recorta los bordes negros automáticamente — todos los frames de salida tienen el mismo tamaño
- Guarda la secuencia estabilizada en una carpeta nueva junto a la original

---

## Flujo de uso

1. Abre la app
2. Arrastra la carpeta con tus frames (o usa "Elegir carpeta")
3. Ajusta los parámetros si es necesario (ver abajo)
4. Elige la calidad de salida: JPEG (1–100) o PNG lossless
5. Haz clic en "Estabilizar"
6. La carpeta de salida aparece junto a la original con el sufijo `_ESTABILIZADO`

---

## Cómo funciona la estabilización

1. **Primera pasada** — detecta la posición de la perforación en cada frame
2. Rechaza detecciones atípicas (outliers a más de 5×MAD o mínimo 80 px) e interpola esos frames desde los vecinos
3. Calcula la **mediana** de las posiciones válidas como punto fijo destino
4. **Segunda pasada** — traslada cada frame para que su perforación quede exactamente en el punto fijo
5. Recorta los bordes negros automáticamente — todos los frames de salida tienen el mismo tamaño y sin bordes

---

## Ajustes de parámetros

| Parámetro | Valor por defecto | Para qué sirve |
|-----------|-------------------|----------------|
| ROI izq.  | 0.22              | Fracción del ancho a buscar desde la izquierda. Bájalo si la perforación está muy pegada al borde |
| Threshold | 210               | Corte de brillo para detectar la perforación. Bájalo si no detecta bien; súbelo si detecta cosas raras |
| Suavizado | 9                 | Radio de la ventana de promedio móvil para suavizar posiciones |

**Si no detecta bien la perforación:**
- Baja Threshold a 200 o 195
- Si detecta falsos positivos, sube Threshold a 215
- Si quieres buscar más pegado al borde izquierdo, baja ROI a 0.18 o 0.20

---

## Calidad de salida

- **JPEG**: elige un valor de 1 a 100. Para archivo de trabajo usa 90–95; para máxima calidad usa 100
- **PNG**: lossless, sin pérdida de calidad, pero archivos más grandes

---

## Salida generada

- **Carpeta `_ESTABILIZADO`** junto a la carpeta original con los frames estabilizados
- **`stabilization_report.txt`** con:
  - Total de frames procesados
  - Detecciones fallidas (frames interpolados)
  - Coordenadas del punto fijo (anchor)
  - Dimensiones de salida
  - Valores de recorte aplicados (left/right/top/bottom en píxeles)

---

## Versión de script (sin empaquetar)

Si prefieres correr directamente el script Python:

```bash
python3 src/perforation_stabilizer_app.py
```

O con doble clic en `src/Perforation_Stabilizer.command`.

Las dependencias se instalan automáticamente en el primer arranque:
- `opencv-python`
- `numpy`
- `tkinterdnd2` (opcional, para arrastrar carpetas)

Si el launcher está bloqueado por permisos:
```bash
chmod +x src/Perforation_Stabilizer.command
```
