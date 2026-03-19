PERFORATION STABILIZER PARA MAC
===============================

Archivos incluidos:
- perforation_stabilizer_app.py
- Perforation_Stabilizer.command

Qué hace:
- Abres el launcher .command con doble clic
- Se abre una ventana
- Arrastras la carpeta de frames o la eliges con botón
- El programa detecta la perforación izquierda
- Fija la perforación en una posición constante en toda la secuencia
- Guarda todo en una carpeta nueva automáticamente

macOS Gatekeeper — primera vez que abres la app:
macOS bloquea apps que no vienen de la App Store. Esto es normal y se resuelve una sola vez:

  Opción A (más fácil):
  1. Clic derecho sobre "Perforation Stabilizer.app"
  2. Seleccionar "Abrir" (Open)
  3. En el aviso que aparece, hacer clic en "Abrir" de nuevo
  → A partir de ahí se abre normal con doble clic

  Opción B (si la Opción A no funciona):
  1. Ir a Ajustes del Sistema → Privacidad y Seguridad
  2. Bajar hasta ver el mensaje sobre la app bloqueada
  3. Hacer clic en "Abrir de todas formas"

  Opción C (Terminal):
  xattr -cr "/ruta/a/Perforation Stabilizer.app"
  → Esto elimina los atributos de cuarentena de macOS

Si no abre por permisos en el launcher .command (versión sin empaquetar):
chmod +x /ruta/al/archivo/Perforation_Stabilizer.command

Dependencias (solo versión .command, no la .app empaquetada):
El script intenta instalar automáticamente:
- opencv-python
- numpy
- tkinterdnd2 (opcional para arrastrar carpetas)

Si la función de arrastrar carpeta no aparece, igual puedes usar el botón "Elegir".

Ajustes recomendados para tus frames:
- ROI izq.: 0.22
- Threshold: 210
- Suavizado: 9

Si notas que no detecta bien la perforación:
- baja Threshold a 200 o 195
- si detecta cosas raras, sube Threshold a 215
- si quieres que busque más pegado a la izquierda, baja ROI a 0.18 o 0.20

Salida:
La carpeta de salida se crea junto a la carpeta original con el sufijo _ESTABILIZADO.
También se genera un archivo stabilization_report.txt con resumen del proceso.
