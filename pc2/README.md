# 🤖 Agente de Asistencia Técnica

Un agente inteligente basado en LangGraph que puede identificar equipos técnicos en imágenes y responder preguntas específicas sobre ellos.

## 📋 Características

- **Visión por Computadora**: Identifica equipos técnicos en imágenes usando Gemini Vision
- **RAG (Retrieval Augmented Generation)**: Busca información técnica en una base de conocimiento
- **Arquitectura Modular**: Código organizado en módulos reutilizables
- **Grafo de Estados**: Flujo de trabajo claro y visualizable con LangGraph

## 🗂️ Estructura del Proyecto

```
agente_tecnico/
│
├── __init__.py              # Módulo principal
├── state.py                 # Definición del estado del agente
├── tools.py                 # Herramientas (RAG)
├── nodes.py                 # Nodos del grafo
├── graph.py                 # Construcción del grafo
│
├── knowledge_base/          # Base de conocimiento
│   ├── tomacorriente_industrial.txt
│   ├── robotino_festo.txt
│   ├── compresor_aire.txt
│   ├── ventilador_industrial.txt
│   ├── proyector_aula.txt
│   └── senal_riesgo_electrico.txt
│
├── setup_knowledge_base.py  # Script para crear la base de conocimiento
├── requirements.txt         # Dependencias
├── demo_agente_tecnico.ipynb # Notebook de demostración
└── README.md               # Este archivo
```

## 🚀 Instalación

1. **Clonar o descargar el proyecto**

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Configurar API Key de Google**:
```python
import os
os.environ["GOOGLE_API_KEY"] = "tu-api-key-aqui"
```

4. **Crear la base de conocimiento**:
```bash
python setup_knowledge_base.py
```

## 💻 Uso Básico

### Desde Python:

```python
from agente_tecnico import ejecutar_agente

# Ejecutar el agente
respuesta = ejecutar_agente(
    archivo_path="ruta/a/imagen.jpg",
    pregunta="¿Cuál es la potencia máxima de este equipo?",
    mostrar_pasos=True
)

print(respuesta)
```

### Desde el Notebook:

1. Abrir `demo_agente_tecnico.ipynb`
2. Ejecutar las celdas en orden
3. Usar la función `probar_agente()` para hacer pruebas

## 🔍 Arquitectura del Agente

El agente sigue este flujo:

1. **Validación**: Verifica que la entrada sea una imagen
2. **Identificación**: Usa visión por computadora para identificar el equipo
3. **Recuperación**: Busca información en la base de conocimiento
4. **Síntesis**: Genera una respuesta amigable
5. **Gestión de Errores**: Maneja entradas no válidas

## 📚 Base de Conocimiento

El agente puede identificar y responder preguntas sobre:

- Tomacorriente Industrial NEMA L6-20R
- Robot Móvil Robotino v3 (Festo)
- Compresor de Aire Silencioso C-200
- Ventilador Industrial V-45
- Proyector Multimedia Epson PowerLite 108
- Señal de Advertencia de Riesgo Eléctrico

## 🛠️ Personalización

### Agregar nuevos equipos:

1. Crear un archivo `.txt` en `knowledge_base/`
2. Actualizar el mapeo en `tools.py`
3. Incluir el equipo en la lista de `nodes.py`

### Modificar el comportamiento:

- **Estado**: Editar `state.py` para agregar nuevos campos
- **Nodos**: Modificar o agregar nodos en `nodes.py`
- **Flujo**: Ajustar las conexiones en `graph.py`

## 📝 Ejemplo de Uso

```python
# Ejemplo completo
from agente_tecnico import construir_grafo

# Construir el grafo
app = construir_grafo()

# Preparar entrada
entrada = {
    "entrada_usuario": {
        "pregunta": "¿Cuánta presión soporta?",
        "archivo": {
            "nombre": "compresor.jpg",
            "contenido": contenido_imagen  # bytes de la imagen
        }
    }
}

# Ejecutar
resultado = app.invoke(entrada)
print(resultado["respuesta_final"])
```

## 🐛 Solución de Problemas

### Error: "No module named 'agente_tecnico'"
- Asegúrate de estar en el directorio correcto
- Verifica que todos los archivos `__init__.py` existan

### Error: "API key not valid"
- Configura tu API key de Google correctamente
- Verifica que tengas acceso a Gemini API

### El agente no identifica el equipo
- Asegúrate de que la imagen sea clara
- Verifica que el equipo esté en la lista de equipos conocidos

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.