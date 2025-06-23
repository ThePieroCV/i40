"""
Utilidades para el agente de asistencia técnica.

Este módulo contiene funciones auxiliares para el procesamiento de imágenes
y la preparación de datos para el agente.
"""

import base64
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Union
import os


def preparar_entrada(archivo_path: str, pregunta: str) -> Dict[str, Any]:
    """
    Prepara la entrada del usuario en el formato esperado por el agente.
    
    Args:
        archivo_path: Ruta al archivo (puede ser imagen o cualquier otro tipo)
        pregunta: Pregunta del usuario
        
    Returns:
        dict: Entrada formateada para el agente
    """
    # Verificar si el archivo existe
    if not os.path.exists(archivo_path):
        raise FileNotFoundError(f"No se encontró el archivo: {archivo_path}")
    
    # Leer el archivo
    with open(archivo_path, 'rb') as f:
        contenido_archivo = f.read()
    
    # Obtener el nombre del archivo
    nombre_archivo = os.path.basename(archivo_path)
    
    return {
        "entrada_usuario": {
            "pregunta": pregunta,
            "archivo": {
                "nombre": nombre_archivo,
                "contenido": contenido_archivo
            }
        }
    }


def imagen_a_base64(imagen_path: str) -> str:
    """
    Convierte una imagen a formato base64 para transmisión.
    
    Args:
        imagen_path: Ruta al archivo de imagen
        
    Returns:
        str: Imagen codificada en base64
    """
    with open(imagen_path, 'rb') as f:
        imagen_bytes = f.read()
    
    # Detectar el tipo de imagen
    if imagen_path.lower().endswith('.png'):
        mime_type = 'image/png'
    elif imagen_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
    elif imagen_path.lower().endswith('.gif'):
        mime_type = 'image/gif'
    else:
        mime_type = 'image/jpeg'  # Default
    
    # Codificar en base64
    base64_string = base64.b64encode(imagen_bytes).decode('utf-8')
    
    # Formato data URI
    return f"data:{mime_type};base64,{base64_string}"


def validar_imagen(archivo_path: str) -> bool:
    """
    Valida si un archivo es una imagen válida.
    
    Args:
        archivo_path: Ruta al archivo
        
    Returns:
        bool: True si es una imagen válida, False en caso contrario
    """
    tipos_validos = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    extension = os.path.splitext(archivo_path)[1].lower()
    
    if extension not in tipos_validos:
        return False
    
    # Intentar abrir la imagen para verificar que es válida
    try:
        Image.open(archivo_path)
        return True
    except:
        return False


def redimensionar_imagen(imagen: Union[Image.Image, bytes], max_size: int = 1024) -> Image.Image:
    """
    Redimensiona una imagen manteniendo su relación de aspecto.
    
    Args:
        imagen: Imagen PIL o bytes
        max_size: Tamaño máximo en píxeles para el lado más largo
        
    Returns:
        Image.Image: Imagen redimensionada
    """
    # Convertir a imagen PIL si es necesario
    if isinstance(imagen, bytes):
        imagen = Image.open(BytesIO(imagen))
    
    # Obtener dimensiones actuales
    ancho, alto = imagen.size
    
    # Si ya es más pequeña, no hacer nada
    if max(ancho, alto) <= max_size:
        return imagen
    
    # Calcular nueva escala
    if ancho > alto:
        nueva_ancho = max_size
        nueva_alto = int(alto * (max_size / ancho))
    else:
        nueva_alto = max_size
        nueva_ancho = int(ancho * (max_size / alto))
    
    # Redimensionar
    return imagen.resize((nueva_ancho, nueva_alto), Image.Resampling.LANCZOS)


def formatear_estado_para_debug(state: Dict[str, Any]) -> str:
    """
    Formatea el estado del agente para visualización en debug.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        str: Estado formateado de manera legible
    """
    lineas = ["=" * 50, "ESTADO ACTUAL DEL AGENTE", "=" * 50]
    
    for clave, valor in state.items():
        if clave == "entrada_usuario" and isinstance(valor, dict):
            lineas.append(f"\n{clave}:")
            lineas.append(f"  - pregunta: {valor.get('pregunta', 'N/A')}")
            if 'archivo' in valor:
                lineas.append(f"  - archivo: {valor['archivo'].get('nombre', 'N/A')}")
        elif valor is not None:
            # Truncar valores muy largos
            valor_str = str(valor)
            if len(valor_str) > 100:
                valor_str = valor_str[:97] + "..."
            lineas.append(f"\n{clave}: {valor_str}")
    
    lineas.append("\n" + "=" * 50)
    return "\n".join(lineas)


def leer_base_conocimiento() -> Dict[str, str]:
    """
    Lee todos los archivos de la base de conocimiento y retorna su contenido.
    
    Returns:
        dict: Diccionario con nombre de archivo como clave y contenido como valor
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(base_dir, "knowledge_base")
    
    archivos = {}
    
    if os.path.exists(knowledge_base_dir):
        for archivo in os.listdir(knowledge_base_dir):
            if archivo.endswith('.txt'):
                archivo_path = os.path.join(knowledge_base_dir, archivo)
                try:
                    with open(archivo_path, 'r', encoding='utf-8') as f:
                        archivos[archivo] = f.read()
                except Exception as e:
                    pass  # Silenciosamente ignorar errores
    
    return archivos