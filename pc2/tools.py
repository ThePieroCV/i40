"""
Herramientas (Tools) para el agente de asistencia técnica.

Este módulo contiene la herramienta principal de RAG (Retrieval Augmented Generation)
que permite buscar información específica en la base de conocimiento.
"""

import os
import google.generativeai as genai
from typing import Optional
from pc2.logger import get_logger
from pc2.models import TipoEquipo


def buscar_informacion_tecnica(equipo: TipoEquipo, pregunta_usuario: str) -> str:
    """
    Herramienta RAG que busca información técnica específica sobre un equipo.
    
    Esta función:
    1. Usa el enum TipoEquipo para obtener directamente el archivo correspondiente
    2. Lee el contenido del archivo
    3. Usa un LLM para extraer la información relevante según la pregunta
    
    Args:
        equipo: Enum TipoEquipo que identifica el equipo
        pregunta_usuario: Pregunta específica del usuario sobre el equipo
        
    Returns:
        str: Respuesta extraída de la documentación técnica
    """
    logger = get_logger("tools.buscar_info")
    
    logger.debug(f"Buscando información para equipo: {equipo.name}")
    logger.debug(f"Archivo correspondiente: {equipo.value}")
    logger.debug(f"Pregunta: '{pregunta_usuario}'")
    
    # Verificar que el equipo sea válido (no EQUIPO_NO_IDENTIFICADO)
    if equipo == TipoEquipo.EQUIPO_NO_IDENTIFICADO:
        logger.warning("Equipo no identificado, no se puede buscar información")
        return "No se pudo identificar el equipo en la imagen."
    
    # Construir la ruta al archivo directamente desde el enum
    base_dir = os.path.dirname(os.path.abspath(__file__))
    archivo_path = os.path.join(base_dir, "knowledge_base", equipo.value)
    logger.debug(f"Ruta completa del archivo: {archivo_path}")
    
    # Verificar si el archivo existe
    if not os.path.exists(archivo_path):
        logger.error(f"Archivo no existe: {archivo_path}")
        return f"El archivo de documentación {equipo.value} no fue encontrado."
    
    # Leer el contenido del archivo
    try:
        with open(archivo_path, 'r', encoding='utf-8') as f:
            contenido = f.read()
        logger.debug(f"Archivo leído exitosamente, longitud: {len(contenido)} caracteres")
    except Exception as e:
        logger.error(f"Error al leer archivo: {str(e)}")
        return f"Error al leer el archivo: {str(e)}"
    
    # Usar Gemini para extraer la información relevante
    try:
        logger.debug("Enviando contenido a Gemini para procesamiento...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Basándote ÚNICAMENTE en el siguiente texto de documentación técnica, 
        responde a la pregunta del usuario de manera precisa y concisa.
        
        Si la información solicitada no está en el texto, indica claramente que 
        no está disponible en la documentación.
        
        Pregunta del usuario: {pregunta_usuario}
        
        Documentación técnica:
        {contenido}
        
        Respuesta:
        """
        
        response = model.generate_content(prompt)
        respuesta = response.text.strip()
        logger.debug(f"Respuesta de Gemini recibida, longitud: {len(respuesta)} caracteres")
        return respuesta
        
    except Exception as e:
        logger.error(f"Error al procesar con Gemini: {str(e)}")
        return f"Error al procesar la información: {str(e)}"


def listar_equipos_disponibles() -> list:
    """
    Función auxiliar que lista todos los equipos disponibles en la base de conocimiento.
    
    Returns:
        list: Lista de nombres de equipos disponibles
    """
    logger = get_logger("tools.listar_equipos")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(base_dir, "knowledge_base")
    logger.debug(f"Directorio de knowledge base: {knowledge_base_dir}")
    
    equipos = []
    if os.path.exists(knowledge_base_dir):
        archivos_encontrados = os.listdir(knowledge_base_dir)
        logger.debug(f"Archivos encontrados: {archivos_encontrados}")
        
        for archivo in archivos_encontrados:
            if archivo.endswith('.txt'):
                # Convertir nombre de archivo a nombre legible
                nombre = archivo.replace('.txt', '').replace('_', ' ').title()
                equipos.append(nombre)
                logger.debug(f"Equipo agregado: {nombre}")
    else:
        logger.error(f"Directorio no existe: {knowledge_base_dir}")
    
    logger.debug(f"Total de equipos encontrados: {len(equipos)}")
    return equipos