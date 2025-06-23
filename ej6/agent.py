"""
Funciones principales para ejecutar el agente de manera abstraída.
"""

import os
import google.generativeai as genai
from ej6.utils import preparar_entrada
from ej6.logger import get_logger, set_debug_mode


def ejecutar_agente(grafo, archivo_path, pregunta, api_key=None, debug=False):
    """
    Ejecuta el agente con máxima abstracción.
    
    Args:
        grafo: El grafo sin compilar construido por el usuario
        archivo_path: Ruta al archivo (imagen o cualquier otro)
        pregunta: Pregunta del usuario
        api_key: API key opcional (si no está en variables de entorno)
        debug: Si True, muestra logs detallados de la ejecución
        
    Returns:
        str: Respuesta del agente
    """
    # Configurar modo debug
    set_debug_mode(debug)
    logger = get_logger("agent")
    
    logger.debug(f"Iniciando ejecución del agente")
    logger.debug(f"Archivo: {archivo_path}")
    logger.debug(f"Pregunta: {pregunta}")
    
    # Configurar API key si se proporciona
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        logger.debug("API key configurada")
    
    try:
        # Preparar entrada
        logger.debug("Preparando entrada...")
        entrada = preparar_entrada(archivo_path, pregunta)
        logger.debug(f"Entrada preparada: archivo={entrada['entrada_usuario']['archivo']['nombre']}")
        
        # Compilar y ejecutar
        logger.debug("Compilando grafo...")
        app = grafo.compile()
        
        logger.debug("Ejecutando grafo...")
        resultado = app.invoke(entrada)
        
        # Log del estado final
        logger.debug("Estados finales:")
        for key, value in resultado.items():
            if key != "entrada_usuario":  # Evitar loggear datos grandes
                logger.debug(f"  {key}: {str(value)[:100]}...")
        
        # Retornar solo la respuesta final
        respuesta = resultado.get("respuesta_final", "No se pudo generar una respuesta")
        logger.debug(f"Respuesta final: {respuesta[:100]}...")
        
        return respuesta
        
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {archivo_path}")
        return "Error: No se encontró el archivo especificado"
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return f"Error al ejecutar el agente: {str(e)}"