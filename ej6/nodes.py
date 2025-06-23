"""
Nodos del grafo para el agente de asistencia técnica.

Este módulo contiene las funciones que representan cada nodo del grafo.
Cada función recibe el estado actual y retorna las actualizaciones necesarias.
"""

import google.generativeai as genai
from typing import Dict, Any
from ej6.state import AgentState
from ej6.tools import buscar_informacion_tecnica
from ej6.models import TipoEquipo, IdentificacionEquipo
from ej6.logger import get_logger
import base64
from io import BytesIO
from PIL import Image
import json


def validar_entrada(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que valida si la entrada del usuario contiene una imagen válida.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con es_valido=True/False
    """
    logger = get_logger("nodes.validar")
    
    entrada = state.get("entrada_usuario", {})
    archivo = entrada.get("archivo", {})
    
    logger.debug(f"Validando entrada...")
    
    # Verificar si hay un archivo
    if not archivo:
        logger.debug("No se encontró archivo en la entrada")
        return {"es_valido": False}
    
    # Verificar el tipo de archivo
    nombre_archivo = archivo.get("nombre", "").lower()
    tipos_imagen = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    logger.debug(f"Nombre de archivo: '{nombre_archivo}'")
    logger.debug(f"Tipos de imagen válidos: {tipos_imagen}")
    
    es_imagen = any(nombre_archivo.endswith(tipo) for tipo in tipos_imagen)
    
    logger.debug(f"¿Es imagen válida? {es_imagen}")
    
    return {"es_valido": es_imagen}


def identificar_equipo_vision(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que usa Gemini Vision para identificar el equipo en la imagen.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con nombre_equipo identificado
    """
    logger = get_logger("nodes.identificar")
    
    entrada = state.get("entrada_usuario", {})
    archivo = entrada.get("archivo", {})
    
    logger.debug("Iniciando identificación con Gemini Vision...")
    
    try:
        # Obtener la lista de equipos válidos del enum
        logger.debug("Obteniendo lista de equipos desde enum...")
        lista_equipos = TipoEquipo.get_lista_para_prompt()
        logger.debug(f"Equipos válidos: {lista_equipos}")
        
        # Configurar el modelo de visión
        logger.debug("Configurando modelo Gemini Vision...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Preparar la imagen
        contenido_imagen = archivo.get("contenido")
        logger.debug(f"Tipo de contenido de imagen: {type(contenido_imagen)}")
        
        # Si el contenido es bytes o base64, convertir a imagen PIL
        if isinstance(contenido_imagen, bytes):
            logger.debug("Convirtiendo bytes a imagen PIL...")
            imagen = Image.open(BytesIO(contenido_imagen))
        elif isinstance(contenido_imagen, str) and contenido_imagen.startswith('data:image'):
            logger.debug("Convirtiendo base64 a imagen PIL...")
            # Es base64, extraer solo los datos
            base64_data = contenido_imagen.split(',')[1]
            imagen_bytes = base64.b64decode(base64_data)
            imagen = Image.open(BytesIO(imagen_bytes))
        else:
            logger.debug("Usando contenido de imagen directamente...")
            imagen = contenido_imagen
        
        logger.debug(f"Imagen preparada, tamaño: {imagen.size if hasattr(imagen, 'size') else 'N/A'}")
        
        # Prompt estructurado para identificar el equipo
        prompt = f"""
        Observa esta imagen cuidadosamente y determina cuál de los siguientes equipos 
        técnicos aparece en ella.
        
        Equipos válidos:
        {lista_equipos}
        
        Responde ÚNICAMENTE con uno de estos valores exactos (sin espacios ni caracteres extra):
        - COMPRESOR_AIRE
        - ROBOTINO_FESTO
        - VENTILADOR_INDUSTRIAL
        - PROYECTOR_AULA
        - TOMACORRIENTE_INDUSTRIAL
        - SENAL_RIESGO_ELECTRICO
        - EQUIPO_NO_IDENTIFICADO (si no puedes identificar ninguno)
        
        Tu respuesta debe ser exactamente uno de estos valores, nada más.
        """
        
        logger.debug("Enviando imagen y prompt estructurado a Gemini...")
        response = model.generate_content([prompt, imagen])
        respuesta_texto = response.text.strip().upper()
        
        logger.debug(f"Respuesta raw de Gemini: '{respuesta_texto}'")
        
        # Intentar convertir la respuesta a enum
        try:
            equipo_enum = TipoEquipo[respuesta_texto]
            logger.debug(f"Equipo identificado exitosamente: {equipo_enum.name}")
            return {"nombre_equipo": equipo_enum}
        except KeyError:
            logger.warning(f"Respuesta no válida de Gemini: '{respuesta_texto}', usando EQUIPO_NO_IDENTIFICADO")
            return {"nombre_equipo": TipoEquipo.EQUIPO_NO_IDENTIFICADO}
        
    except Exception as e:
        logger.error(f"Error en identificación: {str(e)}")
        return {"nombre_equipo": TipoEquipo.EQUIPO_NO_IDENTIFICADO}


def recuperar_informacion_especifica(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que recupera información específica del equipo usando la herramienta RAG.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con info_recuperada
    """
    logger = get_logger("nodes.recuperar")
    
    equipo_enum = state.get("nombre_equipo")
    pregunta_usuario = state.get("entrada_usuario", {}).get("pregunta", "")
    
    logger.debug(f"Recuperando información para: {equipo_enum}")
    logger.debug(f"Pregunta: '{pregunta_usuario}'")
    
    # Verificar si se identificó un equipo válido
    if not isinstance(equipo_enum, TipoEquipo) or equipo_enum == TipoEquipo.EQUIPO_NO_IDENTIFICADO:
        logger.warning(f"Equipo no válido identificado: {equipo_enum}")
        return {
            "info_recuperada": "No se pudo identificar el equipo en la imagen."
        }
    
    # Usar la herramienta RAG para buscar información
    logger.debug("Llamando a la herramienta RAG...")
    informacion = buscar_informacion_tecnica(equipo_enum, pregunta_usuario)
    
    logger.debug(f"Información recuperada (primeros 100 chars): {informacion[:100]}...")
    
    return {"info_recuperada": informacion}


def sintetizar_respuesta_final(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que genera una respuesta final amigable para el usuario.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con respuesta_final
    """
    logger = get_logger("nodes.sintetizar")
    
    info_recuperada = state.get("info_recuperada", "")
    equipo_enum = state.get("nombre_equipo")
    pregunta_usuario = state.get("entrada_usuario", {}).get("pregunta", "")
    
    # Obtener nombre amigable del equipo
    nombre_amigable = equipo_enum.nombre_amigable if isinstance(equipo_enum, TipoEquipo) else "Equipo desconocido"
    
    logger.debug("Sintetizando respuesta final...")
    logger.debug(f"Equipo: {equipo_enum} -> '{nombre_amigable}'")
    logger.debug(f"Info disponible: {len(info_recuperada)} caracteres")
    
    try:
        # Usar Gemini para generar una respuesta amigable
        logger.debug("Generando respuesta con Gemini...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Genera una respuesta amigable y profesional para el usuario basándote en la 
        siguiente información:
        
        Equipo identificado: {nombre_amigable}
        Pregunta del usuario: {pregunta_usuario}
        Información técnica encontrada: {info_recuperada}
        
        La respuesta debe:
        1. Confirmar qué equipo se identificó en la imagen
        2. Responder directamente a la pregunta del usuario
        3. Ser clara, concisa y profesional
        4. Si no se encontró información específica, explicarlo amablemente
        
        Respuesta:
        """
        
        response = model.generate_content(prompt)
        respuesta_final = response.text.strip()
        logger.debug(f"Respuesta final generada: {len(respuesta_final)} caracteres")
        
    except Exception as e:
        logger.warning(f"Error al generar respuesta con Gemini: {str(e)}")
        logger.debug("Usando respuesta de fallback...")
        # Respuesta de fallback si hay error con el LLM
        respuesta_final = f"""
        He identificado un {nombre_amigable} en la imagen.
        
        Respecto a tu pregunta: {pregunta_usuario}
        
        {info_recuperada}
        """
    
    return {"respuesta_final": respuesta_final}


def gestionar_error(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que maneja errores cuando la entrada no es válida.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con mensaje de error
    """
    logger = get_logger("nodes.error")
    
    entrada = state.get("entrada_usuario", {})
    archivo = entrada.get("archivo", {})
    nombre_archivo = archivo.get("nombre", "archivo_desconocido")
    
    logger.debug(f"Manejando error para archivo: '{nombre_archivo}'")
    
    mensaje_error = """
    Lo siento, este agente solo puede procesar archivos de imagen (.jpg, .jpeg, .png, etc.).
    
    Por favor, adjunta una imagen de un equipo técnico junto con tu pregunta para que 
    pueda ayudarte con información específica sobre ese equipo.
    
    Los equipos que puedo identificar incluyen:
    - Tomacorriente Industrial
    - Robot Robotino
    - Compresor de Aire
    - Ventilador Industrial
    - Proyector de Aula
    - Señal de Riesgo Eléctrico
    """
    
    logger.debug("Mensaje de error generado")
    
    return {"respuesta_final": mensaje_error.strip()}