"""
Nodos del grafo para el agente multimedia de asistencia técnica.

Este módulo contiene las funciones que representan cada nodo del grafo.
Cada función recibe el estado actual y retorna las actualizaciones necesarias.
Incluye nodos para procesamiento de imágenes y videos.
"""

import google.generativeai as genai
from typing import Dict, Any, List
from i40.pc2.state import AgentState
from i40.pc2.tools import buscar_informacion_tecnica
from i40.pc2.models import TipoEquipo, IdentificacionEquipo
from i40.pc2.logger import get_logger
from i40.pc2.video_processor import extraer_frames_video, validar_formato_video
from i40.pc2.risk_analyzer import analizar_riesgos_frame_simple, RiskAnalyzer
import base64
from io import BytesIO
from PIL import Image
import json


def validar_entrada_multimedia(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que valida si la entrada del usuario contiene una imagen o video válido.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con es_valido y tipo_entrada
    """
    logger = get_logger("nodes.validar_multimedia")
    
    entrada = state.get("entrada_usuario", {})
    archivo = entrada.get("archivo", {})
    
    logger.debug("Validando entrada multimedia...")
    
    # Verificar si hay un archivo
    if not archivo:
        logger.debug("No se encontró archivo en la entrada")
        return {
            "es_valido": False,
            "tipo_entrada": "desconocido"
        }
    
    nombre_archivo = archivo.get("nombre", "").lower()
    contenido_archivo = archivo.get("contenido", b"")
    
    logger.debug(f"Nombre de archivo: '{nombre_archivo}'")
    logger.debug(f"Tamaño del archivo: {len(contenido_archivo)} bytes")
    
    # Tipos de archivo soportados
    tipos_imagen = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    tipos_video = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']
    
    # Verificar si es imagen
    es_imagen = any(nombre_archivo.endswith(tipo) for tipo in tipos_imagen)
    
    # Verificar si es video
    es_video = validar_formato_video(contenido_archivo, nombre_archivo)
    
    logger.debug(f"¿Es imagen? {es_imagen}")
    logger.debug(f"¿Es video? {es_video}")
    
    if es_imagen:
        return {
            "es_valido": True,
            "tipo_entrada": "imagen"
        }
    elif es_video:
        return {
            "es_valido": True,
            "tipo_entrada": "video"
        }
    else:
        return {
            "es_valido": False,
            "tipo_entrada": "desconocido"
        }


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
    Lo siento, este agente solo puede procesar archivos de imagen (.jpg, .jpeg, .png, etc.) 
    o videos (.mp4, .avi, .mov, etc.).
    
    Por favor, adjunta una imagen o video de un equipo técnico junto con tu pregunta para que 
    pueda ayudarte con información específica sobre ese equipo.
    
    Los equipos que puedo identificar incluyen:
    - Tomacorriente Industrial
    - Robot Robotino
    - Compresor de Aire
    - Ventilador Industrial
    - Proyector de Aula
    - Señal de Riesgo Eléctrico
    
    Si adjuntas un video, además analizaré riesgos críticos de seguridad en cada frame.
    """
    
    logger.debug("Mensaje de error generado")
    
    return {"respuesta_final": mensaje_error.strip()}


# ============================================================================
# NUEVOS NODOS PARA PROCESAMIENTO MULTIMEDIA
# ============================================================================

def procesar_video(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que extrae frames de un video para análisis posterior.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con frames_extraidos
    """
    logger = get_logger("nodes.procesar_video")
    
    entrada = state.get("entrada_usuario", {})
    archivo = entrada.get("archivo", {})
    contenido_video = archivo.get("contenido", b"")
    
    logger.debug("Iniciando procesamiento de video...")
    logger.debug(f"Tamaño del video: {len(contenido_video)} bytes")
    
    try:
        # Extraer 6 frames distribuidos proporcionalmente
        frames = extraer_frames_video(contenido_video, num_frames=6)
        
        logger.debug(f"Frames extraídos exitosamente: {len(frames)}")
        
        return {"frames_extraidos": frames}
        
    except Exception as e:
        logger.error(f"Error procesando video: {str(e)}")
        return {
            "frames_extraidos": [],
            "respuesta_final": f"Error al procesar el video: {str(e)}"
        }


def identificar_equipos_en_frames(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que identifica equipos en cada frame extraído del video.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con equipos_por_frame
    """
    logger = get_logger("nodes.identificar_frames")
    
    frames = state.get("frames_extraidos", [])
    
    logger.debug(f"Identificando equipos en {len(frames)} frames...")
    
    equipos_por_frame = []
    
    for i, frame in enumerate(frames):
        logger.debug(f"Procesando frame {i+1}/{len(frames)}")
        
        try:
            # Usar la misma lógica de identificación que para imágenes individuales
            # Obtener la lista de equipos válidos del enum
            lista_equipos = TipoEquipo.get_lista_para_prompt()
            
            # Configurar el modelo de visión
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Prompt estructurado para identificar el equipo
            prompt = f"""
            Observa esta imagen cuidadosamente y determina cuál de los siguientes equipos 
            técnicos aparece en ella.
            
            Equipos válidos: {lista_equipos}
            
            Responde ÚNICAMENTE con uno de estos valores exactos:
            - COMPRESOR_AIRE
            - ROBOTINO_FESTO
            - VENTILADOR_INDUSTRIAL
            - PROYECTOR_AULA
            - TOMACORRIENTE_INDUSTRIAL
            - SENAL_RIESGO_ELECTRICO
            - EQUIPO_NO_IDENTIFICADO (si no puedes identificar ninguno)
            
            Tu respuesta debe ser exactamente uno de estos valores, nada más.
            """
            
            response = model.generate_content([prompt, frame])
            respuesta_texto = response.text.strip().upper()
            
            # Intentar convertir la respuesta a enum
            try:
                equipo_enum = TipoEquipo[respuesta_texto]
                logger.debug(f"Frame {i+1}: {equipo_enum.name}")
                equipos_por_frame.append(equipo_enum)
            except KeyError:
                logger.warning(f"Frame {i+1}: Respuesta no válida '{respuesta_texto}', usando EQUIPO_NO_IDENTIFICADO")
                equipos_por_frame.append(TipoEquipo.EQUIPO_NO_IDENTIFICADO)
                
        except Exception as e:
            logger.error(f"Error identificando equipo en frame {i+1}: {str(e)}")
            equipos_por_frame.append(TipoEquipo.EQUIPO_NO_IDENTIFICADO)
    
    logger.debug(f"Identificación completada: {[eq.name for eq in equipos_por_frame]}")
    
    return {"equipos_por_frame": equipos_por_frame}


def analizar_riesgos_frames(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que analiza riesgos críticos en cada frame del video.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con riesgos_por_frame y resumen_riesgos
    """
    logger = get_logger("nodes.analizar_riesgos")
    
    frames = state.get("frames_extraidos", [])
    equipos_por_frame = state.get("equipos_por_frame", [])
    
    logger.debug(f"Analizando riesgos en {len(frames)} frames...")
    
    riesgos_por_frame = []
    
    # Analizar cada frame individualmente
    for i, (frame, equipos) in enumerate(zip(frames, equipos_por_frame)):
        logger.debug(f"Analizando riesgos en frame {i+1}")
        
        # Convertir equipos a lista si es un solo equipo
        equipos_lista = [equipos] if not isinstance(equipos, list) else equipos
        
        # Analizar riesgos del frame
        analisis_frame = analizar_riesgos_frame_simple(frame, equipos_lista, i+1)
        riesgos_por_frame.append(analisis_frame)
    
    # Generar resumen consolidado
    logger.debug("Generando resumen consolidado de riesgos...")
    
    try:
        risk_analyzer = RiskAnalyzer()
        resumen_consolidado = risk_analyzer.analizar_riesgos_multiples_frames(riesgos_por_frame)
    except Exception as e:
        logger.error(f"Error generando resumen consolidado: {str(e)}")
        resumen_consolidado = {
            "total_frames_analizados": len(frames),
            "criticidad_maxima": "ERROR",
            "resumen_consolidado": f"Error en consolidación: {str(e)}",
            "detalles_por_frame": riesgos_por_frame
        }
    
    logger.debug("Análisis de riesgos completado")
    
    return {
        "riesgos_por_frame": riesgos_por_frame,
        "resumen_riesgos": resumen_consolidado
    }


def responder_consulta_multimedia(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que responde la consulta específica del usuario basándose en el contenido analizado.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con respuesta_consulta
    """
    logger = get_logger("nodes.respuesta_consulta")
    
    tipo_entrada = state.get("tipo_entrada", "")
    pregunta_usuario = state.get("entrada_usuario", {}).get("pregunta", "")
    
    logger.debug(f"Respondiendo consulta para tipo: {tipo_entrada}")
    logger.debug(f"Pregunta: {pregunta_usuario}")
    
    if tipo_entrada == "imagen":
        # Para imágenes, usar el flujo original
        equipo = state.get("nombre_equipo")
        info_recuperada = state.get("info_recuperada", "")
        
        if isinstance(equipo, TipoEquipo) and equipo != TipoEquipo.EQUIPO_NO_IDENTIFICADO:
            respuesta = f"Respecto a tu pregunta sobre el {equipo.nombre_amigable}: {info_recuperada}"
        else:
            respuesta = "No pude identificar un equipo específico en la imagen para responder tu pregunta."
            
    elif tipo_entrada == "video":
        # Para videos, analizar los equipos encontrados en los frames
        equipos_por_frame = state.get("equipos_por_frame", [])
        
        # Encontrar el equipo más común en los frames
        equipos_validos = [eq for eq in equipos_por_frame if eq != TipoEquipo.EQUIPO_NO_IDENTIFICADO]
        
        if equipos_validos:
            # Contar frecuencia de equipos
            from collections import Counter
            contador_equipos = Counter(equipos_validos)
            equipo_principal = contador_equipos.most_common(1)[0][0]
            
            # Buscar información sobre el equipo principal
            try:
                info_tecnica = buscar_informacion_tecnica(equipo_principal, pregunta_usuario)
                respuesta = f"En el video identifiqué principalmente: {equipo_principal.nombre_amigable}. {info_tecnica}"
            except Exception as e:
                respuesta = f"Identifiqué {equipo_principal.nombre_amigable} en el video, pero no pude recuperar información técnica específica."
        else:
            respuesta = "No pude identificar equipos específicos en los frames del video para responder tu pregunta técnica."
    else:
        respuesta = "No se pudo determinar el tipo de entrada para responder la consulta."
    
    logger.debug(f"Respuesta generada: {respuesta[:100]}...")
    
    return {"respuesta_consulta": respuesta}


def sintetizar_respuesta_multimedia(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que genera la respuesta final combinando análisis técnico y de riesgos.
    
    Args:
        state: Estado actual del agente
        
    Returns:
        dict: Actualización del estado con respuesta_final
    """
    logger = get_logger("nodes.sintetizar_multimedia")
    
    tipo_entrada = state.get("tipo_entrada", "")
    respuesta_consulta = state.get("respuesta_consulta", "")
    
    logger.debug(f"Sintetizando respuesta final para tipo: {tipo_entrada}")
    
    if tipo_entrada == "imagen":
        # Para imágenes, usar solo la respuesta de la consulta (comportamiento original)
        respuesta_final = respuesta_consulta
        
    elif tipo_entrada == "video":
        # Para videos, combinar respuesta técnica + análisis de riesgos
        resumen_riesgos = state.get("resumen_riesgos", {})
        
        respuesta_final = f"""
{respuesta_consulta}

--- ANÁLISIS DE RIESGOS CRÍTICOS ---

{resumen_riesgos.get('resumen_consolidado', 'No se pudo generar análisis de riesgos.')}

Frames analizados: {resumen_riesgos.get('total_frames_analizados', 0)}
Nivel de criticidad máximo: {resumen_riesgos.get('criticidad_maxima', 'N/A')}
        """.strip()
    else:
        respuesta_final = "Error: No se pudo procesar el archivo correctamente."
    
    logger.debug(f"Respuesta final generada: {len(respuesta_final)} caracteres")
    
    return {"respuesta_final": respuesta_final}