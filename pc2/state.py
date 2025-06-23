"""
Definición del estado del agente multimedia de asistencia técnica.

Este módulo contiene la estructura de datos que representa el estado
del agente mientras procesa una solicitud del usuario (imagen o video).
"""

from typing import TypedDict, Optional, Dict, Any, List
from PIL import Image


class AgentState(TypedDict):
    """
    Estado del agente multimedia que se va actualizando a través del grafo.
    
    Attributes:
        entrada_usuario: Diccionario con la pregunta del usuario y el archivo adjunto
        tipo_entrada: Tipo de entrada detectado ("imagen" | "video" | "desconocido")
        es_valido: Booleano que indica si la entrada es válida
        
        # Para imágenes (comportamiento original)
        nombre_equipo: Nombre del equipo identificado en la imagen
        info_recuperada: Información técnica recuperada de la base de conocimiento
        
        # Para videos (nueva funcionalidad)
        frames_extraidos: Lista de frames extraídos del video
        equipos_por_frame: Lista de equipos detectados en cada frame
        riesgos_por_frame: Lista de análisis de riesgos por frame
        resumen_riesgos: Resumen consolidado de riesgos del video
        
        # Respuesta final (común para imagen y video)
        respuesta_consulta: Respuesta a la pregunta específica del usuario
        respuesta_final: Respuesta final completa formateada para el usuario
    """
    # Estado base
    entrada_usuario: Dict[str, Any]
    tipo_entrada: Optional[str]
    es_valido: Optional[bool]
    
    # Para imágenes (flujo original)
    nombre_equipo: Optional[Any]  # TipoEquipo enum
    info_recuperada: Optional[str]
    
    # Para videos (flujo nuevo)
    frames_extraidos: Optional[List[Image.Image]]
    equipos_por_frame: Optional[List[Any]]  # List[TipoEquipo]
    riesgos_por_frame: Optional[List[Dict[str, Any]]]
    resumen_riesgos: Optional[Dict[str, Any]]
    
    # Respuestas finales
    respuesta_consulta: Optional[str]
    respuesta_final: Optional[str]