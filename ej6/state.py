"""
Definición del estado del agente de asistencia técnica.

Este módulo contiene la estructura de datos que representa el estado
del agente mientras procesa una solicitud del usuario.
"""

from typing import TypedDict, Optional, Dict, Any


class AgentState(TypedDict):
    """
    Estado del agente que se va actualizando a través del grafo.
    
    Attributes:
        entrada_usuario: Diccionario con la pregunta del usuario y el archivo adjunto
        es_valido: Booleano que indica si la entrada es una imagen válida
        nombre_equipo: Nombre del equipo identificado en la imagen
        info_recuperada: Información técnica recuperada de la base de conocimiento
        respuesta_final: Respuesta final formateada para el usuario
    """
    entrada_usuario: Dict[str, Any]
    es_valido: Optional[bool]
    nombre_equipo: Optional[str]
    info_recuperada: Optional[str]
    respuesta_final: Optional[str]