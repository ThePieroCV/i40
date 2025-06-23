"""
Funciones auxiliares para la construcción del grafo.
"""

from langgraph.graph import StateGraph, END
from ej6.state import AgentState


def crear_grafo_base():
    """
    Crea un grafo base con el estado definido.
    
    Returns:
        StateGraph: Grafo vacío listo para agregar nodos y conexiones
    """
    return StateGraph(AgentState)


def crear_router_validacion():
    """
    Crea una función router para decidir el flujo después de la validación.
    
    Returns:
        callable: Función que decide el siguiente paso basado en es_valido
    """
    def router(state):
        if state.get("es_valido", False):
            return "continuar"
        else:
            return "error"
    return router