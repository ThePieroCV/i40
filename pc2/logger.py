"""
Sistema de logging configurable para el agente.
"""

import logging
import sys

# Variable global para el modo debug
_debug_mode = False


def set_debug_mode(debug: bool):
    """Establece el modo debug globalmente."""
    global _debug_mode
    _debug_mode = debug
    
    # Reconfigurar todos los loggers existentes
    for name in logging.Logger.manager.loggerDict:
        if name.startswith('agente.'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG if debug else logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado según el modo debug actual.
    
    Args:
        name: Nombre del módulo/componente
        
    Returns:
        Logger configurado
    """
    logger_name = f"agente.{name}"
    logger = logging.getLogger(logger_name)
    
    # Configurar nivel según modo debug
    logger.setLevel(logging.DEBUG if _debug_mode else logging.WARNING)
    
    # Agregar handler solo si no existe
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Evitar propagación para prevenir logs duplicados
    logger.propagate = False
    
    return logger