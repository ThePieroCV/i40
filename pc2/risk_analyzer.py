"""
Analizador de riesgos críticos para entornos industriales.

Este módulo se especializa en identificar y evaluar riesgos de seguridad
en imágenes de equipos y entornos industriales.
"""

import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any
from pc2.models import TipoEquipo
from pc2.logger import get_logger


class RiskAnalyzer:
    """
    Analizador de riesgos críticos para seguridad industrial.
    """
    
    def __init__(self):
        self.logger = get_logger("risk_analyzer")
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analizar_riesgos_frame(self, frame: Image.Image, equipos_detectados: List[TipoEquipo], frame_numero: int) -> Dict[str, Any]:
        """
        Analiza riesgos críticos en un frame específico.
        
        Args:
            frame: Imagen del frame a analizar
            equipos_detectados: Lista de equipos detectados en el frame
            frame_numero: Número del frame (para referencia)
            
        Returns:
            dict: Análisis de riesgos del frame
        """
        self.logger.debug(f"Analizando riesgos en frame {frame_numero}")
        self.logger.debug(f"Equipos detectados: {[eq.name for eq in equipos_detectados]}")
        
        try:
            # Crear lista de equipos para el prompt
            equipos_str = ", ".join([eq.nombre_amigable for eq in equipos_detectados]) if equipos_detectados else "Ningún equipo específico detectado"
            
            # Prompt especializado en análisis de riesgos industriales
            prompt = f"""
            ANÁLISIS DE RIESGOS CRÍTICOS - FRAME {frame_numero}
            
            Analiza esta imagen desde la perspectiva de SEGURIDAD INDUSTRIAL y identifica RIESGOS CRÍTICOS.
            
            Equipos detectados en este frame: {equipos_str}
            
            Enfócate en identificar:
            
            1. RIESGOS ELÉCTRICOS:
            - Cables expuestos o dañados
            - Conexiones inseguras
            - Falta de señalización de riesgo eléctrico
            - Equipos sin puesta a tierra visible
            
            2. RIESGOS MECÁNICOS:
            - Partes móviles sin protección
            - Guardas de seguridad faltantes o abiertas
            - Herramientas en mal estado
            - Piezas sueltas o mal aseguradas
            
            3. RIESGOS DE PRESIÓN:
            - Mangueras o tuberías dañadas
            - Conexiones de presión inseguras
            - Válvulas de seguridad bloqueadas
            - Sobrepresión evidente
            
            4. RIESGOS ERGONÓMICOS:
            - Posturas peligrosas de trabajadores
            - Espacios de trabajo inadecuados
            - Falta de elementos de apoyo
            
            5. RIESGOS AMBIENTALES:
            - Iluminación insuficiente
            - Orden y limpieza deficientes
            - Obstrucción de vías de evacuación
            - Materiales peligrosos mal almacenados
            
            6. EQUIPOS DE PROTECCIÓN PERSONAL (EPP):
            - Falta de EPP en trabajadores visibles
            - EPP inadecuado para la tarea
            - EPP dañado o mal usado
            
            FORMATO DE RESPUESTA:
            Para cada riesgo identificado, indica:
            - TIPO DE RIESGO: (Eléctrico/Mecánico/Presión/Ergonómico/Ambiental/EPP)
            - DESCRIPCIÓN: Qué específicamente observas
            - NIVEL: CRÍTICO/ALTO/MEDIO/BAJO
            - RECOMENDACIÓN: Acción inmediata sugerida
            
            Si NO observas riesgos significativos, indica: "No se observan riesgos críticos evidentes en este frame."
            
            RESPUESTA:
            """
            
            self.logger.debug("Enviando frame y prompt de riesgos a Gemini...")
            response = self.model.generate_content([prompt, frame])
            analisis_riesgos = response.text.strip()
            
            self.logger.debug(f"Análisis de riesgos completado para frame {frame_numero}")
            
            # Determinar nivel de criticidad general del frame
            nivel_criticidad = self._determinar_nivel_criticidad(analisis_riesgos)
            
            return {
                "frame_numero": frame_numero,
                "equipos_detectados": [eq.nombre_amigable for eq in equipos_detectados],
                "analisis_riesgos": analisis_riesgos,
                "nivel_criticidad": nivel_criticidad,
                "timestamp_analisis": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de riesgos frame {frame_numero}: {str(e)}")
            return {
                "frame_numero": frame_numero,
                "equipos_detectados": [eq.nombre_amigable for eq in equipos_detectados] if equipos_detectados else [],
                "analisis_riesgos": f"Error en análisis: {str(e)}",
                "nivel_criticidad": "ERROR",
                "timestamp_analisis": self._get_timestamp()
            }
    
    def analizar_riesgos_multiples_frames(self, frames_data: List[Dict]) -> Dict[str, Any]:
        """
        Analiza riesgos en múltiples frames y genera un resumen consolidado.
        
        Args:
            frames_data: Lista de datos de frames con riesgos analizados
            
        Returns:
            dict: Resumen consolidado de riesgos
        """
        self.logger.debug(f"Consolidando análisis de {len(frames_data)} frames")
        
        try:
            # Extraer todos los análisis
            analisis_frames = [frame_data.get("analisis_riesgos", "") for frame_data in frames_data]
            niveles_criticidad = [frame_data.get("nivel_criticidad", "BAJO") for frame_data in frames_data]
            
            # Determinar criticidad máxima
            orden_criticidad = {"CRÍTICO": 4, "ALTO": 3, "MEDIO": 2, "BAJO": 1, "ERROR": 0}
            criticidad_maxima = max(niveles_criticidad, key=lambda x: orden_criticidad.get(x, 0))
            
            # Crear prompt para resumen consolidado
            prompt = f"""
            RESUMEN CONSOLIDADO DE ANÁLISIS DE RIESGOS
            
            Se han analizado {len(frames_data)} frames de un video industrial.
            
            Análisis por frame:
            {chr(10).join([f"FRAME {i+1}: {analisis}" for i, analisis in enumerate(analisis_frames)])}
            
            Genera un RESUMEN EJECUTIVO que incluya:
            
            1. RIESGOS CRÍTICOS IDENTIFICADOS: Lista los riesgos más importantes encontrados
            2. PATRONES DE RIESGO: Riesgos que se repiten a lo largo del video
            3. RECOMENDACIONES PRIORITARIAS: Top 3 acciones más urgentes
            4. EVALUACIÓN GENERAL: Nivel de seguridad del entorno observado
            
            Mantén el resumen conciso pero completo, enfocándote en los aspectos más críticos para la seguridad.
            """
            
            response = self.model.generate_content(prompt)
            resumen_consolidado = response.text.strip()
            
            return {
                "total_frames_analizados": len(frames_data),
                "criticidad_maxima": criticidad_maxima,
                "resumen_consolidado": resumen_consolidado,
                "detalles_por_frame": frames_data,
                "timestamp_consolidacion": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error en consolidación de riesgos: {str(e)}")
            return {
                "total_frames_analizados": len(frames_data),
                "criticidad_maxima": "ERROR",
                "resumen_consolidado": f"Error en consolidación: {str(e)}",
                "detalles_por_frame": frames_data,
                "timestamp_consolidacion": self._get_timestamp()
            }
    
    def _determinar_nivel_criticidad(self, analisis_texto: str) -> str:
        """
        Determina el nivel de criticidad basado en el texto del análisis.
        
        Args:
            analisis_texto: Texto del análisis de riesgos
            
        Returns:
            str: Nivel de criticidad (CRÍTICO/ALTO/MEDIO/BAJO)
        """
        texto_lower = analisis_texto.lower()
        
        # Palabras clave para determinar criticidad
        palabras_criticas = ["crítico", "peligro inmediato", "riesgo grave", "urgente", "fatal"]
        palabras_altas = ["alto", "significativo", "importante", "preocupante"]
        palabras_medias = ["medio", "moderado", "atención", "considerar"]
        
        if any(palabra in texto_lower for palabra in palabras_criticas):
            return "CRÍTICO"
        elif any(palabra in texto_lower for palabra in palabras_altas):
            return "ALTO"
        elif any(palabra in texto_lower for palabra in palabras_medias):
            return "MEDIO"
        elif "no se observan riesgos" in texto_lower:
            return "BAJO"
        else:
            return "MEDIO"  # Default
    
    def _get_timestamp(self) -> str:
        """
        Obtiene timestamp actual para el análisis.
        
        Returns:
            str: Timestamp formateado
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def analizar_riesgos_frame_simple(frame: Image.Image, equipos_detectados: List[TipoEquipo], frame_numero: int) -> Dict[str, Any]:
    """
    Función simplificada para análisis de riesgos de un frame.
    
    Args:
        frame: Imagen del frame
        equipos_detectados: Equipos detectados
        frame_numero: Número del frame
        
    Returns:
        dict: Análisis de riesgos
    """
    analyzer = RiskAnalyzer()
    return analyzer.analizar_riesgos_frame(frame, equipos_detectados, frame_numero)