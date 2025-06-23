"""
Modelos Pydantic y Enums para el agente de asistencia técnica.

Este módulo define las estructuras de datos que garantizan respuestas
consistentes y eliminan errores de mapeo.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class TipoEquipo(Enum):
    """
    Enum que mapea directamente cada equipo a su archivo de documentación.
    
    Esto elimina completamente los errores de mapeo ya que cada valor
    del enum corresponde exactamente al nombre del archivo .txt.
    """
    COMPRESOR_AIRE = "compresor_aire.txt"
    ROBOTINO_FESTO = "robotino_festo.txt"
    VENTILADOR_INDUSTRIAL = "ventilador_industrial.txt"
    PROYECTOR_AULA = "proyector_aula.txt"
    TOMACORRIENTE_INDUSTRIAL = "tomacorriente_industrial.txt"
    SENAL_RIESGO_ELECTRICO = "senal_riesgo_electrico.txt"
    EQUIPO_NO_IDENTIFICADO = None  # Para casos donde no se puede identificar

    @property
    def nombre_amigable(self) -> str:
        """Retorna un nombre amigable para mostrar al usuario."""
        nombres = {
            self.COMPRESOR_AIRE: "Compresor de Aire",
            self.ROBOTINO_FESTO: "Robot Robotino Festo",
            self.VENTILADOR_INDUSTRIAL: "Ventilador Industrial",
            self.PROYECTOR_AULA: "Proyector de Aula",
            self.TOMACORRIENTE_INDUSTRIAL: "Tomacorriente Industrial",
            self.SENAL_RIESGO_ELECTRICO: "Señal de Riesgo Eléctrico",
            self.EQUIPO_NO_IDENTIFICADO: "Equipo no identificado"
        }
        return nombres.get(self, "Equipo desconocido")

    @classmethod
    def get_lista_para_prompt(cls) -> str:
        """
        Retorna una lista formateada de equipos para usar en prompts.
        
        Returns:
            str: Lista de equipos válidos separados por comas
        """
        equipos_validos = [equipo.name for equipo in cls if equipo != cls.EQUIPO_NO_IDENTIFICADO]
        return ", ".join(equipos_validos)


class IdentificacionEquipo(BaseModel):
    """
    Modelo Pydantic para la respuesta estructurada de identificación de equipos.
    
    Garantiza que Gemini responda exactamente con uno de los valores válidos
    del enum TipoEquipo.
    """
    equipo: TipoEquipo = Field(
        description="Tipo de equipo identificado en la imagen"
    )
    confianza: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza en la identificación (0.0 a 1.0)"
    )
    observaciones: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Observaciones adicionales sobre la identificación"
    )

    class Config:
        use_enum_values = False  # Mantener el enum, no convertir a string


class RespuestaRAG(BaseModel):
    """
    Modelo para la respuesta estructurada del sistema RAG.
    
    Estructura la información recuperada de la base de conocimiento.
    """
    informacion_encontrada: bool = Field(
        description="Si se encontró información relevante"
    )
    respuesta: str = Field(
        description="Respuesta a la pregunta del usuario"
    )
    fuente: Optional[str] = Field(
        default=None,
        description="Archivo fuente de la información"
    )
    confianza_respuesta: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confianza en la respuesta generada"
    )