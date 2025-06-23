"""
Procesador de video para extracción de frames.

Este módulo se encarga de procesar videos y extraer frames distribuidos
proporcionalmente a lo largo de la duración del video.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Tuple
from pc2.logger import get_logger


def extraer_frames_video(video_bytes: bytes, num_frames: int = 6) -> List[Image.Image]:
    """
    Extrae frames de un video distribuidos proporcionalmente en el tiempo.
    
    Args:
        video_bytes: Contenido del video en bytes
        num_frames: Número de frames a extraer (default: 6)
        
    Returns:
        List[Image.Image]: Lista de frames como imágenes PIL
    """
    logger = get_logger("video_processor.extraer_frames")
    
    logger.debug(f"Iniciando extracción de {num_frames} frames...")
    logger.debug(f"Tamaño del video: {len(video_bytes)} bytes")
    
    try:
        # Crear un archivo temporal en memoria para OpenCV
        temp_file = BytesIO(video_bytes)
        
        # Escribir bytes a archivo temporal
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_file_path = tmp_file.name
        
        logger.debug(f"Archivo temporal creado: {tmp_file_path}")
        
        # Abrir video con OpenCV
        cap = cv2.VideoCapture(tmp_file_path)
        
        if not cap.isOpened():
            logger.error("No se pudo abrir el video con OpenCV")
            raise ValueError("No se pudo abrir el archivo de video")
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.debug(f"Total frames: {total_frames}")
        logger.debug(f"FPS: {fps}")
        logger.debug(f"Duración: {duration:.2f} segundos")
        
        if total_frames < num_frames:
            logger.warning(f"Video tiene menos frames ({total_frames}) que los solicitados ({num_frames})")
            num_frames = total_frames
        
        # Calcular posiciones de frames a extraer (distribuidos proporcionalmente)
        if num_frames == 1:
            frame_positions = [total_frames // 2]
        else:
            step = total_frames / (num_frames + 1)
            frame_positions = [int(step * (i + 1)) for i in range(num_frames)]
        
        logger.debug(f"Posiciones de frames a extraer: {frame_positions}")
        
        frames_extraidos = []
        
        for i, frame_pos in enumerate(frame_positions):
            # Ir a la posición específica del frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            ret, frame = cap.read()
            if ret:
                # Convertir de BGR a RGB (OpenCV usa BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convertir a PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames_extraidos.append(pil_image)
                
                logger.debug(f"Frame {i+1}/{num_frames} extraído en posición {frame_pos}")
            else:
                logger.warning(f"No se pudo leer frame en posición {frame_pos}")
        
        # Limpiar recursos
        cap.release()
        os.unlink(tmp_file_path)  # Eliminar archivo temporal
        
        logger.debug(f"Extracción completada: {len(frames_extraidos)} frames")
        
        return frames_extraidos
        
    except Exception as e:
        logger.error(f"Error durante extracción de frames: {str(e)}")
        raise


def validar_formato_video(archivo_bytes: bytes, nombre_archivo: str) -> bool:
    """
    Valida si un archivo es un video soportado.
    
    Args:
        archivo_bytes: Contenido del archivo en bytes
        nombre_archivo: Nombre del archivo para verificar extensión
        
    Returns:
        bool: True si es un video válido, False en caso contrario
    """
    logger = get_logger("video_processor.validar_formato")
    
    # Extensiones de video soportadas
    extensiones_video = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']
    
    # Verificar extensión
    extension = '.' + nombre_archivo.lower().split('.')[-1] if '.' in nombre_archivo else ''
    es_extension_valida = extension in extensiones_video
    
    logger.debug(f"Archivo: {nombre_archivo}")
    logger.debug(f"Extensión: {extension}")
    logger.debug(f"¿Extensión válida? {es_extension_valida}")
    
    if not es_extension_valida:
        return False
    
    # Verificar que el archivo no esté vacío
    if len(archivo_bytes) < 1024:  # Al menos 1KB
        logger.debug("Archivo demasiado pequeño para ser un video válido")
        return False
    
    # Verificar signature de archivos de video comunes
    signatures = {
        b'\x00\x00\x00\x18ftypmp4': 'MP4',
        b'\x00\x00\x00\x20ftypmp4': 'MP4',
        b'RIFF': 'AVI',
        b'\x1a\x45\xdf\xa3': 'MKV/WEBM'
    }
    
    for signature, formato in signatures.items():
        if archivo_bytes.startswith(signature):
            logger.debug(f"Formato detectado: {formato}")
            return True
    
    # Si no se detecta signature específica pero tiene extensión válida, asumir que es válido
    logger.debug("No se detectó signature específica, pero extensión es válida")
    return True


def obtener_info_video(video_bytes: bytes) -> dict:
    """
    Obtiene información básica del video.
    
    Args:
        video_bytes: Contenido del video en bytes
        
    Returns:
        dict: Información del video (duración, fps, resolución, etc.)
    """
    logger = get_logger("video_processor.info_video")
    
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_file_path = tmp_file.name
        
        cap = cv2.VideoCapture(tmp_file_path)
        
        if not cap.isOpened():
            raise ValueError("No se pudo abrir el archivo de video")
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
        
        cap.release()
        os.unlink(tmp_file_path)
        
        logger.debug(f"Info del video: {info}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error obteniendo info del video: {str(e)}")
        return {}