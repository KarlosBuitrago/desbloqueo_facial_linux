"""
Gestor de Cámara
===============

Este módulo maneja toda la funcionalidad relacionada con la cámara web,
incluyendo su inicialización, captura de frames y liberación de recursos.

Características:
- Inicialización automática con reintentos
- Captura de video en hilo separado
- Manejo de errores y excepciones
- Liberación segura de recursos
"""

import cv2
import time
import threading
import logging

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Clase para gestionar la cámara web y la captura de video.
    
    Esta clase proporciona una interfaz para:
    - Inicializar la cámara
    - Capturar frames
    - Procesar video en tiempo real
    - Liberar recursos
    """
    
    def __init__(self, resolution=(320, 240)):
        """
        Inicializa el gestor de cámara.
        
        Args:
            resolution (tuple): Resolución de la cámara (ancho, alto)
        """
        self.resolution = resolution
        self.cap = None
        self.is_recording = False
        self.camera_thread = None
        self.frame_callback = None
        
    def initialize_camera(self):
        """
        Inicializa la cámara con reintentos.
        
        Intenta conectar la cámara con diferentes índices hasta 3 veces,
        configurando la resolución especificada.
        
        Returns:
            bool: True si la cámara se inicializó correctamente
        """
        max_attempts = 3
        camera_indices = [0, 1, 2]  # Intentar diferentes índices de cámara
        
        for index in camera_indices:
            for attempt in range(max_attempts):
                try:
                    # Siempre intentar inicializar la cámara. Si ya existe, liberarla primero.
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None # Asegurarse de que sea None después de liberar

                    logger.info(f"Intentando conectar cámara en índice {index}")
                        
                    # Configurar la cámara con parámetros específicos
                    self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                        
                    if self.cap.isOpened():
                        # Configuración para formato YUYV
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Obtener información de la cámara
                        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        logger.info(f"Configuración de cámara: {width}x{height} @ {fps}fps")
                        
                        # Intentar leer frames con timeouts más cortos
                        frames_leidos = 0
                        for _ in range(3):  # Reducir a 3 intentos
                            ret, frame = self.cap.read()
                            if ret and frame is not None:
                                frames_leidos += 1
                                logger.debug(f"Frame {frames_leidos} leído exitosamente")
                            time.sleep(0.05)  # Reducir el tiempo de espera
                        
                        if frames_leidos > 0:
                            logger.info(f"Cámara conectada exitosamente en índice {index}")
                            logger.info(f"Frames leídos exitosamente: {frames_leidos}/3")
                            return True
                        else:
                            logger.warning(f"No se pudo leer frames de la cámara en índice {index}")
                            self.cap.release()
                            self.cap = None
                    else:
                        logger.warning(f"No se pudo abrir la cámara en índice {index}")
                        
                except Exception as e:
                    logger.error(f"Error al conectar cámara en índice {index}: {str(e)}")
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    time.sleep(0.5)  # Reducir el tiempo de espera entre intentos
                    
        logger.error("No se pudo conectar ninguna cámara")
        return False
        
    def start_capture(self, callback=None):
        """
        Inicia la captura de video.
        
        Crea un hilo separado para la captura de video y llama
        a la función callback con cada frame capturado.
        
        Args:
            callback (function): Función a llamar con cada frame
            
        Returns:
            bool: True si la cámara se inició y está grabando, False en caso contrario.
        """
        if self.camera_thread and self.camera_thread.is_alive():
            logger.info("La captura de cámara ya está activa.")
            return True # Ya está activa, consideramos éxito.
            
        # Intentar inicializar la cámara antes de iniciar el hilo.
        if not self.initialize_camera():
            logger.error("No se pudo inicializar la cámara antes de iniciar la captura.")
            self.is_recording = False # Asegurarse de que no esté en estado de grabación
            return False

        self.frame_callback = callback
        self.is_recording = True # Esto es lo que controla el bucle
        
        # Si el hilo ya existe pero no está activo, o no existe, lo creamos y empezamos
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.camera_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.camera_thread.start()
            logger.info("Captura de cámara iniciada.")
        else:
            logger.info("El hilo de captura ya está corriendo, ajustando solo la bandera de grabación.")
        
        return True # Si llegamos aquí, la cámara se inicializó y el hilo se inició o ya estaba activo.

    def _capture_loop(self):
        """
        Bucle principal de captura de video.
        
        Captura frames continuamente mientras is_recording sea True
        y llama a la función callback con cada frame.
        """
        logger.debug("Iniciando bucle de captura de cámara.")
        # La inicialización de la cámara ahora se hace en start_capture().
        # if not self.initialize_camera():
        #     logger.error("No se pudo inicializar la cámara para el bucle de captura.")
        #     self.is_recording = False # Asegurarse de que el estado sea consistente
        #     return
            
        consecutive_failures = 0
        max_failures = 5
            
        while self.is_recording and self.cap is not None: # Añadir check para self.cap
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Error al leer frame ({consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Demasiados errores consecutivos, reiniciando cámara...")
                        if not self.initialize_camera():
                            logger.error("No se pudo reiniciar la cámara")
                            self.is_recording = False # Detener si no se puede reiniciar
                            break
                        consecutive_failures = 0
                    time.sleep(0.1)
                    continue
                    
                consecutive_failures = 0
                if self.frame_callback:
                    self.frame_callback(frame)
                    
            except Exception as e:
                logger.error(f"Error en captura: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Demasiados errores consecutivos, reiniciando cámara...")
                    if not self.initialize_camera():
                        logger.error("No se pudo reiniciar la cámara")
                        self.is_recording = False # Detener si no se puede reiniciar
                        break
                    consecutive_failures = 0
                time.sleep(0.1)
                continue
                
        logger.info("Bucle de captura detenido. Liberando recursos de la cámara.")
        # La liberación del cap se ha movido a stop_camera para asegurar liberación inmediata.
        # if self.cap is not None: 
        #     self.cap.release()
        #     self.cap = None
        
    def stop_camera(self):
        """
        Detiene la captura de video y libera recursos.
        
        Detiene el hilo de captura estableciendo la bandera `is_recording` a False.
        La liberación del objeto de cámara `cap` se maneja aquí directamente.
        """
        logger.info("Solicitando detención de cámara...")
        self.is_recording = False

        # Liberar el recurso de la cámara inmediatamente
        if self.cap is not None:
            logger.info("Liberando recurso de cámara (cap.release())...")
            self.cap.release()
            self.cap = None
        
        # Esperar a que el hilo de captura termine si no estamos en el propio hilo de la cámara.
        if self.camera_thread and self.camera_thread.is_alive() and threading.current_thread() != self.camera_thread:
            logger.info("Esperando que el hilo de captura termine...")
            self.camera_thread.join(timeout=2.0) # Ajustar timeout si es necesario
            if self.camera_thread.is_alive():
                logger.warning("El hilo de captura no terminó a tiempo.")
            
        logger.info("Solicitud de detención de cámara procesada.")
            
    def save_frame(self, path):
        """
        Guarda el frame actual en un archivo.
        
        Args:
            path (str): Ruta donde guardar el frame
            
        Returns:
            bool: True si el frame se guardó correctamente
        """
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(path, frame)
                logger.info(f"Frame guardado en: {path}")
                return True
            else:
                logger.warning(f"No se pudo leer el frame para guardar en {path}")
        else:
            logger.warning("Cámara no inicializada, no se puede guardar el frame.")
        return False 