"""
Sistema de Reconocimiento Facial para Desbloqueo de PC
===================================================

Este módulo implementa un sistema de reconocimiento facial para desbloquear
una computadora linux. Utiliza OpenCV para el procesamiento de imágenes
y reconocimiento facial, y tkinter para la interfaz gráfica.

Características principales:
- Registro de usuarios con múltiples fotos
- Reconocimiento facial automático
- Desbloqueo automático del PC
- Interfaz gráfica intuitiva
- Gestión de usuarios y fotos

Autor: Cipa 1
Fecha: [Fecha]
"""

import os
import time
import threading
import cv2 # type: ignore
from datetime import datetime
import tkinter as tk
import subprocess
import tkinter.messagebox as messagebox
import logging

import pyautogui # type: ignore

# Configuración del logging principal
logging.basicConfig(
    level=logging.INFO, # Cambiar a logging.DEBUG para ver más mensajes
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(), # Salida a consola
        logging.FileHandler('app.log') # Salida a archivo
    ]
)
logger = logging.getLogger(__name__)

# Configuración del logging para autenticaciones
auth_logger = logging.getLogger('auth_log')
auth_logger.setLevel(logging.INFO)
auth_handler = logging.FileHandler('autenticaciones.log')
auth_formatter = logging.Formatter('%(asctime)s - %(message)s')
auth_handler.setFormatter(auth_formatter)
auth_logger.addHandler(auth_handler)

from camera_manager import CameraManager
from face_recognizer import FaceRecognizer
from user_manager import UserManager
from gui_manager import GUIManager
from face_trainer import FaceTrainer

# Configuración del sistema
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "usuarios"
)
RESOLUTION = (320, 240)  # Resolución de la cámara
MAX_INITIAL_PHOTOS = 50  # Primera toma de fotos para entrenamiento
ADDITIONAL_PHOTOS_STEP = 5 # Fotos adicionales a tomar si el usuario lo desea
MAX_PHOTOS_PER_USER = 100  # Límite total de fotos por usuario
PHOTO_RETENTION_DAYS = 90  # Días de retención para fotos antiguas (si no están entre las últimas 50)
MIN_RECENT_PHOTOS_TO_KEEP = 50 # Número mínimo de fotos recientes a retener sin importar la antigüedad
CAMERA_CHECK_INTERVAL = 5  # Intervalo entre intentos
MAX_ATTEMPTS = 5  # Máximo de intentos
CONFIDENCE_THRESHOLD = 2000  # Umbral de confianza para reconocimiento con Eigenfaces (menor valor es mejor)
MODEL_PATH = "modeloEigenFaces.xml"  # Ruta al modelo entrenado


class FaceRecognitionApp:
    """
    Clase principal que coordina el sistema de reconocimiento facial.
    
    Esta clase maneja la interacción entre los diferentes componentes:
    - Gestión de usuarios
    - Captura de video
    - Reconocimiento facial
    - Interfaz gráfica
    - Desbloqueo del PC
    """
    
    def __init__(self):
        """
        Inicializa el sistema de reconocimiento facial.
        
        Configura todos los componentes necesarios y prepara la interfaz
        gráfica para la interacción con el usuario.
        """
        # Inicializar componentes
        self.user_manager = UserManager(
            BASE_DIR,
            MAX_PHOTOS_PER_USER,
            PHOTO_RETENTION_DAYS,
            MIN_RECENT_PHOTOS_TO_KEEP
        )
        self.camera_manager = CameraManager(RESOLUTION)
        
        # Inicializar reconocedor facial a None inicialmente para cargarlo asíncronamente
        self.face_recognizer = None

        # Inicializar el entrenador de rostros a None inicialmente para configurarlo asíncronamente
        self.face_trainer = None

        # Crear ventana principal
        self.root = tk.Tk()
        self.gui_manager = GUIManager(self.root, RESOLUTION)
        
        # Configurar callbacks
        self.gui_manager.set_callbacks(
            status_callback=self.update_status,
            register_callback=self.start_registration,
            recognition_callback=self.start_recognition,
            delete_callback=self.delete_user,
            stop_recognition_callback=self.stop_recognition
        )

        # Configurar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Actualizar lista de usuarios
        self.update_user_list()

        # Iniciar la carga del modelo y otras tareas en segundo plano
        self.root.after(100, self._post_gui_setup_tasks)

        # Inicializar estados de la aplicación
        self.recognition_active = False  # Estado del reconocimiento
        self.last_camera_check = datetime.now()  # Última verificación
        self.registration_active = False  # Estado del registro
        self.recognition_thread = None  # Hilo de reconocimiento
        self.current_user_registration = None # Para mantener el usuario durante el registro

    def _post_gui_setup_tasks(self):
        """
        Tareas a ejecutar después de que la GUI ha sido inicializada y está activa.
        Esto incluye la carga del modelo y la preparación del entrenador.
        """
        logger.info("Iniciando tareas de configuración en segundo plano...")
        self.update_status("Cargando modelo facial y preparando entrenador...")

        # Cargar el reconocedor facial
        self.face_recognizer = FaceRecognizer(
            MODEL_PATH,
            BASE_DIR,
            CONFIDENCE_THRESHOLD
        )
        # Ahora que face_recognizer está instanciado, inicializar face_trainer
        self.face_trainer = FaceTrainer(BASE_DIR, self.face_recognizer)
        
        if os.path.exists(MODEL_PATH) and self.face_recognizer.user_list: # Verificar si el modelo se cargó y hay usuarios
            self.update_status("Modelo facial cargado. Listo para reconocimiento.")
        else:
            self.update_status("Modelo no encontrado o sin usuarios. Por favor, registre usuarios y entrene el modelo.")
            self.gui_manager.show_message(
                "Advertencia",
                "No se encontró un modelo entrenado o usuarios. Por favor, registre usuarios y entrene el modelo.",
                "warning"
            )
        logger.info("Tareas de configuración en segundo plano completadas.")
            
    def update_user_list(self):
        """
        Actualiza la lista de usuarios en la interfaz gráfica.
        
        Obtiene la lista de usuarios registrados y la muestra en el
        combobox de la interfaz.
        """
        users = self.user_manager.get_users()
        self.gui_manager.update_user_list(users)

    def update_status(self, status):
        """
        Actualiza el mensaje de estado en la interfaz.
        
        Args:
            status (str): Mensaje de estado a mostrar
        """
        self.gui_manager.update_status(status)
        
    def start_registration(self, current_photos_count=0):
        """
        Inicia o continúa el proceso de registro de un nuevo usuario.
        
        Verifica que el PC esté desbloqueado, solicita el nombre del usuario
        y comienza o continúa el proceso de captura de fotos para el entrenamiento.
        
        Args:
            current_photos_count (int): Número de fotos ya tomadas para el usuario actual.
        """
        logger.debug(f"start_registration llamado. current_photos_count={current_photos_count}")
        username = None # Definir username fuera de los bloques para alcance

        if not self.registration_active:
            username = self.gui_manager.ask_string(
                "Registro",
                "Ingrese el nombre del usuario:"
            )
            if not username:
                logger.debug("start_registration: Nombre de usuario no proporcionado, regresando.")
                return
            self.current_user_registration = username
            
            user_path = os.path.join(BASE_DIR, username)
            logger.debug(f"start_registration: user_path='{user_path}'")
            logger.debug(f"start_registration: os.path.exists(user_path)={os.path.exists(user_path)}")

            existing_photos_count = 0
            if os.path.exists(user_path):
                existing_photos = [
                    f for f in os.listdir(user_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                existing_photos_count = len(existing_photos)
            logger.debug(f"start_registration: existing_photos_count calculado={existing_photos_count}")

            if existing_photos_count == 0: # Nuevo usuario o usuario existente sin fotos
                logger.debug("start_registration: Entrando a la rama de Nuevo usuario o usuario existente sin fotos.")
                if not os.path.exists(user_path): # Es un usuario realmente nuevo (el directorio no existe)
                    logger.debug(f"start_registration: Directorio '{user_path}' NO existe, intentando crearlo.")
                    if not self.user_manager.create_user(username): 
                        logger.error(f"Error de sistema o permisos al intentar crear el directorio para '{username}'.")
                        self.gui_manager.show_message("Error", f"Error de sistema o permisos al intentar crear el directorio para '{username}'.", "error")
                        return
                    self.update_status(f"Registrando nuevo usuario {username}.")
                else: # El usuario ya existe pero tiene 0 fotos (el directorio sí existe pero está vacío)
                    logger.debug(f"start_registration: Directorio '{user_path}' SÍ existe pero tiene 0 fotos. No se creará de nuevo.")
                    self.update_status(f"Usuario {username} existente pero sin fotos. Iniciando captura de fotos iniciales.")
                
                # En cualquier caso (nuevo o existente con 0 fotos), el objetivo es MAX_INITIAL_PHOTOS
                target_photos_in_session = MAX_INITIAL_PHOTOS
                self.registration_active = True
                self.camera_manager.start_capture(
                    callback=lambda frame: self.process_registration_frame(
                        frame, username, max_photos_to_take_in_session=target_photos_in_session
                    )
                )

            else: # El usuario ya existe y tiene fotos (>0 fotos)
                logger.debug(f"start_registration: Entrando a la rama de usuario existente con {existing_photos_count} fotos.")
                msg = (
                    f"El usuario '{username}' ya tiene "
                    f"{existing_photos_count} fotos registradas. "
                    f"¿Desea tomar {ADDITIONAL_PHOTOS_STEP} fotos adicionales (hasta {MAX_PHOTOS_PER_USER})?"
                )
                
                if self.gui_manager.ask_confirmation("Usuario Existente", msg):
                    logger.debug("start_registration: Usuario quiere tomar más fotos.")
                    # El usuario quiere tomar más fotos. Calcular objetivo para esta sesión.
                    target_photos_in_session = min(existing_photos_count + ADDITIONAL_PHOTOS_STEP, MAX_PHOTOS_PER_USER)
                    if target_photos_in_session <= existing_photos_count: # Si ya está en el máximo o no puede añadir más
                         logger.info("start_registration: Ya se alcanzó el máximo de fotos, informando y entrenando.")
                         self.gui_manager.show_message("Información", f"El usuario {username} ya tiene el máximo de fotos ({MAX_PHOTOS_PER_USER}). Se entrenará el modelo.", "info")
                         if self.camera_manager.is_recording:
                             self.camera_manager.stop_camera()
                         self.update_status("Registro completado. Iniciando entrenamiento...")
                         self.current_user_registration = None
                         self.train_model(show_message_box=True)
                         self.gui_manager.update_user_list(self.user_manager.get_users())
                         return

                    self.update_status(f"Continuando registro para {username}. Fotos actuales: {existing_photos_count}. Objetivo sesión: {target_photos_in_session}")
                    self.registration_active = True
                    self.camera_manager.start_capture(
                        callback=lambda frame: self.process_registration_frame(
                            frame, username, max_photos_to_take_in_session=target_photos_in_session
                        )
                    )
                else: # El usuario no quiere tomar más fotos, proceder a entrenar
                    logger.info("start_registration: Usuario NO quiere más fotos, entrenando modelo.")
                    self.gui_manager.show_message("Información", "No se tomarán más fotos. Iniciando entrenamiento del modelo con las fotos existentes.", "info")
                    if self.camera_manager.is_recording: 
                        self.camera_manager.stop_camera()
                    self.update_status("Registro completado. Iniciando entrenamiento...")
                    self.current_user_registration = None 
                    self.train_model(show_message_box=True)
                    self.gui_manager.update_user_list(self.user_manager.get_users()) 
                    return # Salir del método ya que no se tomarán fotos

        else: # Si el registro ya está activo (continuando desde el prompt de 'tomar X más')
            logger.debug(f"start_registration: Registro activo, continuando. current_photos_count={current_photos_count}")
            username = self.current_user_registration
            # current_photos_count aquí es el total de fotos antes de esta nueva tanda.
            target_photos_in_session = min(current_photos_count + ADDITIONAL_PHOTOS_STEP, MAX_PHOTOS_PER_USER)
            
            if target_photos_in_session <= current_photos_count: # Si no se pueden tomar más fotos, ir a entrenar
                logger.info("start_registration: Se alcanzó el máximo de fotos durante la continuación. Entrenando modelo.")
                self.gui_manager.show_message("Información", f"El usuario {username} ya tiene el máximo de fotos ({MAX_PHOTOS_PER_USER}). Se entrenará el modelo.", "info")
                if self.camera_manager.is_recording:
                    self.camera_manager.stop_camera()
                self.update_status("Registro completado. Iniciando entrenamiento...")
                self.current_user_registration = None
                self.train_model(show_message_box=True)
                self.gui_manager.update_user_list(self.user_manager.get_users())
                return

            self.update_status(f"Continuando registro para {username}. Fotos actuales: {current_photos_count}. Objetivo sesión: {target_photos_in_session}")
            self.registration_active = True
            self.camera_manager.start_capture(
                callback=lambda frame: self.process_registration_frame(
                    frame, username, max_photos_to_take_in_session=target_photos_in_session
                )
            )
            
    def process_registration_frame(self, frame, username, max_photos_to_take_in_session=None):
        """
        Procesa cada frame durante el registro de usuario.
        
        Args:
            frame: Frame capturado por la cámara
            username (str): Nombre del usuario en registro
            max_photos_to_take_in_session (int, optional): Límite absoluto de fotos a alcanzar 
                                                         en esta sesión de registro. 
                                                         Si es None, el límite general es MAX_PHOTOS_PER_USER.
        """
        if not self.registration_active:
            logger.debug("process_registration_frame: Registro inactivo, regresando.")
            return
            
        # Usa self.face_recognizer para detectar rostros en tiempo real
        faces, gray = self.face_recognizer.detect_faces(frame)
        
        # Frame para mostrar
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        self.gui_manager.show_frame(display_frame)
        
        current_total_photos_for_user = self.user_manager.get_user_photo_count(username)
        logger.debug(f"process_registration_frame: Fotos actuales para '{username}': {current_total_photos_for_user}. Objetivo sesión: {max_photos_to_take_in_session}")

        # Solo toma fotos si se detecta un rostro y el total actual de fotos 
        # es menor que el límite objetivo de esta sesión.
        if len(faces) == 1 and current_total_photos_for_user < max_photos_to_take_in_session:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (128, 128))

            # Usar user_manager para añadir y gestionar la foto
            self.user_manager.add_photo_to_user(username, face_roi)
            current_total_photos_for_user = self.user_manager.get_user_photo_count(username) # Volver a obtener el conteo actualizado
            self.update_status(f"Registrando usuario {username}. Fotos tomadas: {current_total_photos_for_user}/{max_photos_to_take_in_session} (sesión) / {MAX_PHOTOS_PER_USER} (total)")
            logger.info(f"Foto tomada para '{username}'. Total: {current_total_photos_for_user}")


        # Verificar si se ha alcanzado o superado el límite objetivo de la sesión.
        # Esto significa que la tanda actual de fotos está completa.
        if current_total_photos_for_user >= max_photos_to_take_in_session:
            logger.info(f"Límite de fotos ({max_photos_to_take_in_session}) alcanzado para la sesión de '{username}'. Deteniendo cámara.")
            self.camera_manager.stop_camera()
            time.sleep(0.5) # Pequeña pausa para permitir que la cámara se detenga
            self.registration_active = False
            self.update_status("Captura de fotos completada para esta sesión.")

            # Lógica para entrenar el modelo después de las 50 fotos iniciales
            if max_photos_to_take_in_session == MAX_INITIAL_PHOTOS and current_total_photos_for_user >= MAX_INITIAL_PHOTOS:
                logger.info(f"Se tomaron {MAX_INITIAL_PHOTOS} fotos iniciales. Iniciando entrenamiento del modelo para '{username}'.")
                self.current_user_registration = None # Resetear el usuario actual para evitar loops
                self.train_model(show_message_box=True)
                self.gui_manager.update_user_list(self.user_manager.get_users())
                # Después de entrenar, ahora preguntamos si quiere más fotos
                if current_total_photos_for_user < MAX_PHOTOS_PER_USER:
                    msg = (
                        f"Se han tomado {current_total_photos_for_user} fotos para '{username}'. "
                        f"El modelo ha sido entrenado. ¿Desea tomar {ADDITIONAL_PHOTOS_STEP} fotos adicionales (hasta {MAX_PHOTOS_PER_USER})?"
                    )
                    if self.gui_manager.ask_confirmation("Continuar Registro", msg):
                        logger.info(f"Usuario '{username}' solicitó tomar más fotos después del entrenamiento inicial.")
                        self.start_registration(current_photos_count=current_total_photos_for_user)
                    else:
                        logger.info(f"Usuario '{username}' NO solicitó tomar más fotos. Registro finalizado.")
                        self.gui_manager.show_message("Información", "No se tomarán más fotos. Registro finalizado.", "info")
                else:
                    logger.info(f"Usuario '{username}' alcanzó el límite máximo de fotos ({MAX_PHOTOS_PER_USER}). Registro finalizado.")
                    self.gui_manager.show_message("Información", "Se ha alcanzado el límite máximo de fotos para este usuario. Registro finalizado.", "info")

            # Lógica para el límite máximo general o continuación de fotos adicionales
            elif current_total_photos_for_user >= MAX_PHOTOS_PER_USER:
                logger.info(f"Usuario '{username}' alcanzó el límite máximo de fotos ({MAX_PHOTOS_PER_USER}). Iniciando entrenamiento del modelo.")
                self.gui_manager.show_message("Información", "Se ha alcanzado el límite máximo de fotos para este usuario. Iniciando entrenamiento del modelo.", "info")
                self.current_user_registration = None
                self.train_model(show_message_box=True)
                self.gui_manager.update_user_list(self.user_manager.get_users()) 
            else:
                # Si no se ha alcanzado el límite máximo general y no es el final de las 50 iniciales, preguntar si el usuario quiere tomar más fotos
                logger.info(f"Sesión de fotos completada para '{username}'. Preguntando por más fotos.")
                msg = (
                    f"Se han tomado {current_total_photos_for_user} fotos para '{username}'. "
                    f"¿Desea tomar {ADDITIONAL_PHOTOS_STEP} fotos adicionales (hasta {MAX_PHOTOS_PER_USER})?"
                )
                if self.gui_manager.ask_confirmation("Continuar Registro", msg):
                    logger.info(f"Usuario '{username}' solicitó tomar más fotos.")
                    self.start_registration(current_photos_count=current_total_photos_for_user)
                else:
                    logger.info(f"Usuario '{username}' NO solicitó tomar más fotos. Iniciando entrenamiento del modelo.")
                    self.gui_manager.show_message("Información", "No se tomarán más fotos. Iniciando entrenamiento del modelo.", "info")
                    self.current_user_registration = None
                    self.train_model(show_message_box=True)
                    self.gui_manager.update_user_list(self.user_manager.get_users()) 

        elif len(faces) != 1:
            self.update_status(f"Esperando un rostro para {username}. Fotos tomadas: {current_total_photos_for_user}")
            logger.debug(f"No se detectó un rostro o más de uno para {username}. Fotos actuales: {current_total_photos_for_user}")

    def train_model(self, show_message_box):
        """
        Inicia el entrenamiento del modelo facial en un hilo separado para no bloquear la GUI.
        """
        self.update_status("Estado: Entrenando modelo (esto puede tardar unos minutos)...")
        # Iniciar el entrenamiento en un hilo separado
        threading.Thread(target=self._perform_training_task, args=(show_message_box,), daemon=True).start()

    def _perform_training_task(self, show_message_box):
        """
        Tarea real de entrenamiento del modelo, ejecutada en un hilo separado.
        Args:
            show_message_box (bool): Si se debe mostrar un cuadro de mensaje de éxito al finalizar.
        """
        try:
            # Usar la instancia existente de face_trainer
            accuracy = self.face_trainer.train()
            msg = f"Modelo entrenado con precisión: {accuracy:.2f}"
            if show_message_box:
                self.gui_manager.root.after(0, lambda: self.gui_manager.show_message("Éxito", msg, "info"))
            # Recargar el reconocedor facial con el nuevo modelo, asegurar que se haga en el hilo principal si hay interacción con GUI
            self.gui_manager.root.after(0, self._reload_recognizer_model)
            self.gui_manager.root.after(0, lambda: self.update_status("Estado: Modelo entrenado exitosamente."))
        except Exception as e:
            error_msg = f"Error al entrenar el modelo: {str(e)}"
            self.gui_manager.root.after(0, lambda: self.gui_manager.show_message("Error", error_msg, "error"))
            self.gui_manager.root.after(0, lambda: self.update_status(f"Estado: Error al entrenar modelo - {str(e)}."))

    def _reload_recognizer_model(self):
        """
        Recarga el reconocedor facial con el modelo actualizado.
        Se asegura de que se llame en el hilo principal de la GUI.
        """
        self.face_recognizer = FaceRecognizer(
            MODEL_PATH,
            BASE_DIR,
            CONFIDENCE_THRESHOLD
        )
        logger.info("Reconocedor facial recargado con el nuevo modelo.")

    def start_recognition(self):
        """
        Inicia el proceso de reconocimiento facial en un hilo separado.
        También verifica si el PC está bloqueado y espera antes de iniciar la cámara.
        """
        logger.debug("FaceRecognitionApp: start_recognition llamado.")
        if self.recognition_active:
            self.update_status("Reconocimiento facial ya está activo.")
            logger.debug("FaceRecognitionApp: Reconocimiento ya activo, regresando.")
            return

        # Verificar si face_recognizer y face_trainer están inicializados y si el modelo existe
        if self.face_recognizer is None or self.face_trainer is None:
            logger.warning("FaceRecognitionApp: Modelo de reconocimiento o entrenador no inicializado.")
            self.gui_manager.show_message(
                "Advertencia",
                "El modelo de reconocimiento aún no está listo. Por favor, espere a que termine la configuración inicial o entrene el modelo si es necesario.",
                "warning"
            )
            return
        
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"FaceRecognitionApp: Modelo no encontrado en {MODEL_PATH}.")
            self.gui_manager.show_message(
                "Advertencia",
                "El modelo de reconocimiento no ha sido entrenado. Por favor, entrene el modelo primero.",
                "warning"
            )
            return

        self.recognition_active = True
        self.update_status("Iniciando verificación de bloqueo...")
        logger.info("FaceRecognitionApp: Iniciando hilo para verificar bloqueo y comenzar reconocimiento.")

        # Ejecutar el nuevo método para gestionar el estado de reconocimiento en segundo plano
        threading.Thread(target=self._manage_recognition_state, daemon=True).start()

    def _manage_recognition_state(self):
        """
        Gestiona el estado del reconocimiento facial basado en el bloqueo del PC.
        Se ejecuta en un hilo separado y comprueba el estado cada 10 segundos.
        """
        logger.debug("FaceRecognitionApp: _manage_recognition_state llamado. Bucle de gestión de estado iniciado.")
        # Flag para controlar si _run_recognition_process está actualmente en ejecución
        recognition_process_running = False

        while self.recognition_active:
            is_pc_locked = self.is_locked()

            if is_pc_locked:
                # PC está bloqueado, debería intentar activar el reconocimiento
                if not recognition_process_running:
                    self.gui_manager.root.after(0, lambda: self.update_status("PC bloqueado. Esperando 10 segundos para iniciar la cámara..."))
                    logger.info("FaceRecognitionApp: PC bloqueado. Esperando 10 segundos antes de activar la cámara.")
                    time.sleep(5) # Esperar antes de intentar iniciar la cámara
                    
                    # Asegurarse de que `self.recognition_thread` sea un nuevo hilo o no esté corriendo
                    if self.recognition_thread is None or not self.recognition_thread.is_alive():
                        self.recognition_thread = threading.Thread(target=self._run_recognition_process, daemon=True)
                        self.recognition_thread.start()
                        recognition_process_running = True
                        logger.info("FaceRecognitionApp: Hilo _run_recognition_process iniciado debido a PC bloqueado.")
                    else:
                        logger.info("FaceRecognitionApp: _run_recognition_process ya estaba ejecutándose.")
                else:
                    self.gui_manager.root.after(0, lambda: self.update_status("PC bloqueado. Reconocimiento activo."))
                    logger.debug("FaceRecognitionApp: PC bloqueado. Reconocimiento ya activo.")
            else:
                # PC no está bloqueado, el reconocimiento no debería estar activo
                if recognition_process_running:
                    self.gui_manager.root.after(0, lambda: self.update_status("PC desbloqueado. Deteniendo reconocimiento..."))
                    self.stop_recognition() # Llama al método para detener la cámara y el hilo
                    recognition_process_running = False
                    logger.info("FaceRecognitionApp: Reconocimiento detenido porque PC está desbloqueado.")
                else:
                    self.gui_manager.root.after(0, lambda: self.update_status("PC desbloqueado. Reconocimiento en espera. Funcionará cuando el PC esté bloqueado."))
                    logger.debug("FaceRecognitionApp: PC desbloqueado. Reconocimiento inactivo (como se espera).")

                # Mostrar la advertencia si el PC no está bloqueado
                # self.gui_manager.root.after(0, lambda: self.gui_manager.show_message(
                #     "Advertencia",
                #     "La función de reconocimiento facial se activará solo cuando el PC esté bloqueado.",
                #     "warning"
                # ))
                logger.info("FaceRecognitionApp: Advertencia - La función de reconocimiento facial se activará solo cuando el PC esté bloqueado.")
            
            time.sleep(10) # Esperar 10 segundos antes de la próxima comprobación de bloqueo
        
        # Al salir del bucle (self.recognition_active es False),
        # asegurar que el reconocimiento esté completamente detenido.
        if recognition_process_running:
            self.gui_manager.root.after(0, lambda: self.update_status("Deteniendo reconocimiento facial."))
            self.stop_recognition()
            logger.info("FaceRecognitionApp: Reconocimiento facial detenido al finalizar el bucle de gestión de estado.")
        else:
            self.gui_manager.root.after(0, lambda: self.update_status("Reconocimiento facial inactivo."))
            logger.info("FaceRecognitionApp: Bucle de gestión de estado finalizado. Reconocimiento ya inactivo.")

    def _run_recognition_process(self):
        """
        Bucle principal para el reconocimiento facial continuo.
        La cámara se enciende por 5 segundos y se apaga por 5 segundos, repitiendo mientras el PC esté bloqueado.
        """
        logger.debug("FaceRecognitionApp: _run_recognition_process llamado. Iniciando bucle de reconocimiento ON/OFF.")
        while self.recognition_active:
            # Encender la cámara
            if not self.camera_manager.start_capture(callback=self._process_recognition_frame):
                self.gui_manager.root.after(0, lambda: self.update_status("Error: No se pudo iniciar la cámara para reconocimiento."))
                self.recognition_active = False
                logger.error("FaceRecognitionApp: Fallo al iniciar la cámara para reconocimiento, terminando _run_recognition_process.")
                return
            logger.info("FaceRecognitionApp: Cámara iniciada para reconocimiento (5 segundos).")
            self.gui_manager.root.after(0, lambda: self.update_status("Reconocimiento facial activo (cámara encendida 5s)."))
            # Mantener la cámara encendida por 5 segundos
            t_end = time.time() + 5
            while self.recognition_active and self.camera_manager.is_recording and time.time() < t_end:
                time.sleep(0.1)
            # Apagar la cámara
            self.camera_manager.stop_camera()
            logger.info("FaceRecognitionApp: Cámara apagada por 5 segundos.")
            self.gui_manager.root.after(0, lambda: self.update_status("Reconocimiento facial en espera (cámara apagada 5s)."))
            # Esperar 5 segundos antes de volver a encender la cámara
            t_wait = time.time() + 5
            while self.recognition_active and time.time() < t_wait:
                time.sleep(0.1)
        # Al salir, asegurarse de que la cámara esté apagada
        self.camera_manager.stop_camera()
        self.gui_manager.root.after(0, lambda: self.update_status("Reconocimiento facial detenido."))
        logger.info("FaceRecognitionApp: _run_recognition_process terminado.")
    def _process_recognition_frame(self, frame):
        """
        Procesa cada frame durante el reconocimiento facial.
        
        Args:
            frame: Frame capturado por la cámara
        """
        if not self.recognition_active:
            return

        if not self.gui_manager.root.winfo_exists():
            return # La ventana ha sido cerrada, no procesar frames

        processed_frame, recognized_users = self.face_recognizer.process_frame(frame)
        self.gui_manager.show_frame(processed_frame)

        if recognized_users:
            username, confidence = recognized_users[0]
            self.update_status(
                f"Rostro de {username} reconocido con confianza {confidence:.0f}."
            )
            # Guardar la foto de reconocimiento si no se ha alcanzado el límite total
            if self.user_manager.get_user_photo_count(username) < MAX_PHOTOS_PER_USER:
                faces, gray = self.face_recognizer.detect_faces(frame) # Re-detectar para obtener el ROI
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (128, 128))
                    self.user_manager.save_recognition_photo_and_manage_retention(username, face_roi)
                    self.update_status("Foto de reconocimiento guardada y gestión de retención aplicada.")
                    self.train_model(show_message_box=False) # Re-entrenar el modelo después de añadir una foto, sin mostrar messagebox

            # Mostrar mensaje de bienvenida en la GUI y logear la autenticación
            self.gui_manager.update_welcome_message(f"Bienvenido {username}!")
            logger.info(f"Usuario '{username}' autenticado correctamente. Iniciando desbloqueo.")

            # Inicia el proceso de desbloqueo en un hilo separado para no bloquear la GUI
            threading.Thread(target=self.try_unlock, daemon=True).start()
            self.stop_recognition() # Detener el reconocimiento una vez que se reconoce a alguien
        else:
            self.update_status("Buscando rostro...")
            self.gui_manager.update_welcome_message("") # Limpiar el mensaje de bienvenida si no hay reconocimiento

    def try_unlock(self):
        """
        Intenta desbloquear el PC simulando la entrada del PIN/contraseña.
        También registra el evento de desbloqueo.
        """
        try:
            auth_logger.info(f"[{datetime.now()}] Intentando desbloquear PC...")
            auth_logger.info(f"[{datetime.now()}] Escribiendo PIN...")
            pyautogui.write("KarGise11e")
            auth_logger.info(f"[{datetime.now()}] Presionando 'enter'...")
            pyautogui.press('enter')
            time.sleep(1)  # Espera breve para confirmar desbloqueo
            auth_logger.info(f"[{datetime.now()}] PC aparentemente desbloqueado.")
            return True
        except Exception as e:
            auth_logger.error(f"[{datetime.now()}] Error al intentar desbloquear PC: {e}")
            return False

    def stop_recognition(self):
        """
        Detiene el proceso de reconocimiento facial.
        """
        if self.recognition_active:
            self.recognition_active = False
            self.camera_manager.stop_camera()
            self.update_status("Reconocimiento facial detenido.")

    def delete_user(self):
        """
        Elimina un usuario y todas sus fotos.
        """
        users = self.user_manager.get_users()
        if not users:
            self.gui_manager.show_message("Advertencia", "No hay usuarios para eliminar.", "warning")
            return

        # Obtener el usuario seleccionado del Combobox
        username_to_delete = self.gui_manager.user_list.get()
        
        if not username_to_delete: # Si no hay ningún usuario seleccionado
            self.gui_manager.show_message("Advertencia", "Por favor, selecciona un usuario para eliminar.", "warning")
            return

        if username_to_delete not in users:
            self.gui_manager.show_message("Error", f"El usuario '{username_to_delete}' no existe en la lista.", "error")
            return

        if self.gui_manager.ask_confirmation("Confirmar Eliminación", f"¿Está seguro que desea eliminar a {username_to_delete} y todas sus fotos?"):
            if self.user_manager.delete_user(username_to_delete):
                self.gui_manager.show_message("Éxito", f"Usuario {username_to_delete} eliminado exitosamente.", "info")
                self.update_user_list()
                # Re-entrenar el modelo después de eliminar un usuario
                self.train_model(show_message_box=True)
            else:
                self.gui_manager.show_message("Error", f"No se pudo eliminar el usuario {username_to_delete}.", "error")

    def is_locked(self):
        """
        Verifica si el PC está bloqueado usando loginctl.
        
        Returns:
            bool: True si el PC está bloqueado, False en caso contrario
        """
        try:
            # Ejecutar loginctl para obtener el estado de la sesión
            command = (
                "loginctl show-session "
                "$(loginctl | grep $(whoami) | awk '{print $1}') "
                "-p LockedHint"
            )
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Comprobar si el sistema está bloqueado
            is_locked = "yes" in result.stdout.lower()
            
            print(f"Verificando estado de bloqueo: {is_locked}")  # Debug print
            
            if is_locked:
                print("PC bloqueado.")
                return True
            else:
                print("PC desbloqueado.")
                return False
                
        except Exception as e:
            print(f"Error al verificar estado de bloqueo: {e}")
            return False

    def on_closing(self):
        """
        Maneja el evento de cierre de ventana.
        Asegura que todos los recursos se liberen correctamente.
        """
        if messagebox.askokcancel("Salir", "¿Desea salir de la aplicación?"):
            # Detener reconocimiento si está activo
            if self.recognition_active:
                self.stop_recognition()
            
            # Detener registro si está activo
            if self.registration_active:
                self.registration_active = False
            
            # Detener la cámara
            if hasattr(self, 'camera_manager'):
                self.camera_manager.stop_camera()  # Cambiado de stop_capture a stop_camera
            
            # Esperar a que los hilos terminen
            if hasattr(self, 'recognition_thread') and self.recognition_thread:
                self.recognition_thread.join(timeout=1.0)
            
            # Destruir la ventana principal
            self.root.destroy()
            
            # Forzar la salida del programa
            os._exit(0)

    def run(self):
        """
        Ejecuta la aplicación.
        """
        self.root.mainloop()


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()