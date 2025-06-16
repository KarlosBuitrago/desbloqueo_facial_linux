"""
Reconocedor Facial usando Red Neuronal
=====================================

Este módulo implementa el reconocimiento facial utilizando
una red neuronal convolucional pre-entrenada.
"""

import os
import cv2
import numpy as np
import logging
# import tensorflow as tf # Ya no es necesario para Eigenfaces

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Clase para el reconocimiento facial usando EigenFaces.
    
    Esta clase maneja:
    - Carga del modelo entrenado
    - Detección de rostros
    - Reconocimiento facial
    - Predicción de identidad
    """
    
    def __init__(self, model_path, base_dir, confidence_threshold=6000):
        """
        Inicializa el reconocedor facial.
        
        Args:
            model_path (str): Ruta al modelo entrenado (.xml)
            base_dir (str): Directorio base con las carpetas de usuarios
            confidence_threshold (float): Umbral de confianza para predicciones
                                        (para Eigenfaces, un valor más BAJO es mejor)
        """
        self.model_path = model_path
        self.base_dir = base_dir
        self.confidence_threshold = confidence_threshold # Umbral de distancia para Eigenfaces
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        self.user_list = []
        self.load_model()
        
    def load_model(self):
        """
        Carga el modelo entrenado desde el archivo XML.
        Actualiza la lista de usuarios según los directorios.
        """
        if os.path.exists(self.model_path):
            try:
                self.face_recognizer.read(self.model_path)
                # Asegurarse de que el orden de los usuarios sea el mismo que al entrenar
                self.user_list = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
                self.user_list.sort() # Importante mantener el orden
                logger.info(f"Modelo EigenFaces cargado desde {self.model_path}. Usuarios: {self.user_list}")
                return True
            except Exception as e:
                logger.error(f"Error al cargar el modelo EigenFaces: {str(e)}")
                self.face_recognizer = cv2.face.EigenFaceRecognizer_create() # Reinicializar
                return False
        else:
            logger.warning(f"El modelo EigenFaces no se encontró en {self.model_path}. Entrenar el modelo primero.")
            return False
        
    def detect_faces(self, frame):
        """
        Detecta rostros en un frame.
        
        Args:
            frame: Frame de video o imagen
            
        Returns:
            tuple: (faces, gray) - Rostros detectados e imagen en escala de grises
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        return faces, gray
        
    def recognize_face(self, face_roi):
        """
        Reconoce un rostro usando el modelo EigenFaces.
        
        Args:
            face_roi: Región de interés del rostro
            
        Returns:
            tuple: (username, confidence) - Nombre del usuario y confianza
        """
        if not self.user_list or not os.path.exists(self.model_path):
            return None, 0.0

        # Preprocesar el rostro
        face_roi = cv2.resize(face_roi, (128, 128))
        
        try:
            # Realizar predicción con Eigenfaces
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Para Eigenfaces, un valor de 'confidence' más bajo significa mayor similitud (menor distancia)
            # Por eso, el umbral de confianza debe ser un valor de distancia MÁXIMO.
            # Si la 'confidence' (distancia) es menor que el umbral, consideramos que hay un reconocimiento.
            if confidence < self.confidence_threshold and label < len(self.user_list):
                username = self.user_list[label]
                # Para consistencia con la CNN, podemos invertir la 'confianza' visualmente
                # para que un valor más alto signifique mejor coincidencia (e.g., 1 - (distancia/max_distancia))
                # Aquí simplemente pasamos la distancia (confidence) tal cual.
                return username, confidence
            else:
                return None, confidence # Devolvemos la distancia incluso si no reconoce
        except Exception as e:
            logger.error(f"Error en reconocimiento con EigenFaces: {str(e)}")
            return None, 0.0
        
    def process_frame(self, frame):
        """
        Procesa un frame completo para reconocimiento facial.
        
        Args:
            frame: Frame de video o imagen
            
        Returns:
            tuple: (frame, recognized_users) - Frame procesado y usuarios reconocidos
        """
        faces, gray = self.detect_faces(frame)
        recognized_users = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            username, confidence = self.recognize_face(face_roi)
            
            if username:
                recognized_users.append((username, confidence))
                # Para Eigenfaces, mostrar la 'confianza' como la distancia es más preciso
                # Un valor más bajo es mejor.
                text = f'{username}: {confidence:.0f}' 
                color = (0, 255, 0) # Verde para reconocido
            else:
                text = f'Desconocido: {confidence:.0f}'
                color = (0, 0, 255) # Rojo para desconocido

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
        return frame, recognized_users 