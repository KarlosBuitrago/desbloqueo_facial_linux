"""
Detector de Rostros
=================

Este módulo implementa la funcionalidad de detección y reconocimiento
facial utilizando OpenCV y el clasificador Haar Cascade.

Características:
- Detección de rostros en tiempo real
- Reconocimiento facial con LBPH
- Entrenamiento automático
- Gestión de datos de entrenamiento
"""

import cv2
import numpy as np
from datetime import datetime
import os

class FaceDetector:
    """
    Clase para la detección y reconocimiento facial.
    
    Esta clase proporciona funcionalidades para:
    - Detectar rostros en imágenes
    - Entrenar el reconocedor facial
    - Reconocer rostros
    - Guardar fotos de entrenamiento
    """
    
    def __init__(self, base_dir, max_training_photos=10, confidence_threshold=0.6):
        """
        Inicializa el detector de rostros.
        
        Args:
            base_dir (str): Directorio base para las fotos
            max_training_photos (int): Máximo de fotos para entrenamiento
            confidence_threshold (float): Umbral de confianza para reconocimiento
        """
        self.base_dir = base_dir
        self.max_training_photos = max_training_photos
        self.confidence_threshold = confidence_threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.training_data = {}
        
        if self.face_cascade.empty():
            raise Exception("No se pudo cargar el clasificador de rostros")
            
    def detect_faces(self, frame):
        """
        Detecta rostros en un frame.
        
        Args:
            frame: Frame de video a procesar
            
        Returns:
            tuple: (rostros_detectados, frame_en_escala_de_grises)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces, gray
        
    def load_training_data(self, username):
        """
        Carga los datos de entrenamiento para un usuario.
        
        Args:
            username (str): Nombre del usuario
            
        Returns:
            tuple: (datos_entrenamiento, etiquetas) o None si no hay datos
        """
        if username in self.training_data:
            return self.training_data[username]
            
        user_dir = os.path.join(self.base_dir, username)
        if not os.path.exists(user_dir):
            return None
            
        photos = [os.path.join(user_dir, f) for f in os.listdir(user_dir) 
                 if f.endswith('.jpg')][-self.max_training_photos:]
        
        if not photos:
            return None
            
        training_data = []
        labels = []
        
        for i, photo in enumerate(photos):
            img = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                training_data.append(img)
                labels.append(i)
                
        if training_data:
            self.training_data[username] = (training_data, np.array(labels))
            return self.training_data[username]
        return None
        
    def recognize_face(self, face_roi):
        """
        Reconoce un rostro en una región de interés.
        
        Args:
            face_roi: Región de interés con el rostro
            
        Returns:
            list: Lista de tuplas (usuario, confianza) reconocidos
        """
        face_roi = cv2.resize(face_roi, (100, 100))
        recognized_users = []
        
        for username in os.listdir(self.base_dir):
            if not os.path.isdir(os.path.join(self.base_dir, username)):
                continue
                
            training_data = self.load_training_data(username)
            if not training_data:
                continue
                
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.train(training_data[0], training_data[1])
            
            try:
                label, confidence = face_recognizer.predict(face_roi)
                confidence = 1 - (confidence / 100)
                
                if confidence > self.confidence_threshold:
                    recognized_users.append((username, confidence))
            except Exception:
                continue
                
        return recognized_users
        
    def save_face(self, username, frame, face_coords):
        """
        Guarda un rostro detectado.
        
        Args:
            username (str): Nombre del usuario
            frame: Frame completo
            face_coords: Coordenadas del rostro (x, y, w, h)
            
        Returns:
            str: Ruta del archivo guardado
        """
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        
        user_dir = os.path.join(self.base_dir, username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = os.path.join(user_dir, f"photo_{timestamp}.jpg")
        cv2.imwrite(photo_path, face_img)
        
        return photo_path 