"""
Entrenador de Red Neuronal para Reconocimiento Facial
===================================================

Este módulo implementa el entrenamiento de una red neuronal convolucional
para el reconocimiento facial, utilizando TensorFlow y Keras.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class FaceTrainer:
    """
    Clase para entrenar el modelo de reconocimiento facial usando EigenFaces.

    Esta clase maneja:
    - Carga de imágenes de entrenamiento
    - Preprocesamiento de datos
    - Entrenamiento del modelo EigenFaces
    - Guardado del modelo entrenado
    """

    def __init__(self, base_dir, face_recognizer):
        """
        Inicializa el entrenador.

        Args:
            base_dir (str): Directorio base donde están las carpetas de usuarios
            face_recognizer (FaceRecognizer): Instancia del reconocedor de rostros (que también detecta)
        """
        self.base_dir = base_dir
        self.face_recognizer_instance = face_recognizer # Usar el reconocedor de rostros inyectado
        self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        self.user_list = []

    def load_training_data(self):
        """
        Carga las imágenes de entrenamiento de todos los usuarios.

        Returns:
            tuple: (faces_data, labels) - Datos de rostros y etiquetas
        """
        faces_data = []
        labels = []
        label_id = 0

        # Obtener lista de usuarios
        self.user_list = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        self.user_list.sort() # Asegurar orden consistente de labels

        # Cargar imágenes por usuario
        for username in self.user_list:
            user_path = os.path.join(self.base_dir, username)
            logger.info(f'Cargando imágenes del usuario: {username}')

            for filename in os.listdir(user_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(user_path, filename)
                    # logger.debug(f'Rostro: {filename}') # Desactivado para evitar mucho output

                    # Cargar y preprocesar imagen
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        logger.warning(f"No se pudo cargar la imagen {image_path}")
                        continue
                    
                    # Ya no necesitamos detectar rostros aquí, ya que las imágenes guardadas
                    # se asumen que ya contienen un solo rostro y están preprocesadas.
                    # Solo necesitamos asegurarnos de que tengan el tamaño correcto para el entrenamiento.
                    
                    # Asegurarse de que la imagen sea de 128x128 (tamaño esperado por Eigenfaces)
                    if image.shape[0] != 128 or image.shape[1] != 128:
                        face_roi = cv2.resize(image, (128, 128))
                        logger.debug(f"Imagen {filename} redimensionada a (128, 128) para entrenamiento.")
                    else:
                        face_roi = image

                    faces_data.append(face_roi)
                    labels.append(label_id)

            label_id += 1
            
        return np.array(faces_data), np.array(labels)
        
    def train(self, test_size=0.2):
        """
        Entrena el modelo de reconocimiento facial.
        
        Args:
            test_size (float): Proporción de datos para prueba
            
        Returns:
            float: Precisión del modelo en el conjunto de prueba
        """
        # Cargar datos
        faces_data, labels = self.load_training_data()
        
        if len(faces_data) == 0:
            logger.error("No se encontraron imágenes para entrenar.")
            raise ValueError("No se encontraron imágenes para entrenar.")
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            faces_data, labels, test_size=test_size, random_state=42
        )
        
        logger.info('Entrenamiento en proceso...')
        self.face_recognizer.train(X_train, y_train)
        logger.info('Entrenamiento finalizado')
        
        # Evaluar modelo
        correct = 0
        total = len(X_test)
        if total > 0:
            for i in range(total):
                label, confidence = self.face_recognizer.predict(X_test[i])
                # Para Eigenfaces, menor confianza (distancia) es mejor
                # Puedes ajustar el umbral aquí para una evaluación más estricta
                if label == y_test[i]: #  and confidence < UMBRAL_AQUI_SI_ES_NECESARIO
                    correct += 1
                    
            accuracy = correct / total
        else:
            accuracy = 0.0
            logger.warning("No hay datos de prueba para evaluar la precisión.")
        
        logger.info(f'\nPrecisión en los datos de prueba: {accuracy:.2f}')
        
        # Guardar modelo
        logger.info('Guardando modelo entrenado...')
        self.face_recognizer.save('modeloEigenFaces.xml')
        logger.info('Modelo almacenado')
        
        return accuracy 