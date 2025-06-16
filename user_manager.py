"""
Gestor de Usuarios
================

Este módulo maneja la gestión de usuarios y sus fotos en el sistema
de reconocimiento facial.

Características:
- Creación y eliminación de usuarios
- Gestión de fotos por usuario con reglas de retención avanzadas
- Mantener límites de almacenamiento
"""

import os
import shutil
from datetime import datetime, timedelta
import cv2

class UserManager:
    """
    Clase para gestionar usuarios y sus fotos.
    
    Esta clase proporciona funcionalidades para:
    - Crear y eliminar usuarios
    - Gestionar fotos de usuarios con reglas de retención avanzadas
    - Mantener límites de almacenamiento
    """
    
    def __init__(self, base_dir, max_photos_per_user=100, photo_retention_days=90, min_recent_photos_to_keep=50):
        """
        Inicializa el gestor de usuarios.
        
        Args:
            base_dir (str): Directorio base para usuarios
            max_photos_per_user (int): Máximo de fotos por usuario (límite superior)
            photo_retention_days (int): Días de retención para fotos antiguas
            min_recent_photos_to_keep (int): Número mínimo de fotos recientes a conservar
        """
        self.base_dir = base_dir
        self.max_photos_per_user = max_photos_per_user
        self.photo_retention_days = photo_retention_days
        self.min_recent_photos_to_keep = min_recent_photos_to_keep
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
    def create_user(self, username):
        """
        Crea un nuevo usuario.
        
        Args:
            username (str): Nombre del usuario
            
        Returns:
            bool: True si se creó correctamente, False si ya existe
        """
        user_dir = os.path.join(self.base_dir, username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            return True
        return False # Usuario ya existe
        
    def delete_user(self, username):
        """
        Elimina un usuario y todas sus fotos.
        
        Args:
            username (str): Nombre del usuario
            
        Returns:
            bool: True si se eliminó correctamente
        """
        user_dir = os.path.join(self.base_dir, username)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            return True
        return False
        
    def get_users(self):
        """
        Obtiene la lista de usuarios registrados.
        
        Returns:
            list: Lista de nombres de usuarios
        """
        return [d for d in os.listdir(self.base_dir) 
                if os.path.isdir(os.path.join(self.base_dir, d))]
                
    def get_user_photo_files(self, username):
        """
        Obtiene la lista de rutas completas de las fotos de un usuario, ordenadas por fecha de creación.
        
        Args:
            username (str): Nombre del usuario
            
        Returns:
            list: Lista de tuplas (ruta_completa, timestamp_creacion)
        """
        user_dir = os.path.join(self.base_dir, username)
        if not os.path.exists(user_dir):
            return []
            
        photo_files = [
            (os.path.join(user_dir, f), os.path.getctime(os.path.join(user_dir, f)))
            for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        photo_files.sort(key=lambda x: x[1]) # Ordenar por timestamp de creación (ascendente)
        return photo_files

    def get_user_photo_count(self, username):
        """
        Obtiene el número de fotos de un usuario.
        """
        return len(self.get_user_photo_files(username))
        
    def add_photo_to_user(self, username, face_roi):
        """
        Guarda una nueva foto para el usuario.
        
        Args:
            username (str): Nombre del usuario
            face_roi (numpy.ndarray): Región de interés del rostro (imagen en escala de grises)
            
        Returns:
            int: El nuevo conteo de fotos para el usuario.
        """
        user_dir = os.path.join(self.base_dir, username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        photo_path = os.path.join(user_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(photo_path, face_roi)
        return self.get_user_photo_count(username)

    def save_recognition_photo_and_manage_retention(self, username, face_roi):
        """
        Guarda una foto de reconocimiento y aplica la lógica de retención.
        
        Args:
            username (str): Nombre del usuario
            face_roi (numpy.ndarray): Región de interés del rostro (imagen en escala de grises)
        """
        self.add_photo_to_user(username, face_roi) # Guarda la nueva foto de reconocimiento

        user_photos = self.get_user_photo_files(username)
        current_total_photos = len(user_photos)
        
        # Si el usuario tiene más de MAX_PHOTOS_PER_USER, eliminamos el excedente
        photos_to_delete_for_limit = max(0, current_total_photos - self.max_photos_per_user)

        # Identificar fotos antiguas que no sean de las más recientes MIN_RECENT_PHOTOS_TO_KEEP
        oldest_acceptable_date = datetime.now() - timedelta(days=self.photo_retention_days)
        
        photos_eligible_for_deletion = []
        for i, (photo_path, timestamp) in enumerate(user_photos):
            photo_date = datetime.fromtimestamp(timestamp)
            is_older_than_retention = photo_date < oldest_acceptable_date
            is_not_among_recent_n = (current_total_photos - 1 - i) >= self.min_recent_photos_to_keep # Index from end

            if is_older_than_retention and is_not_among_recent_n:
                photos_eligible_for_deletion.append(photo_path)

        # Combinar y priorizar la eliminación: primero las más antiguas que cumplen criterios de retención, 
        # luego si aún hay muchas, eliminar de las restantes hasta el límite superior.
        photos_to_actually_delete = []
        
        # Eliminar fotos antiguas primero que no sean las 50 más recientes
        for photo_path in photos_eligible_for_deletion:
            if os.path.exists(photo_path):
                os.remove(photo_path)
                # print(f"[UserManager] Eliminada foto antigua (>{self.photo_retention_days} días) y no reciente: {os.path.basename(photo_path)}")
                current_total_photos -= 1

        # Si todavía hay demasiadas fotos después de la limpieza por antigüedad/recencia, eliminar las más antiguas restantes
        if current_total_photos > self.max_photos_per_user:
            user_photos_after_first_pass = self.get_user_photo_files(username)
            num_to_trim = current_total_photos - self.max_photos_per_user
            
            for i in range(num_to_trim):
                if i < len(user_photos_after_first_pass):
                    photo_path_to_remove = user_photos_after_first_pass[i][0]
                    if os.path.exists(photo_path_to_remove):
                        os.remove(photo_path_to_remove)
                        # print(f"[UserManager] Eliminada foto para reducir al límite: {os.path.basename(photo_path_to_remove)}")
                else:
                    break # No more photos to remove
        # print(f"[UserManager] Fotos finales para {username}: {self.get_user_photo_count(username)}") 