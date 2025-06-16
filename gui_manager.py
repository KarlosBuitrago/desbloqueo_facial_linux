"""
Gestor de Interfaz Gráfica
========================

Este módulo maneja la interfaz gráfica del sistema de reconocimiento facial.

Características:
- Interfaz intuitiva y fácil de usar
- Visualización de video en tiempo real
- Gestión de usuarios mediante botones
- Mensajes y confirmaciones interactivas
"""

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import cv2
import logging

logger = logging.getLogger(__name__)

class GUIManager:
    """
    Clase para gestionar la interfaz gráfica.
    Esta clase proporciona funcionalidades para:
    - Crear y gestionar widgets
    - Mostrar video en tiempo real
    - Manejar eventos de usuario
    - Mostrar mensajes y confirmaciones
    """
    
    def __init__(self, root, resolution=(320, 240)):
        """
        Inicializa la interfaz gráfica.
        
        Args:
            root (tk.Tk): Ventana principal
            resolution (tuple): Resolución del video
        """
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("800x600")
        
        self.resolution = resolution
        self.status_callback = None
        self.register_callback = None
        self.recognition_callback = None
        self.delete_callback = None
        
        self.create_widgets()
        
        # Configurar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """
        Crea y configura los widgets de la interfaz.
        
        Crea:
        - Frame principal
        - Frame de botones
        - Botones de acción
        - Lista de usuarios
        - Etiqueta de estado
        - Canvas para video
        - Etiqueta para el mensaje de bienvenida/reconocimiento
        """
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Frame para botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=5)
        
        # Botones
        self.register_btn = tk.Button(button_frame, text="Registrar Usuario", 
                                    command=self._on_register)
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        self.activate_btn = tk.Button(button_frame, text="Activar Reconocimiento", 
                                    command=self._on_recognition)
        self.activate_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_recognition_btn = tk.Button(button_frame, text="Desactivar Reconocimiento", command=self.stop_recognition)
        self.stop_recognition_btn.pack(side=tk.LEFT, padx=5)
        
        self.delete_btn = tk.Button(button_frame, text="Eliminar Usuario",
                                  command=self._on_delete)
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Lista de usuarios
        self.user_list = ttk.Combobox(main_frame, state='readonly')
        self.user_list.pack(pady=5)
        
        # Label para mostrar estado
        self.status_label = tk.Label(main_frame, text="Estado: Inactivo")
        self.status_label.pack(pady=5)
        
        # Canvas para mostrar video
        self.canvas = tk.Canvas(main_frame, width=self.resolution[0], 
                              height=self.resolution[1])
        self.canvas.pack(pady=10)
        
        # Etiqueta para el mensaje de bienvenida/reconocimiento
        self.welcome_message_label = tk.Label(main_frame, text="", font=("Arial", 14, "bold"), fg="green")
        self.welcome_message_label.pack(pady=5)
        
    def set_callbacks(self, status_callback=None, register_callback=None,
                     recognition_callback=None, delete_callback=None,
                     stop_recognition_callback=None):
        """
        Configura las funciones callback.
        
        Args:
            status_callback (callable): Función para actualizar estado
            register_callback (callable): Función para registro de usuario
            recognition_callback (callable): Función para activar reconocimiento
            delete_callback (callable): Función para eliminar usuario
            stop_recognition_callback (callable): Función para detener reconocimiento
        """
        self.status_callback = status_callback
        self.register_callback = register_callback
        self.recognition_callback = recognition_callback
        self.delete_callback = delete_callback
        self.stop_recognition_callback = stop_recognition_callback
        
    def update_user_list(self, users):
        """
        Actualiza la lista de usuarios en el dropdown.
        
        Args:
            users (list): Lista de nombres de usuarios
        """
        self.user_list['values'] = users
        if users:
            self.user_list.set(users[0])
            
    def update_status(self, status):
        """
        Actualiza el texto de estado.
        
        Args:
            status (str): Nuevo estado a mostrar
        """
        self.status_label.config(text=f"Estado: {status}")
        
    def show_frame(self, frame):
        """
        Muestra un frame de video en el canvas.
        
        Args:
            frame (numpy.ndarray): Frame de video a mostrar
        """
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        except Exception as e:
            print(f"Error al mostrar frame: {str(e)}")
            
    def update_welcome_message(self, message):
        """
        Actualiza el mensaje de bienvenida debajo del canvas del video.
        
        Args:
            message (str): El mensaje a mostrar (e.g., "Bienvenido [nombre]!").
        """
        self.welcome_message_label.config(text=message)
        
    def show_message(self, title, message, type="info"):
        """
        Muestra un mensaje al usuario.
        
        Args:
            title (str): Título del mensaje
            message (str): Contenido del mensaje
            type (str): Tipo de mensaje (info/error/warning)
        """
        if type == "info":
            messagebox.showinfo(title, message)
        elif type == "error":
            messagebox.showerror(title, message)
        elif type == "warning":
            messagebox.showwarning(title, message)
            
    def ask_confirmation(self, title, message):
        """
        Solicita confirmación al usuario.
        
        Args:
            title (str): Título del diálogo
            message (str): Mensaje de confirmación
            
        Returns:
            bool: True si el usuario confirma
        """
        return messagebox.askyesno(title, message)
        
    def ask_string(self, title, prompt):
        """
        Solicita una cadena de texto al usuario.
        
        Args:
            title (str): Título del diálogo
            prompt (str): Mensaje de solicitud
            
        Returns:
            str: Texto ingresado por el usuario
        """
        return simpledialog.askstring(title, prompt)
        
    def _on_register(self):
        """Maneja el evento de registro de usuario"""
        if self.register_callback:
            self.register_callback()
            
    def _on_recognition(self):
        """Maneja el evento de activación de reconocimiento"""
        logger.debug("GUIManager: _on_recognition llamado. Intentando ejecutar recognition_callback.")
        if self.recognition_callback:
            self.recognition_callback()
            
    def _on_delete(self):
        """Maneja el evento de eliminación de usuario"""
        if self.delete_callback:
            self.delete_callback()
            
    def stop_recognition(self):
        """
        Detiene el reconocimiento facial.
        """
        if hasattr(self, 'stop_recognition_callback') and self.stop_recognition_callback:
            self.stop_recognition_callback()       
            
    def on_closing(self):
        """
        Maneja el evento de cierre de ventana.
        
        Solicita confirmación antes de cerrar la aplicación.
        """
        if messagebox.askokcancel("Salir", "¿Desea salir de la aplicación?"):
            self.root.destroy()
            
    def run(self):
        """
        Inicia el bucle principal de la interfaz.
        """
        self.root.mainloop() 