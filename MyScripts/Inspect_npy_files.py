import os
import numpy as np

# Ruta del directorio que contiene los .npy
input_dir = "/home/ncaytuir/data-local/LION_necs/PRUEBAS/all_objects"

# Recorrer todos los archivos .npy
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_dir, filename)
        
        try:
            data = np.load(filepath)
            print(f"{filename}: {data.shape[0]} puntos, dimensión: {data.shape}")
        except Exception as e:
            print(f"❌ Error al leer {filename}: {e}")
