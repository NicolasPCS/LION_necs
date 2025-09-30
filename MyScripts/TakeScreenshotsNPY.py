import polyscope as ps
import numpy as np
import argparse
import os

# Argument parser
input_path = "/home/ncaytuir/data-local/LION_necs/datasets/test_data/pcs_airplane"
output_path = "/home/ncaytuir/data-local/LION_necs/datasets/test_data/screenshots_airplane"

# Init polyscope
ps.set_allow_headless_backends(True)
ps.init()

# Polyscope set-ups
ps.set_up_dir("neg_z_up")            # -Z Up
ps.set_front_dir("y_front")          # Y Front
ps.set_view_projection_mode("orthographic")
ps.set_screenshot_extension(".png") 
ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

cont = 1

for filename in os.listdir(input_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_path, filename)
        
        # Load point cloud
        points = np.load(file_path)
        
        print(f"Cantidad de puntos en la nube {cont} {points.shape}")

        ps.remove_all_structures()

        # Register point clouds
        pc = ps.register_point_cloud("PC", points)
        pc.set_color((0,0,1))
        pc.add_scalar_quantity("Scalar", points[:, 0], enabled=True)

        #ps.show()

        ss_path = os.path.join(output_path, filename.replace(".npy", ".png"))
        ps.screenshot(ss_path)
        print(f"SS guardado en {ss_path}")