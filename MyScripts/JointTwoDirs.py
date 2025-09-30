import os
import shutil

dir_ref = "/home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/eval_405objects/complete_point_clouds"
dir_gen = "/home/ncaytuir/data-local/LION_necs/datasets/test_data/pcs_airplane"
dir_out = "/home/ncaytuir/data-local/LION_necs/PRUEBAS/all_objects"

for filename in os.listdir(dir_ref):
    if filename.endswith(".npy"):
        source_path = os.path.join(dir_ref, filename)
        dest_path = os.path.join(dir_out, f"reference_{filename}")
        shutil.copy2(source_path, dest_path)

for filename in os.listdir(dir_gen):
    if filename.endswith(".npy"):
        source_path = os.path.join(dir_gen, filename)
        dest_path = os.path.join(dir_out, f"generated_{filename}")
        shutil.copy2(source_path, dest_path)

print("Hecho")