import polyscope as ps
import numpy as np
import torch
from loguru import logger
import os
import json

from utils.evaluation_metrics_fast import _pairwise_EMD_CD_, knn, print_results
from utils.data_helper import normalize_point_clouds

def init_polyscope():
    # Init polyscope
    ps.set_allow_headless_backends(True)
    ps.init()

    # Polyscope set-ups
    ps.set_up_dir("neg_z_up")            # -Z Up
    ps.set_front_dir("y_front")          # Y Front
    ps.set_view_projection_mode("orthographic")
    ps.set_screenshot_extension(".png") 
    ps.set_ground_plane_mode("none")  # Desactiva la cuadrícula/base

def get_color_from_index(idx):
    # Red -: Generated
    # Green -: Reference
    return (1, 0, 0) if idx < 405 else (0, 1, 0)

def draw_and_save_in_png(min_vals, min_idx, sample_pc, ref_pcs):
    if not ps.is_initialized():
        init_polyscope()

    ps.remove_all_structures()

    sample_cloud = sample_pc.squeeze(0).cpu().numpy()
    cloud_1 = ref_pcs[min_idx[1]].squeeze(0).cpu().numpy()
    cloud_2 = ref_pcs[min_idx[2]].squeeze(0).cpu().numpy()
    cloud_3 = ref_pcs[min_idx[3]].squeeze(0).cpu().numpy()

    sample_cloud += np.array([0, -1, 0])
    cloud_1     += np.array([0.7, 1, 0.2])
    cloud_2     += np.array([1.4, -1, 0])
    cloud_3     += np.array([2.1, 1, 0.2])

    # Register point clouds
    pc_register = ps.register_point_cloud("PC sample", sample_cloud)
    pc_register.set_color((0,0,1))
     
    pc_register2 = ps.register_point_cloud(f"PC ref {min_idx[1]}", cloud_1)
    pc_register2.set_color(get_color_from_index(min_idx[1]))

    pc_register3 = ps.register_point_cloud(f"PC ref {min_idx[2]}", cloud_2)
    pc_register3.set_color(get_color_from_index(min_idx[2]))

    pc_register4 = ps.register_point_cloud(f"PC ref {min_idx[3]}", cloud_3)
    pc_register4.set_color(get_color_from_index(min_idx[3]))

    ss_path = os.path.join(path_to_save_ss, f"Screenshot_{min_idx[0]}.png")
    ps.screenshot(ss_path)
    print(f"SS guardado en {ss_path}")

    return ss_path

def compute_all_metrics(sample_pcs, ref_pcs, labels, batch_size,
                        verbose=True, accelerated_cd=False, metric1='CD', **print_kwargs):
    results = {}
    
    if verbose:
        logger.info("Pairwise CD")
    
    batch_size = ref_pcs.shape[0] // 2 if ref_pcs.shape[0] != batch_size else batch_size
    v1 = False

    # --- eval CD results --- #
    metric = metric1  # 'CD'
    if verbose:
        logger.info('eval metric: {}; batch-size={}, device: {}, {}',
                    metric, batch_size, ref_pcs.device, sample_pcs.device)

    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                          sample_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)
    
    #logger.info('M_rs_cd: {}', M_rs_cd) # This prints

    #print(M_rs_cd.shape)

    min_vals, min_idx = torch.topk(M_rs_cd.squeeze(), k=4, largest=False)

    saved_path = draw_and_save_in_png(min_vals, min_idx, sample_pcs, ref_pcs)
    
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(metric, ref_pcs,
                                          ref_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(metric, sample_pcs,
                                          sample_pcs,
                                          batch_size,
                                          accelerated_cd=accelerated_cd,
                                          require_grad=False, verbose=v1)
    
    #logger.info('M_rr_cd: {}, M_ss_cd: {}', M_rr_cd, M_ss_cd)
    
    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update(
        {"1-NN-%s-%s" % (metric, k): v.item()
         for k, v in one_nn_cd_res.items() if 'acc' in k})
    # logger.info('results: {}', results)
    if verbose:
        print_results(results, **print_kwargs)

    # Save values in dict
    diccionario[saved_path] = {
        "Indexes": f"{min_idx[1].item()}. {min_idx[2].item()}, {min_idx[3].item()}",
        "M_rs_cd": f"{min_vals[1].item()}. {min_vals[2].item()}, {min_vals[3].item()}",
        "1-NN-CD-acc_t": results['1-NN-CD-acc_t'],
        "1-NN-CD-acc": results['1-NN-CD-acc']
    }

    return results

@torch.no_grad()
def compute_score_own(pcs_path, batch_size_test=810, device_str='cuda',
                    device=None, accelerated_cd=True, writer=None, exp=None,
                    norm_box=True, skip_write=False, **print_kwargs):
    """
    Args: 
        pcs_path (str) path to clouds
        print_kwargs (dict): entries: dataset, hash, step, epoch; 
    """
    
    device = torch.device(device_str) if device is None else device

    # 1. Load al the .npy files to memory
    all_data = []
    labels = []

    for f in sorted(os.listdir(pcs_path)):
        if f.endswith(".npy"):
            full_path = os.path.joins(pcs_path, f)
            pc = np.load(full_path)
            all_data.append(pc)

            if "generated" in f:
                labels.append("generated")
            elif "reference" in f:
                labels.append("reference")

    """ all_files = sorted([f for f in os.listdir(pcs_path) if f.endswith(".npy")])
    all_paths = [os.path.join(pcs_path, f) for f in all_files]
    all_pcs = [np.load(f) for f in all_paths] """

    # Validate shape
    N = all_data[0].shape[0]
    for i, pc in enumerate(all_pcs):
        if pc.shape != (N, 3):
            raise ValueError(f"{all_data[i]} has shape {pc.shape}, but ({N}, 3) was expected.")
        
    # Convert to tensor with shape (B, N, 3)
    all_pcs = torch.from_numpy(np.stack(all_data)).float().to(device)

    # Normalize point clouds
    if norm_box:
        all_pcs = 0.5 * torch.stack(normalize_point_clouds(all_pcs), dim=0)
        print_kwargs['dataset'] = print_kwargs.get('dataset', '') + '-normbox'
    
    logger.info("print_kwargs: {}", print_kwargs)

    # 2. Iterate over each cloud
    for i in range(len(all_pcs)):
        current_pc = all_pcs[i].unsqueeze(0) # Convert from (N, 3) to (1, N, 3)
        #all_ref_pcs = torch.cat([all_pcs[:i], all_pcs[i+1:]], dim=0) # (809, N, 3) \\ To index i, From index i+1

        logger.info('[data] shape: {}, current_pc: {}, all_pcs: {}', 
                                        all_pcs.shape, current_pc.shape, all_pcs.shape)
        
        results = compute_all_metrics(current_pc.to(device).float(),
                                        all_pcs.to(device).float(), labels, batch_size_test, 
                                        accelerated_cd=accelerated_cd, **print_kwargs)
            
        #print("Estos son los resultados: ", results)

        #quit()

    # Save dict
    json_output_path = os.path.join(path_to_save_json, "resultados_metrics.json")

    with open(json_output_path, "w") as f:
        json.dump(diccionario, f, indent=4)

    print("Done!")

    return 


pcs_path = "/home/ncaytuir/data-local/LION_necs/PRUEBAS/all_objects"
path_to_save_ss = "/home/ncaytuir/data-local/LION_necs/PRUEBAS/screenshots"
path_to_save_json = "/home/ncaytuir/data-local/LION_necs/PRUEBAS"

diccionario = {}

init_polyscope()
compute_score_own(pcs_path=pcs_path)