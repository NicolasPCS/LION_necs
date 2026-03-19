from utils.eval_helper import compute_score 
# samples = sys.argv[1]
# ref = sys.argv[2]

#samples = '/home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Airplane/generated_pth/samples_ours_ckpt7999.pth'
samples = '/home/ncaytuir/data/Datasets/Resultados_XCube/OverHalf/gen_pth/reference_airplane_xcube_2048.pth'
ref = '/home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth'
compute_score(samples, ref_name=ref, norm_box=False)