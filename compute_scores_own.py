from utils.eval_helper import compute_score 
# samples = sys.argv[1]
# ref = sys.argv[2]

samples = '/home/ncaytuir/data-local/PVD_necs/output/ckpt_original_samples.pth'
ref = '/home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_airplane.pt'
compute_score(samples, ref_name=ref, norm_box=True)