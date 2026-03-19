import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.eval_helper import compute_score

generated_samples = "/home/ncaytuir/data-local/LION_necs/lion_ckpt/unconditional/airplane/eval/samples_407999s1Hedbb5diet.pt"
reference_samples = "/home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_airplane.pt"

#print(reference_samples['ref'])

# Solo pasar rutas, NO cargar con torch.load
compute_score(generated_samples, ref_name=reference_samples, norm_box=True)

"""
2025-06-10 11:26:09.639 | INFO     | utils.eval_helper:compute_score:229 - [compute sample metric] sample: /home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/eval_410objects/complete_point_clouds/complete_generated_shapes.pt and ref: /home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_airplane.pt
[DEBUG] ref type: <class 'dict'>
2025-06-10 11:26:09.656 | INFO     | utils.eval_helper:compute_score:253 - [data shape] ref_pcs: torch.Size([405, 2048, 3]), gen_pcs: torch.Size([410, 2048, 3]), mean=torch.Size([405, 1, 3]), std=torch.Size([405, 1, 1]); norm_box=True
2025-06-10 11:26:09.783 | INFO     | utils.eval_helper:compute_score:324 - print_kwargs: {'dataset': '-normbox'}
2025-06-10 11:26:15.388 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:470 - Pairwise EMD CD
2025-06-10 11:26:15.388 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:477 - eval metric: CD; batch-size=202, device: cuda:0, cuda:0
2025-06-10 11:26:19.822 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042                0       40.99            0             0              0       0
2025-06-10 11:26:24.249 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042                0       40.99            0         81.23              0       0
2025-06-10 11:26:24.249 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:522 - eval metric: EMD
2025-06-10 11:27:15.067 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=1.7m
2025-06-10 11:28:31.366 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.1m
2025-06-10 11:31:04.877 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=2.6m
2025-06-10 11:32:21.764 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.3m
2025-06-10 11:34:03.770 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042           0.5718       40.99           40         81.23              0       0
2025-06-10 11:34:55.512 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=1.7m
2025-06-10 11:36:12.332 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.1m
2025-06-10 11:38:46.008 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=2.6m
2025-06-10 11:40:02.916 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.3m
2025-06-10 11:41:44.948 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042           0.5718       40.99           40         81.23          74.44       0
2025-06-10 11:41:48.848 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042           0.5718       40.99           40         81.23          74.44    0.11
"""

#compute_score(generated_samples, ref_name=reference_samples, norm_box=False)

"""
2025-06-10 11:51:31.498 | INFO     | utils.eval_helper:compute_score:253 - [data shape] ref_pcs: torch.Size([405, 2048, 3]), gen_pcs: torch.Size([410, 2048, 3]), mean=torch.Size([405, 1, 3]), std=torch.Size([405, 1, 1]); norm_box=False
2025-06-10 11:51:31.535 | INFO     | utils.eval_helper:compute_score:324 - print_kwargs: {}
2025-06-10 11:51:37.130 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:470 - Pairwise EMD CD
2025-06-10 11:51:37.131 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:477 - eval metric: CD; batch-size=202, device: cuda:0, cuda:0
2025-06-10 11:51:41.570 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
  MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
         0.4253                0       24.69            0             0              0       0
2025-06-10 11:51:45.986 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
  MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
         0.4253                0       24.69            0         94.44              0       0
2025-06-10 11:51:45.986 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:522 - eval metric: EMD
2025-06-10 11:52:36.813 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=1.7m
2025-06-10 11:53:53.176 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.1m
2025-06-10 11:56:26.888 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=2.6m
2025-06-10 11:57:43.869 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.3m
2025-06-10 11:59:25.931 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
  MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
         0.4253           0.5265       24.69        36.54         94.44              0       0
2025-06-10 12:00:17.756 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=1.7m
2025-06-10 12:01:34.681 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.1m
2025-06-10 12:04:08.564 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=2.6m
2025-06-10 12:05:25.558 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.3m
2025-06-10 12:07:07.677 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
  MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
         0.4253           0.5265       24.69        36.54         94.44          85.31       0
2025-06-10 12:07:11.626 | INFO     | utils.evaluation_metrics_fast:print_results:268 - 
  MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
         0.4253           0.5265       24.69        36.54         94.44          85.31     0.1
"""

"""
2025-06-10 17:13:53.905 | INFO     | utils.eval_helper:compute_score:229 - [compute sample metric] sample: /home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/eval_408objects/complete_point_clouds/complete_generated_shapes.pt and ref: /home/ncaytuir/data-local/LION_necs/datasets/test_data/ref_val_airplane.pt
[DEBUG] ref type: <class 'dict'>
2025-06-10 17:13:53.923 | INFO     | utils.eval_helper:compute_score:253 - [data shape] ref_pcs: torch.Size([405, 2048, 3]), gen_pcs: torch.Size([410, 2048, 3]), mean=torch.Size([405, 1, 3]), std=torch.Size([405, 1, 1]); norm_box=True
2025-06-10 17:13:54.047 | INFO     | utils.eval_helper:compute_score:324 - print_kwargs: {'dataset': '-normbox'}
2025-06-10 17:13:59.560 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:470 - Pairwise EMD CD
2025-06-10 17:13:59.560 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:477 - eval metric: CD; batch-size=202, device: cuda:0, cuda:0
2025-06-10 17:14:03.956 | INFO     | utils.evaluation_metrics_fast:print_results:268 -
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042                0       40.99            0             0              0       0
2025-06-10 17:14:08.350 | INFO     | utils.evaluation_metrics_fast:print_results:268 -
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042                0       40.99            0         81.23              0       0
2025-06-10 17:14:08.350 | INFO     | utils.evaluation_metrics_fast:compute_all_metrics:522 - eval metric: EMD
2025-06-10 17:14:59.186 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=1.7m
2025-06-10 17:16:15.488 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.1m
2025-06-10 17:18:49.008 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=2.6m
2025-06-10 17:20:05.980 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.3m
2025-06-10 17:21:48.118 | INFO     | utils.evaluation_metrics_fast:print_results:268 -
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042           0.5718       40.99           40         81.23              0       0
2025-06-10 17:22:39.966 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=1.7m
2025-06-10 17:23:56.962 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.1m
2025-06-10 17:26:30.923 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 33.3%(405) eta=2.6m
2025-06-10 17:27:47.945 | INFO     | utils.evaluation_metrics_fast:_pairwise_EMD_CD_:340 - done 66.7%(405) eta=1.3m
2025-06-10 17:29:30.126 | INFO     | utils.evaluation_metrics_fast:print_results:268 -
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042           0.5718       40.99           40         81.23          74.44       0
2025-06-10 17:29:34.024 | INFO     | utils.evaluation_metrics_fast:print_results:268 -
Dataset      MMD-CDx0.001↓    MMD-EMDx0.01↓    COV-CD%↑    COV-EMD%↑    1-NNA-CD%↓    1-NNA-EMD%↓    JSD↓
-normbox            0.3042           0.5718       40.99           40         81.23          74.44    0.11
"""