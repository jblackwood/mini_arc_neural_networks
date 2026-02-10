Configuration:
{'data_dir': PosixPath('data/MINI-ARC'),
 'output_dir': PosixPath('output/mini_arc_jepa'),
 'test_ratio': 0.2,
 'random_seed': 42,
 'max_augmentations': 500,
 'd_model': 128,
 'nhead': 8,
 'num_layers': 8,
 'dim_feedforward': 512,
 'dropout': 0.1,
 'embedding_dim': 512,
 'vocab_size': 11,
 'num_epochs': 100,
 'batch_size': 128,
 'learning_rate': 0.0002,
 'lambd': 0.05,
 'num_slices': 256,
 'mode': 'train',
 'checkpoint_save_interval': 10,
 'eval_epoch_interval': 10,
 'google_drive_dir': '/content/drive/MyDrive/sparse_arc',
 'jepa_load_model_path': None,
 'pred_load_model_path': None,
 'timestamp': '20260210_164044',
 'train_data_dir': 'output/mini_arc_jepa/train',
 'test_data_dir': 'output/mini_arc_jepa/test',
 'tensorboard_log_dir': 'output/mini_arc_jepa/tensorboard/20260210_164044',
 'checkpoint_dir': 'output/mini_arc_jepa/checkpoints',
 'model_save_dir': 'output/mini_arc_jepa/models',
 'model_save_path': 'output/mini_arc_jepa/models/20260210_164044_model.pt'}

MINI-ARC dataset already exists in 'data/MINI-ARC'. Skipping download.
Output directories already exist and contain data:
  Train directory: output/mini_arc_jepa/train (39336 files)
  Test directory: output/mini_arc_jepa/test (8384 files)
Skipping dataset creation.
Using device: cuda
Train dataset size: 39336
Test dataset size: 8384
Combined dataset size: 47720
Compiling models with torch.compile...
JEPA model has 1,654,272 trainable parameters
Prediction model has 2,126,347 trainable parameters
Total parameters: 3,780,619

Starting training...
Epoch 1/100 - Train JEPA Loss: 0.502885, Train Pred Loss: 1.110385, Time: 292.98s
  JEPA Loss Components - Train Sim: 0.068472, Train SigReg: 8.756742
  Timing Breakdown - Data: -19.96s (-6.8%), JEPA Compute: 95.46s (32.6%), JEPA Backward: 164.63s (56.2%), Pred Train: 25.03s (8.5%)
  Model Norms - JEPA: 88.6067, Pred: 130.7492, JEPA Out Scale: 13.6783, Pred Out Scale: 2.1716
Epoch 2/100 - Train JEPA Loss: 0.378519, Train Pred Loss: 0.845809, Time: 302.90s
  JEPA Loss Components - Train Sim: 0.069279, Train SigReg: 6.254077
  Timing Breakdown - Data: -0.25s (-0.1%), JEPA Compute: 91.85s (30.3%), JEPA Backward: 180.04s (59.4%), Pred Train: 22.46s (7.4%)
  Model Norms - JEPA: 88.7466, Pred: 131.3875, JEPA Out Scale: 13.8274, Pred Out Scale: 2.3837
Epoch 3/100 - Train JEPA Loss: 0.350510, Train Pred Loss: 0.708257, Time: 301.78s
  JEPA Loss Components - Train Sim: 0.072226, Train SigReg: 5.637896
  Timing Breakdown - Data: -0.07s (-0.0%), JEPA Compute: 91.58s (30.3%), JEPA Backward: 179.83s (59.6%), Pred Train: 22.52s (7.5%)
  Model Norms - JEPA: 88.9296, Pred: 131.7881, JEPA Out Scale: 14.0370, Pred Out Scale: 2.5098
Epoch 4/100 - Train JEPA Loss: 0.321379, Train Pred Loss: 0.646869, Time: 302.65s
  JEPA Loss Components - Train Sim: 0.076019, Train SigReg: 4.983210
  Timing Breakdown - Data: 0.07s (0.0%), JEPA Compute: 91.68s (30.3%), JEPA Backward: 179.87s (59.4%), Pred Train: 22.47s (7.4%)
  Model Norms - JEPA: 89.1073, Pred: 132.2001, JEPA Out Scale: 14.2503, Pred Out Scale: 2.6220
Epoch 5/100 - Train JEPA Loss: 0.303364, Train Pred Loss: 0.600297, Time: 301.97s
  JEPA Loss Components - Train Sim: 0.077877, Train SigReg: 4.587616
  Timing Breakdown - Data: -0.12s (-0.0%), JEPA Compute: 91.85s (30.4%), JEPA Backward: 179.80s (59.5%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 89.2691, Pred: 132.6158, JEPA Out Scale: 14.4356, Pred Out Scale: 2.7172
Epoch 6/100 - Train JEPA Loss: 0.290809, Train Pred Loss: 0.558052, Time: 302.03s
  JEPA Loss Components - Train Sim: 0.078340, Train SigReg: 4.327725
  Timing Breakdown - Data: -0.07s (-0.0%), JEPA Compute: 91.87s (30.4%), JEPA Backward: 179.79s (59.5%), Pred Train: 22.48s (7.4%)
  Model Norms - JEPA: 89.4203, Pred: 133.0504, JEPA Out Scale: 14.5685, Pred Out Scale: 2.8016
Epoch 7/100 - Train JEPA Loss: 0.278446, Train Pred Loss: 0.517142, Time: 302.63s
  JEPA Loss Components - Train Sim: 0.078852, Train SigReg: 4.070730
  Timing Breakdown - Data: -0.08s (-0.0%), JEPA Compute: 91.81s (30.3%), JEPA Backward: 179.78s (59.4%), Pred Train: 22.46s (7.4%)
  Model Norms - JEPA: 89.5709, Pred: 133.4853, JEPA Out Scale: 14.6864, Pred Out Scale: 2.8771
Epoch 8/100 - Train JEPA Loss: 0.271272, Train Pred Loss: 0.478744, Time: 301.88s
  JEPA Loss Components - Train Sim: 0.079068, Train SigReg: 3.923146
  Timing Breakdown - Data: -0.17s (-0.1%), JEPA Compute: 91.81s (30.4%), JEPA Backward: 179.78s (59.6%), Pred Train: 22.46s (7.4%)
  Model Norms - JEPA: 89.7254, Pred: 133.9081, JEPA Out Scale: 14.8058, Pred Out Scale: 2.9446
Epoch 9/100 - Train JEPA Loss: 0.267208, Train Pred Loss: 0.442634, Time: 302.43s
  JEPA Loss Components - Train Sim: 0.078126, Train SigReg: 3.859767
  Timing Breakdown - Data: -0.12s (-0.0%), JEPA Compute: 91.82s (30.4%), JEPA Backward: 179.64s (59.4%), Pred Train: 22.43s (7.4%)
  Model Norms - JEPA: 89.8751, Pred: 134.3230, JEPA Out Scale: 14.8981, Pred Out Scale: 3.0051
Epoch 10/100 - Train JEPA Loss: 0.260319, Train Pred Loss: 0.406913, Eval Pred Loss: 0.440668, Time: 473.36s
  JEPA Loss Components - Train Sim: 0.076469, Train SigReg: 3.753480
  Eval Accuracy (Train Examples) - Accuracy: 68.69%, Perfect: 9.84%
  Eval Accuracy (Test Examples) - Accuracy: 67.30%, Perfect: 5.94%
  Timing Breakdown - Data: -2.13s (-0.4%), JEPA Compute: 91.07s (19.2%), JEPA Backward: 179.65s (38.0%), Pred Train: 22.45s (4.7%), Pred Eval: 172.17s (36.4%)
  Model Norms - JEPA: 90.0475, Pred: 134.7230, JEPA Out Scale: 15.0159, Pred Out Scale: 3.0639
Saved JEPA checkpoint to output/mini_arc_jepa/checkpoints/jepa_20260210_164044_epoch_10_checkpoint.pt
Saved prediction checkpoint to output/mini_arc_jepa/checkpoints/pred_20260210_164044_epoch_10_checkpoint.pt
Copied JEPA checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/jepa_20260210_164044_epoch_10_checkpoint.pt
Copied prediction checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/pred_20260210_164044_epoch_10_checkpoint.pt
Epoch 11/100 - Train JEPA Loss: 0.255041, Train Pred Loss: 0.373519, Time: 302.67s
  JEPA Loss Components - Train Sim: 0.076114, Train SigReg: 3.654662
  Timing Breakdown - Data: -0.18s (-0.1%), JEPA Compute: 91.83s (30.3%), JEPA Backward: 179.68s (59.4%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 90.2244, Pred: 135.1155, JEPA Out Scale: 15.1330, Pred Out Scale: 3.1190
Epoch 12/100 - Train JEPA Loss: 0.251237, Train Pred Loss: 0.341484, Time: 301.79s
  JEPA Loss Components - Train Sim: 0.074609, Train SigReg: 3.607155
  Timing Breakdown - Data: -0.15s (-0.1%), JEPA Compute: 91.76s (30.4%), JEPA Backward: 179.59s (59.5%), Pred Train: 22.48s (7.4%)
  Model Norms - JEPA: 90.4114, Pred: 135.4852, JEPA Out Scale: 15.2268, Pred Out Scale: 3.1712
Epoch 13/100 - Train JEPA Loss: 0.245180, Train Pred Loss: 0.311034, Time: 302.05s
  JEPA Loss Components - Train Sim: 0.071956, Train SigReg: 3.536432
  Timing Breakdown - Data: -0.11s (-0.0%), JEPA Compute: 91.92s (30.4%), JEPA Backward: 179.56s (59.4%), Pred Train: 22.49s (7.4%)
  Model Norms - JEPA: 90.6155, Pred: 135.8420, JEPA Out Scale: 15.3222, Pred Out Scale: 3.2208
Epoch 14/100 - Train JEPA Loss: 0.238551, Train Pred Loss: 0.283087, Time: 302.31s
  JEPA Loss Components - Train Sim: 0.068898, Train SigReg: 3.461950
  Timing Breakdown - Data: -0.17s (-0.1%), JEPA Compute: 91.70s (30.3%), JEPA Backward: 179.48s (59.4%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 90.8344, Pred: 136.1833, JEPA Out Scale: 15.4063, Pred Out Scale: 3.2715
Epoch 15/100 - Train JEPA Loss: 0.232767, Train Pred Loss: 0.258146, Time: 301.38s
  JEPA Loss Components - Train Sim: 0.064180, Train SigReg: 3.435924
  Timing Breakdown - Data: -0.12s (-0.0%), JEPA Compute: 91.52s (30.4%), JEPA Backward: 179.42s (59.5%), Pred Train: 22.47s (7.5%)
  Model Norms - JEPA: 91.0814, Pred: 136.5079, JEPA Out Scale: 15.5156, Pred Out Scale: 3.3225
Epoch 16/100 - Train JEPA Loss: 0.221088, Train Pred Loss: 0.235760, Time: 302.27s
  JEPA Loss Components - Train Sim: 0.058826, Train SigReg: 3.304071
  Timing Breakdown - Data: -0.14s (-0.0%), JEPA Compute: 91.63s (30.3%), JEPA Backward: 179.43s (59.4%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 91.3283, Pred: 136.8116, JEPA Out Scale: 15.6316, Pred Out Scale: 3.3662
Epoch 17/100 - Train JEPA Loss: 0.214541, Train Pred Loss: 0.217267, Time: 301.39s
  JEPA Loss Components - Train Sim: 0.054267, Train SigReg: 3.259735
  Timing Breakdown - Data: -0.15s (-0.1%), JEPA Compute: 91.58s (30.4%), JEPA Backward: 179.34s (59.5%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 91.5681, Pred: 137.1047, JEPA Out Scale: 15.7504, Pred Out Scale: 3.4093
Epoch 18/100 - Train JEPA Loss: 0.208131, Train Pred Loss: 0.199799, Time: 301.26s
  JEPA Loss Components - Train Sim: 0.049782, Train SigReg: 3.216749
  Timing Breakdown - Data: -0.20s (-0.1%), JEPA Compute: 91.58s (30.4%), JEPA Backward: 179.31s (59.5%), Pred Train: 22.44s (7.4%)
  Model Norms - JEPA: 91.7922, Pred: 137.3795, JEPA Out Scale: 15.8514, Pred Out Scale: 3.4495
Epoch 19/100 - Train JEPA Loss: 0.200671, Train Pred Loss: 0.182839, Time: 301.90s
  JEPA Loss Components - Train Sim: 0.046382, Train SigReg: 3.132158
  Timing Breakdown - Data: -0.06s (-0.0%), JEPA Compute: 91.53s (30.3%), JEPA Backward: 179.24s (59.4%), Pred Train: 22.43s (7.4%)
  Model Norms - JEPA: 92.0057, Pred: 137.6381, JEPA Out Scale: 15.9237, Pred Out Scale: 3.4917
Epoch 20/100 - Train JEPA Loss: 0.194141, Train Pred Loss: 0.168289, Eval Pred Loss: 0.383913, Time: 472.23s
  JEPA Loss Components - Train Sim: 0.042585, Train SigReg: 3.073711
  Eval Accuracy (Train Examples) - Accuracy: 79.53%, Perfect: 43.52%
  Eval Accuracy (Test Examples) - Accuracy: 70.36%, Perfect: 14.92%
  Timing Breakdown - Data: 0.29s (0.1%), JEPA Compute: 91.10s (19.3%), JEPA Backward: 179.31s (38.0%), Pred Train: 22.46s (4.8%), Pred Eval: 171.29s (36.3%)
  Model Norms - JEPA: 92.2137, Pred: 137.8874, JEPA Out Scale: 16.0121, Pred Out Scale: 3.5320
Saved JEPA checkpoint to output/mini_arc_jepa/checkpoints/jepa_20260210_164044_epoch_20_checkpoint.pt
Saved prediction checkpoint to output/mini_arc_jepa/checkpoints/pred_20260210_164044_epoch_20_checkpoint.pt
Copied JEPA checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/jepa_20260210_164044_epoch_20_checkpoint.pt
Copied prediction checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/pred_20260210_164044_epoch_20_checkpoint.pt
Epoch 21/100 - Train JEPA Loss: 0.189717, Train Pred Loss: 0.157185, Time: 302.11s
  JEPA Loss Components - Train Sim: 0.039818, Train SigReg: 3.037803
  Timing Breakdown - Data: -0.02s (-0.0%), JEPA Compute: 91.59s (30.3%), JEPA Backward: 179.27s (59.3%), Pred Train: 22.47s (7.4%)
  Model Norms - JEPA: 92.4229, Pred: 138.1323, JEPA Out Scale: 16.0709, Pred Out Scale: 3.5699
Epoch 22/100 - Train JEPA Loss: 0.184314, Train Pred Loss: 0.145876, Time: 301.50s
  JEPA Loss Components - Train Sim: 0.037858, Train SigReg: 2.966981
  Timing Breakdown - Data: -0.02s (-0.0%), JEPA Compute: 91.76s (30.4%), JEPA Backward: 179.34s (59.5%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 92.6184, Pred: 138.3635, JEPA Out Scale: 16.1451, Pred Out Scale: 3.6098
Epoch 23/100 - Train JEPA Loss: 0.181535, Train Pred Loss: 0.136774, Time: 302.08s
  JEPA Loss Components - Train Sim: 0.035920, Train SigReg: 2.948231
  Timing Breakdown - Data: -0.15s (-0.0%), JEPA Compute: 91.74s (30.4%), JEPA Backward: 179.27s (59.3%), Pred Train: 22.43s (7.4%)
  Model Norms - JEPA: 92.8113, Pred: 138.5951, JEPA Out Scale: 16.2142, Pred Out Scale: 3.6470
Epoch 24/100 - Train JEPA Loss: 0.177049, Train Pred Loss: 0.128548, Time: 301.45s
  JEPA Loss Components - Train Sim: 0.034441, Train SigReg: 2.886600
  Timing Breakdown - Data: -0.01s (-0.0%), JEPA Compute: 91.82s (30.5%), JEPA Backward: 179.25s (59.5%), Pred Train: 22.43s (7.4%)
  Model Norms - JEPA: 93.0015, Pred: 138.8169, JEPA Out Scale: 16.2500, Pred Out Scale: 3.6828
Epoch 25/100 - Train JEPA Loss: 0.175007, Train Pred Loss: 0.120791, Time: 301.50s
  JEPA Loss Components - Train Sim: 0.032806, Train SigReg: 2.876813
  Timing Breakdown - Data: -0.13s (-0.0%), JEPA Compute: 91.80s (30.4%), JEPA Backward: 179.26s (59.5%), Pred Train: 22.46s (7.4%)
  Model Norms - JEPA: 93.1927, Pred: 139.0278, JEPA Out Scale: 16.3137, Pred Out Scale: 3.7198
Epoch 26/100 - Train JEPA Loss: 0.171723, Train Pred Loss: 0.113916, Time: 302.20s
  JEPA Loss Components - Train Sim: 0.031677, Train SigReg: 2.832609
  Timing Breakdown - Data: -0.03s (-0.0%), JEPA Compute: 91.78s (30.4%), JEPA Backward: 179.27s (59.3%), Pred Train: 22.45s (7.4%)
  Model Norms - JEPA: 93.3839, Pred: 139.2394, JEPA Out Scale: 16.3594, Pred Out Scale: 3.7532
Epoch 27/100 - Train JEPA Loss: 0.169545, Train Pred Loss: 0.107912, Time: 301.24s
  JEPA Loss Components - Train Sim: 0.031092, Train SigReg: 2.800146
  Timing Breakdown - Data: -0.02s (-0.0%), JEPA Compute: 91.72s (30.4%), JEPA Backward: 179.24s (59.5%), Pred Train: 22.42s (7.4%)
  Model Norms - JEPA: 93.5839, Pred: 139.4454, JEPA Out Scale: 16.4308, Pred Out Scale: 3.7878
Epoch 28/100 - Train JEPA Loss: 0.167632, Train Pred Loss: 0.103231, Time: 301.84s
  JEPA Loss Components - Train Sim: 0.029692, Train SigReg: 2.788491
  Timing Breakdown - Data: -0.04s (-0.0%), JEPA Compute: 91.69s (30.4%), JEPA Backward: 179.24s (59.4%), Pred Train: 22.40s (7.4%)
  Model Norms - JEPA: 93.7806, Pred: 139.6471, JEPA Out Scale: 16.4899, Pred Out Scale: 3.8206
Epoch 29/100 - Train JEPA Loss: 0.166183, Train Pred Loss: 0.097933, Time: 301.09s
  JEPA Loss Components - Train Sim: 0.029078, Train SigReg: 2.771175
  Timing Breakdown - Data: -0.07s (-0.0%), JEPA Compute: 91.69s (30.5%), JEPA Backward: 179.19s (59.5%), Pred Train: 22.42s (7.4%)
  Model Norms - JEPA: 93.9841, Pred: 139.8390, JEPA Out Scale: 16.5214, Pred Out Scale: 3.8522
Epoch 30/100 - Train JEPA Loss: 0.162702, Train Pred Loss: 0.094059, Eval Pred Loss: 0.424690, Time: 467.22s
  JEPA Loss Components - Train Sim: 0.028095, Train SigReg: 2.720235
  Eval Accuracy (Train Examples) - Accuracy: 88.11%, Perfect: 67.40%
  Eval Accuracy (Test Examples) - Accuracy: 71.87%, Perfect: 18.11%
  Timing Breakdown - Data: 0.32s (0.1%), JEPA Compute: 90.88s (19.5%), JEPA Backward: 179.06s (38.3%), Pred Train: 22.38s (4.8%), Pred Eval: 167.00s (35.7%)
  Model Norms - JEPA: 94.1844, Pred: 140.0361, JEPA Out Scale: 16.5826, Pred Out Scale: 3.8811
Saved JEPA checkpoint to output/mini_arc_jepa/checkpoints/jepa_20260210_164044_epoch_30_checkpoint.pt
Saved prediction checkpoint to output/mini_arc_jepa/checkpoints/pred_20260210_164044_epoch_30_checkpoint.pt
Copied JEPA checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/jepa_20260210_164044_epoch_30_checkpoint.pt
Copied prediction checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/pred_20260210_164044_epoch_30_checkpoint.pt
Epoch 31/100 - Train JEPA Loss: 0.160390, Train Pred Loss: 0.089302, Time: 301.45s
  JEPA Loss Components - Train Sim: 0.027184, Train SigReg: 2.691314
  Timing Breakdown - Data: -0.27s (-0.1%), JEPA Compute: 91.50s (30.4%), JEPA Backward: 179.10s (59.4%), Pred Train: 22.38s (7.4%)
  Model Norms - JEPA: 94.3762, Pred: 140.2233, JEPA Out Scale: 16.6385, Pred Out Scale: 3.9142
Epoch 32/100 - Train JEPA Loss: 0.157612, Train Pred Loss: 0.085447, Time: 300.80s
  JEPA Loss Components - Train Sim: 0.026278, Train SigReg: 2.652975
  Timing Breakdown - Data: -0.05s (-0.0%), JEPA Compute: 91.52s (30.4%), JEPA Backward: 178.99s (59.5%), Pred Train: 22.42s (7.5%)
  Model Norms - JEPA: 94.5844, Pred: 140.4073, JEPA Out Scale: 16.7108, Pred Out Scale: 3.9453
Epoch 33/100 - Train JEPA Loss: 0.155780, Train Pred Loss: 0.081813, Time: 300.65s
  JEPA Loss Components - Train Sim: 0.025646, Train SigReg: 2.628331
  Timing Breakdown - Data: -0.09s (-0.0%), JEPA Compute: 91.50s (30.4%), JEPA Backward: 178.96s (59.5%), Pred Train: 22.39s (7.4%)
  Model Norms - JEPA: 94.7875, Pred: 140.5836, JEPA Out Scale: 16.7637, Pred Out Scale: 3.9754
Epoch 34/100 - Train JEPA Loss: 0.152789, Train Pred Loss: 0.079039, Time: 301.68s
  JEPA Loss Components - Train Sim: 0.024343, Train SigReg: 2.593253
  Timing Breakdown - Data: 0.06s (0.0%), JEPA Compute: 91.56s (30.3%), JEPA Backward: 178.99s (59.3%), Pred Train: 22.40s (7.4%)
  Model Norms - JEPA: 94.9876, Pred: 140.7565, JEPA Out Scale: 16.8095, Pred Out Scale: 4.0063
Epoch 35/100 - Train JEPA Loss: 0.149624, Train Pred Loss: 0.075782, Time: 300.76s
  JEPA Loss Components - Train Sim: 0.023714, Train SigReg: 2.541916
  Timing Breakdown - Data: -0.03s (-0.0%), JEPA Compute: 91.43s (30.4%), JEPA Backward: 179.01s (59.5%), Pred Train: 22.43s (7.5%)
  Model Norms - JEPA: 95.1867, Pred: 140.9288, JEPA Out Scale: 16.8539, Pred Out Scale: 4.0368
Epoch 36/100 - Train JEPA Loss: 0.149095, Train Pred Loss: 0.074133, Time: 301.27s
  JEPA Loss Components - Train Sim: 0.022958, Train SigReg: 2.545694
  Timing Breakdown - Data: -0.09s (-0.0%), JEPA Compute: 91.38s (30.3%), JEPA Backward: 178.98s (59.4%), Pred Train: 22.39s (7.4%)
  Model Norms - JEPA: 95.3986, Pred: 141.0931, JEPA Out Scale: 16.8987, Pred Out Scale: 4.0636
Epoch 37/100 - Train JEPA Loss: 0.146259, Train Pred Loss: 0.071147, Time: 300.63s
  JEPA Loss Components - Train Sim: 0.021947, Train SigReg: 2.508193
  Timing Breakdown - Data: -0.09s (-0.0%), JEPA Compute: 91.43s (30.4%), JEPA Backward: 178.97s (59.5%), Pred Train: 22.41s (7.5%)
  Model Norms - JEPA: 95.5864, Pred: 141.2605, JEPA Out Scale: 16.9377, Pred Out Scale: 4.0945
Epoch 38/100 - Train JEPA Loss: 0.144678, Train Pred Loss: 0.068865, Time: 300.65s
  JEPA Loss Components - Train Sim: 0.021166, Train SigReg: 2.491393
  Timing Breakdown - Data: -0.17s (-0.1%), JEPA Compute: 91.45s (30.4%), JEPA Backward: 178.98s (59.5%), Pred Train: 22.39s (7.4%)
  Model Norms - JEPA: 95.7804, Pred: 141.4263, JEPA Out Scale: 16.9650, Pred Out Scale: 4.1262
Epoch 39/100 - Train JEPA Loss: 0.141102, Train Pred Loss: 0.066058, Time: 301.66s
  JEPA Loss Components - Train Sim: 0.020066, Train SigReg: 2.440788
  Timing Breakdown - Data: -0.03s (-0.0%), JEPA Compute: 91.68s (30.4%), JEPA Backward: 179.09s (59.4%), Pred Train: 22.38s (7.4%)
  Model Norms - JEPA: 95.9755, Pred: 141.5813, JEPA Out Scale: 17.0121, Pred Out Scale: 4.1564
Epoch 40/100 - Train JEPA Loss: 0.139645, Train Pred Loss: 0.064223, Eval Pred Loss: 0.458002, Time: 467.61s
  JEPA Loss Components - Train Sim: 0.019334, Train SigReg: 2.425561
  Eval Accuracy (Train Examples) - Accuracy: 91.93%, Perfect: 77.13%
  Eval Accuracy (Test Examples) - Accuracy: 72.82%, Perfect: 19.65%
  Timing Breakdown - Data: -0.56s (-0.1%), JEPA Compute: 90.86s (19.4%), JEPA Backward: 178.96s (38.3%), Pred Train: 22.40s (4.8%), Pred Eval: 167.46s (35.8%)
  Model Norms - JEPA: 96.1621, Pred: 141.7387, JEPA Out Scale: 17.0519, Pred Out Scale: 4.1860
Saved JEPA checkpoint to output/mini_arc_jepa/checkpoints/jepa_20260210_164044_epoch_40_checkpoint.pt
Saved prediction checkpoint to output/mini_arc_jepa/checkpoints/pred_20260210_164044_epoch_40_checkpoint.pt
Copied JEPA checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/jepa_20260210_164044_epoch_40_checkpoint.pt
Copied prediction checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/pred_20260210_164044_epoch_40_checkpoint.pt
Epoch 41/100 - Train JEPA Loss: 0.138859, Train Pred Loss: 0.062364, Time: 301.62s
  JEPA Loss Components - Train Sim: 0.018724, Train SigReg: 2.421415
  Timing Breakdown - Data: -0.20s (-0.1%), JEPA Compute: 91.61s (30.4%), JEPA Backward: 179.10s (59.4%), Pred Train: 22.42s (7.4%)
  Model Norms - JEPA: 96.3469, Pred: 141.8927, JEPA Out Scale: 17.0791, Pred Out Scale: 4.2182
Epoch 42/100 - Train JEPA Loss: 0.136086, Train Pred Loss: 0.060636, Time: 301.27s
  JEPA Loss Components - Train Sim: 0.017736, Train SigReg: 2.384734
  Timing Breakdown - Data: -0.08s (-0.0%), JEPA Compute: 91.74s (30.4%), JEPA Backward: 179.11s (59.5%), Pred Train: 22.45s (7.5%)
  Model Norms - JEPA: 96.5431, Pred: 142.0450, JEPA Out Scale: 17.1337, Pred Out Scale: 4.2465
Epoch 43/100 - Train JEPA Loss: 0.134209, Train Pred Loss: 0.058903, Time: 301.74s
  JEPA Loss Components - Train Sim: 0.016819, Train SigReg: 2.364622
  Timing Breakdown - Data: 0.02s (0.0%), JEPA Compute: 91.72s (30.4%), JEPA Backward: 179.03s (59.3%), Pred Train: 22.39s (7.4%)
  Model Norms - JEPA: 96.7372, Pred: 142.1930, JEPA Out Scale: 17.1836, Pred Out Scale: 4.2770
Epoch 44/100 - Train JEPA Loss: 0.133412, Train Pred Loss: 0.057264, Time: 301.07s
  JEPA Loss Components - Train Sim: 0.016235, Train SigReg: 2.359781
  Timing Breakdown - Data: -0.11s (-0.0%), JEPA Compute: 91.74s (30.5%), JEPA Backward: 179.04s (59.5%), Pred Train: 22.41s (7.4%)
  Model Norms - JEPA: 96.9126, Pred: 142.3423, JEPA Out Scale: 17.2184, Pred Out Scale: 4.3048
Epoch 45/100 - Train JEPA Loss: 0.130728, Train Pred Loss: 0.056247, Time: 301.03s
  JEPA Loss Components - Train Sim: 0.015772, Train SigReg: 2.314892
  Timing Breakdown - Data: -0.17s (-0.1%), JEPA Compute: 91.76s (30.5%), JEPA Backward: 179.00s (59.5%), Pred Train: 22.39s (7.4%)
  Model Norms - JEPA: 97.1021, Pred: 142.4868, JEPA Out Scale: 17.2632, Pred Out Scale: 4.3321
Epoch 46/100 - Train JEPA Loss: 0.130219, Train Pred Loss: 0.054351, Time: 301.68s
  JEPA Loss Components - Train Sim: 0.014852, Train SigReg: 2.322183
  Timing Breakdown - Data: -0.06s (-0.0%), JEPA Compute: 91.70s (30.4%), JEPA Backward: 179.04s (59.3%), Pred Train: 22.41s (7.4%)
  Model Norms - JEPA: 97.2853, Pred: 142.6351, JEPA Out Scale: 17.2952, Pred Out Scale: 4.3630
Epoch 47/100 - Train JEPA Loss: 0.127482, Train Pred Loss: 0.053087, Time: 300.84s
  JEPA Loss Components - Train Sim: 0.014100, Train SigReg: 2.281734
  Timing Breakdown - Data: -0.11s (-0.0%), JEPA Compute: 91.67s (30.5%), JEPA Backward: 179.06s (59.5%), Pred Train: 22.36s (7.4%)
  Model Norms - JEPA: 97.4581, Pred: 142.7811, JEPA Out Scale: 17.3319, Pred Out Scale: 4.3895
Epoch 48/100 - Train JEPA Loss: 0.127374, Train Pred Loss: 0.051496, Time: 301.51s
  JEPA Loss Components - Train Sim: 0.013321, Train SigReg: 2.294390
  Timing Breakdown - Data: -0.09s (-0.0%), JEPA Compute: 91.68s (30.4%), JEPA Backward: 179.00s (59.4%), Pred Train: 22.38s (7.4%)
  Model Norms - JEPA: 97.6282, Pred: 142.9134, JEPA Out Scale: 17.3375, Pred Out Scale: 4.4203
Epoch 49/100 - Train JEPA Loss: 0.125250, Train Pred Loss: 0.050294, Time: 301.04s
  JEPA Loss Components - Train Sim: 0.012692, Train SigReg: 2.263855
  Timing Breakdown - Data: -0.07s (-0.0%), JEPA Compute: 91.75s (30.5%), JEPA Backward: 178.98s (59.5%), Pred Train: 22.42s (7.4%)
  Model Norms - JEPA: 97.8071, Pred: 143.0518, JEPA Out Scale: 17.3866, Pred Out Scale: 4.4506
Epoch 50/100 - Train JEPA Loss: 0.124526, Train Pred Loss: 0.048743, Eval Pred Loss: 0.490963, Time: 467.19s
  JEPA Loss Components - Train Sim: 0.012194, Train SigReg: 2.258831
  Eval Accuracy (Train Examples) - Accuracy: 93.76%, Perfect: 81.84%
  Eval Accuracy (Test Examples) - Accuracy: 73.50%, Perfect: 20.73%
  Timing Breakdown - Data: 0.14s (0.0%), JEPA Compute: 90.87s (19.5%), JEPA Backward: 178.87s (38.3%), Pred Train: 22.39s (4.8%), Pred Eval: 167.11s (35.8%)
  Model Norms - JEPA: 97.9735, Pred: 143.1873, JEPA Out Scale: 17.3898, Pred Out Scale: 4.4791
Saved JEPA checkpoint to output/mini_arc_jepa/checkpoints/jepa_20260210_164044_epoch_50_checkpoint.pt
Saved prediction checkpoint to output/mini_arc_jepa/checkpoints/pred_20260210_164044_epoch_50_checkpoint.pt
Copied JEPA checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/jepa_20260210_164044_epoch_50_checkpoint.pt
Copied prediction checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/pred_20260210_164044_epoch_50_checkpoint.pt