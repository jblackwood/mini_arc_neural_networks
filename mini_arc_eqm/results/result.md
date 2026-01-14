Configuration:
{'data_dir': PosixPath('data/MINI-ARC'),
 'output_dir': PosixPath('output/mini_arc_eqm5'),
 'test_ratio': 0.2,
 'random_seed': 42,
 'max_augmentations': 500,
 'd_model': 256,
 'nhead': 8,
 'num_layers': 8,
 'dim_feedforward': 1024,
 'dropout': 0.1,
 'batch_size': 32,
 'num_epochs': 300,
 'learning_rate': 0.0001,
 'weight_decay': 0.1,
 'mode': 'train',
 'checkpoint_save_interval': 30,
 'vocab_size': 11,
 'eval_denoise_epoch_interval': 1,
 'eval_denoise_num_iterations': 500,
 'google_drive_dir': '/content/drive/MyDrive/sparse_arc',
 'load_model_path': '/content/20260112_213143_epoch_150_checkpoint.pt',
 'timestamp': '20260113_215545',
 'tensorboard_log_dir': 'output/mini_arc_eqm5/runs/20260113_215545_model',
 'model_save_dir': 'output/mini_arc_eqm5/models',
 'model_save_path': 'output/mini_arc_eqm5/models/20260113_215545_model.pt',
 'checkpoint_dir': 'output/mini_arc_eqm5/checkpoints',
 'train_data_dir': 'output/mini_arc_eqm5/train',
 'test_data_dir': 'output/mini_arc_eqm5/test'}

Downloading MINI-ARC from GitHub...
Downloaded to: data/mini-arc-master.zip
MINI-ARC dataset extracted to 'data/MINI-ARC' directory.
Found 149 task files
Train tasks: 120
Test tasks: 29

Processing train files...
Wrote 39336 task JSON files to output/mini_arc_eqm5/train

Processing test files...
Wrote 8384 task JSON files to output/mini_arc_eqm5/test

Datasets created successfully!
Train directory: output/mini_arc_eqm5/train
Test directory: output/mini_arc_eqm5/test
Using device: cuda
/usr/local/lib/python3.12/dist-packages/torch/__init__.py:1617: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  _C._set_float32_matmul_precision(precision)
Train dataset size: 47720
Test dataset size: 8384

Building task_id to task_index mapping...
Found 47720 unique tasks
Sample task_ids: ['miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipa', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipa_0to4_4to0', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipd', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipv', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipv_0to4_4to0']
Compiling model with torch.compile...
Model has 18,545,033 trainable parameters
  Embedding parameters: 12,216,320 (65.9%)
  Other parameters: 6,328,713 (34.1%)

Loading existing model from /content/20260112_213143_epoch_150_checkpoint.pt
Resumed from epoch 150
Previous train loss: 0.005823627243624643
Previous test loss: 1.8287644705064745

Starting training...
W0113 21:56:44.692000 3997 torch/_inductor/utils.py:1558] [0/0] Not enough SMs to use max_autotune_gemm mode
Epoch 151/450 - Train Loss: 0.005766, Test Loss: 1.966910, Time: 171.86s, Weight Norm: 378.6285, Logit Scale: 8.1446

Evaluating denoising accuracy at epoch 151...
  Train Accuracy: 99.73% (100% acc: 97.5%), Test Accuracy: 70.90% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.3±0.5 (max: 3), Time: 7.47s

Epoch 152/450 - Train Loss: 0.005713, Test Loss: 1.870527, Time: 92.83s, Weight Norm: 373.4123, Logit Scale: 8.1496

Evaluating denoising accuracy at epoch 152...
  Train Accuracy: 99.83% (100% acc: 95.8%), Test Accuracy: 72.28% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.7 (max: 4), Time: 4.73s

Epoch 153/450 - Train Loss: 0.005132, Test Loss: 1.960059, Time: 93.21s, Weight Norm: 368.2848, Logit Scale: 8.1617

Evaluating denoising accuracy at epoch 153...
  Train Accuracy: 99.50% (100% acc: 94.2%), Test Accuracy: 70.62% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.4 (max: 2), Time: 4.74s

Epoch 154/450 - Train Loss: 0.005965, Test Loss: 1.981012, Time: 93.27s, Weight Norm: 363.3002, Logit Scale: 8.1683

Evaluating denoising accuracy at epoch 154...
  Train Accuracy: 99.77% (100% acc: 94.2%), Test Accuracy: 71.86% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.6 (max: 4), Time: 4.74s

Epoch 155/450 - Train Loss: 0.005421, Test Loss: 1.965413, Time: 93.43s, Weight Norm: 358.3608, Logit Scale: 8.1794

Evaluating denoising accuracy at epoch 155...
  Train Accuracy: 99.37% (100% acc: 95.0%), Test Accuracy: 71.45% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.5±0.6 (max: 3), Time: 4.74s

Epoch 156/450 - Train Loss: 0.005614, Test Loss: 1.872688, Time: 93.21s, Weight Norm: 353.5394, Logit Scale: 8.1883

Evaluating denoising accuracy at epoch 156...
  Train Accuracy: 98.43% (100% acc: 88.3%), Test Accuracy: 71.03% (100% acc: 10.3%)
  Train Best Iter: 1.1±0.3 (max: 4), Test Best Iter: 1.5±0.6 (max: 3), Time: 4.74s

Epoch 157/450 - Train Loss: 0.005318, Test Loss: 2.002251, Time: 93.37s, Weight Norm: 348.7056, Logit Scale: 8.1883

Evaluating denoising accuracy at epoch 157...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 71.31% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 2), Test Best Iter: 1.5±0.7 (max: 4), Time: 4.73s

Epoch 158/450 - Train Loss: 0.004838, Test Loss: 2.002342, Time: 93.15s, Weight Norm: 344.0157, Logit Scale: 8.2110

Evaluating denoising accuracy at epoch 158...
  Train Accuracy: 99.67% (100% acc: 95.8%), Test Accuracy: 72.41% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.2±0.5 (max: 3), Time: 4.72s

Epoch 159/450 - Train Loss: 0.004885, Test Loss: 1.975297, Time: 93.16s, Weight Norm: 339.3927, Logit Scale: 8.2360

Evaluating denoising accuracy at epoch 159...
  Train Accuracy: 99.33% (100% acc: 93.3%), Test Accuracy: 72.00% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.4±1.1 (max: 7), Time: 4.71s

Epoch 160/450 - Train Loss: 0.005786, Test Loss: 2.040998, Time: 93.18s, Weight Norm: 334.8880, Logit Scale: 8.2431

Evaluating denoising accuracy at epoch 160...
  Train Accuracy: 99.40% (100% acc: 95.0%), Test Accuracy: 71.03% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.72s

Epoch 161/450 - Train Loss: 0.004283, Test Loss: 1.904735, Time: 93.51s, Weight Norm: 330.3762, Logit Scale: 8.2616

Evaluating denoising accuracy at epoch 161...
  Train Accuracy: 98.97% (100% acc: 96.7%), Test Accuracy: 70.76% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 2), Test Best Iter: 1.6±1.2 (max: 7), Time: 4.73s

Epoch 162/450 - Train Loss: 0.005795, Test Loss: 1.979195, Time: 93.26s, Weight Norm: 326.0529, Logit Scale: 8.2718

Evaluating denoising accuracy at epoch 162...
  Train Accuracy: 99.70% (100% acc: 94.2%), Test Accuracy: 72.00% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 2), Time: 4.72s

Epoch 163/450 - Train Loss: 0.004498, Test Loss: 1.887908, Time: 93.19s, Weight Norm: 321.6688, Logit Scale: 8.2831

Evaluating denoising accuracy at epoch 163...
  Train Accuracy: 99.77% (100% acc: 95.0%), Test Accuracy: 71.72% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.2 (max: 2), Test Best Iter: 1.2±0.4 (max: 2), Time: 4.73s

Epoch 164/450 - Train Loss: 0.005250, Test Loss: 1.794183, Time: 93.02s, Weight Norm: 317.5435, Logit Scale: 8.3014

Evaluating denoising accuracy at epoch 164...
  Train Accuracy: 99.37% (100% acc: 93.3%), Test Accuracy: 74.90% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.6 (max: 3), Time: 4.74s

Epoch 165/450 - Train Loss: 0.005234, Test Loss: 1.907651, Time: 93.34s, Weight Norm: 313.3283, Logit Scale: 8.3035

Evaluating denoising accuracy at epoch 165...
  Train Accuracy: 99.63% (100% acc: 96.7%), Test Accuracy: 71.59% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.8 (max: 4), Time: 4.74s

Epoch 166/450 - Train Loss: 0.004910, Test Loss: 1.790524, Time: 92.95s, Weight Norm: 309.1984, Logit Scale: 8.3052

Evaluating denoising accuracy at epoch 166...
  Train Accuracy: 99.40% (100% acc: 89.2%), Test Accuracy: 70.21% (100% acc: 6.9%)
  Train Best Iter: 1.1±0.3 (max: 4), Test Best Iter: 1.3±0.5 (max: 2), Time: 4.74s

Epoch 167/450 - Train Loss: 0.004686, Test Loss: 1.892353, Time: 93.34s, Weight Norm: 305.1418, Logit Scale: 8.3204

Evaluating denoising accuracy at epoch 167...
  Train Accuracy: 99.03% (100% acc: 97.5%), Test Accuracy: 72.55% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.2±0.4 (max: 2), Time: 4.72s

Epoch 168/450 - Train Loss: 0.005256, Test Loss: 1.851097, Time: 93.15s, Weight Norm: 301.2319, Logit Scale: 8.3307

Evaluating denoising accuracy at epoch 168...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 71.86% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.71s

Epoch 169/450 - Train Loss: 0.004717, Test Loss: 2.135633, Time: 93.00s, Weight Norm: 297.2804, Logit Scale: 8.3400

Evaluating denoising accuracy at epoch 169...
  Train Accuracy: 99.80% (100% acc: 96.7%), Test Accuracy: 70.76% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±0.7 (max: 3), Time: 4.72s

Epoch 170/450 - Train Loss: 0.005337, Test Loss: 1.932816, Time: 93.23s, Weight Norm: 293.5555, Logit Scale: 8.3428

Evaluating denoising accuracy at epoch 170...
  Train Accuracy: 99.87% (100% acc: 96.7%), Test Accuracy: 72.55% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.73s

Epoch 171/450 - Train Loss: 0.004723, Test Loss: 1.742143, Time: 93.13s, Weight Norm: 289.7132, Logit Scale: 8.3486

Evaluating denoising accuracy at epoch 171...
  Train Accuracy: 99.37% (100% acc: 94.2%), Test Accuracy: 73.66% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.2±0.6 (max: 4), Time: 4.72s

Epoch 172/450 - Train Loss: 0.004649, Test Loss: 1.831692, Time: 93.32s, Weight Norm: 285.9820, Logit Scale: 8.3526

Evaluating denoising accuracy at epoch 172...
  Train Accuracy: 99.40% (100% acc: 95.8%), Test Accuracy: 71.86% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.4±0.5 (max: 2), Time: 4.73s

Epoch 173/450 - Train Loss: 0.004636, Test Loss: 1.835526, Time: 93.15s, Weight Norm: 282.3383, Logit Scale: 8.3674

Evaluating denoising accuracy at epoch 173...
  Train Accuracy: 99.80% (100% acc: 96.7%), Test Accuracy: 71.31% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.6±0.7 (max: 4), Time: 4.73s

Epoch 174/450 - Train Loss: 0.004438, Test Loss: 2.051275, Time: 93.10s, Weight Norm: 278.7402, Logit Scale: 8.3798

Evaluating denoising accuracy at epoch 174...
  Train Accuracy: 99.87% (100% acc: 96.7%), Test Accuracy: 72.55% (100% acc: 13.8%)
  Train Best Iter: 1.1±0.3 (max: 4), Test Best Iter: 1.6±0.8 (max: 4), Time: 4.73s

Epoch 175/450 - Train Loss: 0.004352, Test Loss: 2.013681, Time: 93.07s, Weight Norm: 275.2191, Logit Scale: 8.4013

Evaluating denoising accuracy at epoch 175...
  Train Accuracy: 99.80% (100% acc: 95.8%), Test Accuracy: 72.41% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.72s

Epoch 176/450 - Train Loss: 0.005445, Test Loss: 1.930217, Time: 92.98s, Weight Norm: 271.8430, Logit Scale: 8.4052

Evaluating denoising accuracy at epoch 176...
  Train Accuracy: 99.87% (100% acc: 96.7%), Test Accuracy: 72.83% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.74s

Epoch 177/450 - Train Loss: 0.004356, Test Loss: 2.018170, Time: 92.81s, Weight Norm: 268.3829, Logit Scale: 8.4062

Evaluating denoising accuracy at epoch 177...
  Train Accuracy: 99.80% (100% acc: 96.7%), Test Accuracy: 73.79% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.5±0.6 (max: 3), Time: 4.71s

Epoch 178/450 - Train Loss: 0.004405, Test Loss: 1.743240, Time: 92.88s, Weight Norm: 265.0825, Logit Scale: 8.4259

Evaluating denoising accuracy at epoch 178...
  Train Accuracy: 99.73% (100% acc: 95.8%), Test Accuracy: 72.00% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 2), Test Best Iter: 1.3±0.6 (max: 3), Time: 4.71s

Epoch 179/450 - Train Loss: 0.004406, Test Loss: 1.940387, Time: 93.04s, Weight Norm: 261.7689, Logit Scale: 8.4401

Evaluating denoising accuracy at epoch 179...
  Train Accuracy: 99.97% (100% acc: 99.2%), Test Accuracy: 73.93% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.2±0.4 (max: 2), Time: 4.76s

Epoch 180/450 - Train Loss: 0.004834, Test Loss: 1.995986, Time: 93.04s, Weight Norm: 258.6169, Logit Scale: 8.4527

Evaluating denoising accuracy at epoch 180...
  Train Accuracy: 99.90% (100% acc: 97.5%), Test Accuracy: 73.38% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.74s

Saved checkpoint to output/mini_arc_eqm5/checkpoints/20260113_215545_epoch_180_checkpoint.pt
Copied checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/20260113_215545_epoch_180_checkpoint.pt
Epoch 181/450 - Train Loss: 0.004264, Test Loss: 1.940112, Time: 93.20s, Weight Norm: 255.3934, Logit Scale: 8.4540

Evaluating denoising accuracy at epoch 181...
  Train Accuracy: 99.90% (100% acc: 97.5%), Test Accuracy: 74.90% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.7 (max: 3), Time: 4.71s

Epoch 182/450 - Train Loss: 0.005531, Test Loss: 1.934686, Time: 93.02s, Weight Norm: 252.3991, Logit Scale: 8.4497

Evaluating denoising accuracy at epoch 182...
  Train Accuracy: 99.83% (100% acc: 95.8%), Test Accuracy: 72.14% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.4 (max: 2), Time: 4.72s

Epoch 183/450 - Train Loss: 0.003606, Test Loss: 1.969078, Time: 93.12s, Weight Norm: 249.2003, Logit Scale: 8.4530

Evaluating denoising accuracy at epoch 183...
  Train Accuracy: 99.77% (100% acc: 95.8%), Test Accuracy: 72.28% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.72s

Epoch 184/450 - Train Loss: 0.004880, Test Loss: 1.876322, Time: 93.22s, Weight Norm: 246.3566, Logit Scale: 8.4669

Evaluating denoising accuracy at epoch 184...
  Train Accuracy: 99.80% (100% acc: 95.0%), Test Accuracy: 74.76% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.5 (max: 2), Time: 4.71s

Epoch 185/450 - Train Loss: 0.004065, Test Loss: 1.899863, Time: 93.11s, Weight Norm: 243.3385, Logit Scale: 8.4641

Evaluating denoising accuracy at epoch 185...
  Train Accuracy: 99.70% (100% acc: 94.2%), Test Accuracy: 73.79% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.5±0.9 (max: 5), Time: 4.72s

Epoch 186/450 - Train Loss: 0.005339, Test Loss: 1.930593, Time: 93.00s, Weight Norm: 240.5709, Logit Scale: 8.4613

Evaluating denoising accuracy at epoch 186...
  Train Accuracy: 99.70% (100% acc: 95.0%), Test Accuracy: 74.07% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.72s

Epoch 187/450 - Train Loss: 0.003673, Test Loss: 1.893105, Time: 93.12s, Weight Norm: 237.5979, Logit Scale: 8.4650

Evaluating denoising accuracy at epoch 187...
  Train Accuracy: 99.77% (100% acc: 95.0%), Test Accuracy: 73.24% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.6 (max: 3), Time: 4.72s

Epoch 188/450 - Train Loss: 0.004290, Test Loss: 1.924077, Time: 93.23s, Weight Norm: 234.8446, Logit Scale: 8.4792

Evaluating denoising accuracy at epoch 188...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 74.48% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.8 (max: 4), Time: 4.72s

Epoch 189/450 - Train Loss: 0.004771, Test Loss: 1.916678, Time: 92.95s, Weight Norm: 232.1590, Logit Scale: 8.4839

Evaluating denoising accuracy at epoch 189...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 74.90% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.72s

Epoch 190/450 - Train Loss: 0.004120, Test Loss: 1.822654, Time: 93.01s, Weight Norm: 229.4464, Logit Scale: 8.4941

Evaluating denoising accuracy at epoch 190...
  Train Accuracy: 99.80% (100% acc: 95.8%), Test Accuracy: 72.69% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 3), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.72s

Epoch 191/450 - Train Loss: 0.004672, Test Loss: 1.860306, Time: 93.06s, Weight Norm: 226.8750, Logit Scale: 8.5018

Evaluating denoising accuracy at epoch 191...
  Train Accuracy: 99.87% (100% acc: 96.7%), Test Accuracy: 74.21% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 2), Test Best Iter: 1.2±0.6 (max: 3), Time: 4.73s

Epoch 192/450 - Train Loss: 0.004042, Test Loss: 1.873093, Time: 93.15s, Weight Norm: 224.3424, Logit Scale: 8.5241

Evaluating denoising accuracy at epoch 192...
  Train Accuracy: 99.33% (100% acc: 96.7%), Test Accuracy: 74.21% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.73s

Epoch 193/450 - Train Loss: 0.004519, Test Loss: 1.936767, Time: 93.27s, Weight Norm: 221.8268, Logit Scale: 8.5238

Evaluating denoising accuracy at epoch 193...
  Train Accuracy: 99.67% (100% acc: 95.0%), Test Accuracy: 75.17% (100% acc: 13.8%)
  Train Best Iter: 1.1±0.4 (max: 5), Test Best Iter: 1.5±0.7 (max: 4), Time: 4.75s

Epoch 194/450 - Train Loss: 0.004297, Test Loss: 1.951551, Time: 92.95s, Weight Norm: 219.3424, Logit Scale: 8.5246

Evaluating denoising accuracy at epoch 194...
  Train Accuracy: 99.83% (100% acc: 97.5%), Test Accuracy: 72.97% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.73s

Epoch 195/450 - Train Loss: 0.004279, Test Loss: 1.945919, Time: 93.08s, Weight Norm: 216.9246, Logit Scale: 8.5334

Evaluating denoising accuracy at epoch 195...
  Train Accuracy: 99.87% (100% acc: 96.7%), Test Accuracy: 76.14% (100% acc: 17.2%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.77s

Epoch 196/450 - Train Loss: 0.003700, Test Loss: 1.866051, Time: 93.04s, Weight Norm: 214.4574, Logit Scale: 8.5453

Evaluating denoising accuracy at epoch 196...
  Train Accuracy: 99.43% (100% acc: 98.3%), Test Accuracy: 73.52% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.5±0.6 (max: 3), Time: 4.80s

Epoch 197/450 - Train Loss: 0.004158, Test Loss: 1.946941, Time: 93.26s, Weight Norm: 212.1963, Logit Scale: 8.5564

Evaluating denoising accuracy at epoch 197...
  Train Accuracy: 99.37% (100% acc: 96.7%), Test Accuracy: 71.86% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.2 (max: 2), Test Best Iter: 1.5±0.6 (max: 3), Time: 4.72s

Epoch 198/450 - Train Loss: 0.003984, Test Loss: 1.939447, Time: 92.94s, Weight Norm: 209.9301, Logit Scale: 8.5753

Evaluating denoising accuracy at epoch 198...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 71.03% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.75s

Epoch 199/450 - Train Loss: 0.004483, Test Loss: 1.869840, Time: 92.95s, Weight Norm: 207.7992, Logit Scale: 8.5903

Evaluating denoising accuracy at epoch 199...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 73.52% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.5 (max: 2), Time: 4.81s

Epoch 200/450 - Train Loss: 0.004911, Test Loss: 1.674194, Time: 93.29s, Weight Norm: 205.6462, Logit Scale: 8.5786

Evaluating denoising accuracy at epoch 200...
  Train Accuracy: 99.07% (100% acc: 85.0%), Test Accuracy: 69.10% (100% acc: 6.9%)
  Train Best Iter: 1.1±0.3 (max: 3), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.72s

Epoch 201/450 - Train Loss: 0.003653, Test Loss: 1.957439, Time: 93.01s, Weight Norm: 203.2576, Logit Scale: 8.5692

Evaluating denoising accuracy at epoch 201...
  Train Accuracy: 99.87% (100% acc: 97.5%), Test Accuracy: 73.93% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.3 (max: 4), Test Best Iter: 1.3±0.6 (max: 3), Time: 4.75s

Epoch 202/450 - Train Loss: 0.004448, Test Loss: 1.849761, Time: 93.37s, Weight Norm: 201.3109, Logit Scale: 8.5871

Evaluating denoising accuracy at epoch 202...
  Train Accuracy: 99.83% (100% acc: 95.8%), Test Accuracy: 74.34% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.4 (max: 2), Time: 4.75s

Epoch 203/450 - Train Loss: 0.003479, Test Loss: 1.907632, Time: 92.98s, Weight Norm: 199.0604, Logit Scale: 8.5930

Evaluating denoising accuracy at epoch 203...
  Train Accuracy: 99.87% (100% acc: 97.5%), Test Accuracy: 74.21% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.74s

Epoch 204/450 - Train Loss: 0.004098, Test Loss: 1.927544, Time: 93.00s, Weight Norm: 197.1289, Logit Scale: 8.6134

Evaluating denoising accuracy at epoch 204...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 72.83% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.2±0.5 (max: 3), Time: 4.71s

Epoch 205/450 - Train Loss: 0.003670, Test Loss: 2.013036, Time: 93.00s, Weight Norm: 195.1292, Logit Scale: 8.6287

Evaluating denoising accuracy at epoch 205...
  Train Accuracy: 99.77% (100% acc: 96.7%), Test Accuracy: 70.90% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.6±1.0 (max: 5), Time: 4.72s

Epoch 206/450 - Train Loss: 0.004836, Test Loss: 1.958540, Time: 92.82s, Weight Norm: 193.3451, Logit Scale: 8.6247

Evaluating denoising accuracy at epoch 206...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 76.14% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.5±0.5 (max: 2), Time: 4.73s

Epoch 207/450 - Train Loss: 0.003714, Test Loss: 1.860307, Time: 93.02s, Weight Norm: 191.3391, Logit Scale: 8.6195

Evaluating denoising accuracy at epoch 207...
  Train Accuracy: 99.77% (100% acc: 96.7%), Test Accuracy: 73.24% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.74s

Epoch 208/450 - Train Loss: 0.004378, Test Loss: 1.783950, Time: 93.26s, Weight Norm: 189.4398, Logit Scale: 8.6023

Evaluating denoising accuracy at epoch 208...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 74.76% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.7 (max: 4), Time: 4.73s

Epoch 209/450 - Train Loss: 0.004557, Test Loss: 1.892660, Time: 92.94s, Weight Norm: 187.7037, Logit Scale: 8.6086

Evaluating denoising accuracy at epoch 209...
  Train Accuracy: 99.80% (100% acc: 97.5%), Test Accuracy: 75.03% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.71s

Epoch 210/450 - Train Loss: 0.003467, Test Loss: 1.817637, Time: 92.95s, Weight Norm: 185.7160, Logit Scale: 8.5969

Evaluating denoising accuracy at epoch 210...
  Train Accuracy: 99.90% (100% acc: 98.3%), Test Accuracy: 76.28% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.5±0.8 (max: 3), Time: 4.74s

Saved checkpoint to output/mini_arc_eqm5/checkpoints/20260113_215545_epoch_210_checkpoint.pt
Copied checkpoint to Google Drive: /content/drive/MyDrive/sparse_arc/20260113_215545_epoch_210_checkpoint.pt
Epoch 211/450 - Train Loss: 0.004248, Test Loss: 2.005034, Time: 92.92s, Weight Norm: 184.1042, Logit Scale: 8.6045

Evaluating denoising accuracy at epoch 211...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 74.76% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.5±1.0 (max: 6), Time: 4.72s

Epoch 212/450 - Train Loss: 0.003680, Test Loss: 1.811990, Time: 92.92s, Weight Norm: 182.3570, Logit Scale: 8.6083

Evaluating denoising accuracy at epoch 212...
  Train Accuracy: 99.90% (100% acc: 97.5%), Test Accuracy: 74.34% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.7 (max: 4), Time: 4.72s

Epoch 213/450 - Train Loss: 0.003457, Test Loss: 1.868272, Time: 92.96s, Weight Norm: 180.6222, Logit Scale: 8.6273

Evaluating denoising accuracy at epoch 213...
  Train Accuracy: 99.97% (100% acc: 99.2%), Test Accuracy: 74.21% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.9 (max: 5), Time: 4.71s

Epoch 214/450 - Train Loss: 0.004424, Test Loss: 1.917380, Time: 93.08s, Weight Norm: 179.1926, Logit Scale: 8.6284

Evaluating denoising accuracy at epoch 214...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 75.03% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.73s

Epoch 215/450 - Train Loss: 0.003778, Test Loss: 1.751146, Time: 92.99s, Weight Norm: 177.5509, Logit Scale: 8.6308

Evaluating denoising accuracy at epoch 215...
  Train Accuracy: 99.97% (100% acc: 99.2%), Test Accuracy: 75.17% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 2), Time: 4.71s

Epoch 216/450 - Train Loss: 0.003719, Test Loss: 1.888037, Time: 93.05s, Weight Norm: 175.9635, Logit Scale: 8.6332

Evaluating denoising accuracy at epoch 216...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 70.76% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.7 (max: 4), Time: 4.72s

Epoch 217/450 - Train Loss: 0.003664, Test Loss: 1.892459, Time: 93.17s, Weight Norm: 174.4371, Logit Scale: 8.6390

Evaluating denoising accuracy at epoch 217...
  Train Accuracy: 99.87% (100% acc: 98.3%), Test Accuracy: 72.00% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.73s

Epoch 218/450 - Train Loss: 0.004539, Test Loss: 1.950764, Time: 93.10s, Weight Norm: 173.0682, Logit Scale: 8.6349

Evaluating denoising accuracy at epoch 218...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 73.52% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.4 (max: 2), Time: 4.74s

Epoch 219/450 - Train Loss: 0.003496, Test Loss: 1.909206, Time: 93.50s, Weight Norm: 171.4191, Logit Scale: 8.6377

Evaluating denoising accuracy at epoch 219...
  Train Accuracy: 99.97% (100% acc: 99.2%), Test Accuracy: 71.17% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.74s

Epoch 220/450 - Train Loss: 0.003814, Test Loss: 1.976290, Time: 93.20s, Weight Norm: 169.9724, Logit Scale: 8.6478

Evaluating denoising accuracy at epoch 220...
  Train Accuracy: 99.87% (100% acc: 97.5%), Test Accuracy: 73.52% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.7 (max: 4), Time: 4.73s

Epoch 221/450 - Train Loss: 0.003945, Test Loss: 1.876289, Time: 93.18s, Weight Norm: 168.6109, Logit Scale: 8.6641

Evaluating denoising accuracy at epoch 221...
  Train Accuracy: 99.73% (100% acc: 96.7%), Test Accuracy: 73.79% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.7 (max: 4), Time: 4.72s

Epoch 222/450 - Train Loss: 0.005068, Test Loss: 1.724527, Time: 93.00s, Weight Norm: 167.4726, Logit Scale: 8.6524

Evaluating denoising accuracy at epoch 222...
  Train Accuracy: 99.40% (100% acc: 98.3%), Test Accuracy: 75.17% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±0.7 (max: 4), Time: 4.75s

Epoch 223/450 - Train Loss: 0.003199, Test Loss: 1.700463, Time: 93.21s, Weight Norm: 165.8436, Logit Scale: 8.6316

Evaluating denoising accuracy at epoch 223...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 74.90% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.7 (max: 3), Time: 4.71s

Epoch 224/450 - Train Loss: 0.003493, Test Loss: 1.850682, Time: 93.17s, Weight Norm: 164.5455, Logit Scale: 8.6455

Evaluating denoising accuracy at epoch 224...
  Train Accuracy: 99.87% (100% acc: 98.3%), Test Accuracy: 74.48% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.71s

Epoch 225/450 - Train Loss: 0.004343, Test Loss: 1.918914, Time: 93.12s, Weight Norm: 163.4500, Logit Scale: 8.6496

Evaluating denoising accuracy at epoch 225...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 75.31% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.71s

Epoch 226/450 - Train Loss: 0.003442, Test Loss: 1.740210, Time: 93.14s, Weight Norm: 162.0198, Logit Scale: 8.6422

Evaluating denoising accuracy at epoch 226...
  Train Accuracy: 99.90% (100% acc: 98.3%), Test Accuracy: 75.45% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.74s

Epoch 227/450 - Train Loss: 0.003096, Test Loss: 1.721196, Time: 93.06s, Weight Norm: 160.7193, Logit Scale: 8.6577

Evaluating denoising accuracy at epoch 227...
  Train Accuracy: 99.80% (100% acc: 96.7%), Test Accuracy: 75.59% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.73s

Epoch 228/450 - Train Loss: 0.004415, Test Loss: 1.936086, Time: 93.13s, Weight Norm: 159.7743, Logit Scale: 8.6609

Evaluating denoising accuracy at epoch 228...
  Train Accuracy: 99.67% (100% acc: 96.7%), Test Accuracy: 74.76% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.6±0.6 (max: 3), Time: 4.72s

Epoch 229/450 - Train Loss: 0.003341, Test Loss: 1.791871, Time: 93.08s, Weight Norm: 158.5103, Logit Scale: 8.6630

Evaluating denoising accuracy at epoch 229...
  Train Accuracy: 99.87% (100% acc: 97.5%), Test Accuracy: 72.00% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.4±0.6 (max: 3), Time: 4.72s

Epoch 230/450 - Train Loss: 0.003722, Test Loss: 1.905489, Time: 93.19s, Weight Norm: 157.4454, Logit Scale: 8.6713

Evaluating denoising accuracy at epoch 230...
  Train Accuracy: 99.90% (100% acc: 97.5%), Test Accuracy: 73.66% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.5±0.5 (max: 2), Time: 4.72s

Epoch 231/450 - Train Loss: 0.004050, Test Loss: 1.860284, Time: 93.08s, Weight Norm: 156.3070, Logit Scale: 8.6630

Evaluating denoising accuracy at epoch 231...
  Train Accuracy: 99.90% (100% acc: 99.2%), Test Accuracy: 74.07% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.4 (max: 2), Time: 4.72s

Epoch 232/450 - Train Loss: 0.003270, Test Loss: 1.894451, Time: 93.10s, Weight Norm: 155.2219, Logit Scale: 8.6769

Evaluating denoising accuracy at epoch 232...
  Train Accuracy: 99.93% (100% acc: 98.3%), Test Accuracy: 74.34% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.2±0.4 (max: 2), Time: 4.72s

Epoch 233/450 - Train Loss: 0.004136, Test Loss: 1.739481, Time: 93.16s, Weight Norm: 154.2971, Logit Scale: 8.6779

Evaluating denoising accuracy at epoch 233...
  Train Accuracy: 99.97% (100% acc: 99.2%), Test Accuracy: 77.38% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.2±0.4 (max: 2), Time: 4.70s

Epoch 234/450 - Train Loss: 0.003961, Test Loss: 1.769279, Time: 92.92s, Weight Norm: 153.2269, Logit Scale: 8.6632

Evaluating denoising accuracy at epoch 234...
  Train Accuracy: 99.93% (100% acc: 99.2%), Test Accuracy: 75.31% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.2±0.4 (max: 2), Time: 4.75s

Epoch 235/450 - Train Loss: 0.003324, Test Loss: 1.830122, Time: 92.97s, Weight Norm: 152.1895, Logit Scale: 8.6590

Evaluating denoising accuracy at epoch 235...
  Train Accuracy: 99.83% (100% acc: 96.7%), Test Accuracy: 72.28% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±1.0 (max: 5), Time: 4.70s

Epoch 236/450 - Train Loss: 0.003793, Test Loss: 1.784100, Time: 93.13s, Weight Norm: 151.1815, Logit Scale: 8.6538

Evaluating denoising accuracy at epoch 236...
  Train Accuracy: 100.00% (100% acc: 100.0%), Test Accuracy: 75.72% (100% acc: 10.3%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.73s

Epoch 237/450 - Train Loss: 0.003271, Test Loss: 1.839692, Time: 93.16s, Weight Norm: 150.2439, Logit Scale: 8.6689

Evaluating denoising accuracy at epoch 237...
  Train Accuracy: 99.90% (100% acc: 98.3%), Test Accuracy: 74.76% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.1 (max: 2), Test Best Iter: 1.3±0.5 (max: 3), Time: 4.74s

Epoch 238/450 - Train Loss: 0.004153, Test Loss: 1.765585, Time: 93.46s, Weight Norm: 149.4229, Logit Scale: 8.6740

Evaluating denoising accuracy at epoch 238...
  Train Accuracy: 99.97% (100% acc: 99.2%), Test Accuracy: 76.83% (100% acc: 13.8%)
  Train Best Iter: 1.0±0.0 (max: 1), Test Best Iter: 1.4±0.7 (max: 3), Time: 4.73s
