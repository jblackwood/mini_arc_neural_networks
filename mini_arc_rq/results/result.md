Configuration:
{'data_dir': PosixPath('data/MINI-ARC'),
 'output_dir': PosixPath('output/mini_arc_rq'),
 'test_ratio': 0.2,
 'random_seed': 42,
 'max_augmentations': 500,
 'd_model': 256,
 'nhead': 8,
 'num_layers': 8,
 'dim_feedforward': 1024,
 'dropout': 0.1,
 'task_codebook_size': 64,
 'task_codebook_dim': 16,
 'num_task_latent_tokens': 10,
 'batch_size': 256,
 'num_epochs': 300,
 'learning_rate': 0.0001,
 'task_embedding_lr': 0.01,
 'weight_decay': 0,
 'task_embedding_weight_decay': 0,
 'label_smoothing': 0,
 'mode': 'train',
 'checkpoint_save_interval': 50,
 'vocab_size': 11,
 'eval_denoise_epoch_interval': 10,
 'google_drive_dir': '/content/drive/MyDrive/sparse_arc',
 'load_model_path': None,
 'timestamp': '20260204_213308',
 'tensorboard_log_dir': 'output/mini_arc_rq/runs/20260204_213308_model',
 'model_save_dir': 'output/mini_arc_rq/models',
 'model_save_path': 'output/mini_arc_rq/models/20260204_213308_model.pt',
 'checkpoint_dir': 'output/mini_arc_rq/checkpoints',
 'train_data_dir': 'output/mini_arc_rq/train',
 'test_data_dir': 'output/mini_arc_rq/test'}

MINI-ARC dataset already exists in 'data/MINI-ARC'. Skipping download.
Output directories already exist and contain data:
  Train directory: output/mini_arc_rq/train (39336 files)
  Test directory: output/mini_arc_rq/test (8384 files)
Skipping dataset creation.
Using device: cuda
/usr/local/lib/python3.12/dist-packages/torch/__init__.py:1617: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  _C._set_float32_matmul_precision(precision)
Train dataset size: 47720
Test dataset size: 8384

Building task_id to task_index mapping...
Found 47720 unique tasks
Sample task_ids: ['miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipa', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipa_0to4_4to0', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipd', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipv', 'miniarc-1_3_5_l6aejqqqc1b47pjr5g4-flipv_0to4_4to0']
Compiling model with torch.compile...
Total parameters: 13,982,795
Trainable parameters: 13,982,795
Non-trainable (codebook): 1,024
  Task embedding: 7,635,200 (54.6%)
  Other parameters: 6,347,595 (45.4%)

Starting training...
W0204 21:33:29.008000 9471 torch/_inductor/utils.py:1558] [0/0] Not enough SMs to use max_autotune_gemm mode
Epoch 1/300 - Train Loss: 1.651458, Test Loss: 1.636952, Time: 118.39s, Weight Norm: 2765.3262, Logit Scale (Grid): 1.9698, (Task): 4.5122
  Layer Mean Squares - Task Emb: 0.999295, Token Emb: 1.009769, Task Proj: 0.020785, Grid Out: 0.001378, Task Out: 0.001243, Transformer: 0.001147
Epoch 2/300 - Train Loss: 1.543437, Test Loss: 1.619924, Time: 86.50s, Weight Norm: 2765.3201, Logit Scale (Grid): 2.0004, (Task): 4.4759
  Layer Mean Squares - Task Emb: 0.999287, Token Emb: 1.010001, Task Proj: 0.020719, Grid Out: 0.001421, Task Out: 0.001223, Transformer: 0.001151
Epoch 3/300 - Train Loss: 1.506728, Test Loss: 1.501227, Time: 87.14s, Weight Norm: 2765.3149, Logit Scale (Grid): 2.0306, (Task): 4.4468
  Layer Mean Squares - Task Emb: 0.999277, Token Emb: 1.009346, Task Proj: 0.020628, Grid Out: 0.001464, Task Out: 0.001207, Transformer: 0.001158
Epoch 4/300 - Train Loss: 1.427826, Test Loss: 1.429291, Time: 87.53s, Weight Norm: 2765.2949, Logit Scale (Grid): 2.0539, (Task): 4.4231
  Layer Mean Squares - Task Emb: 0.999257, Token Emb: 1.008418, Task Proj: 0.020452, Grid Out: 0.001498, Task Out: 0.001194, Transformer: 0.001165
Epoch 5/300 - Train Loss: 1.383969, Test Loss: 1.384376, Time: 87.46s, Weight Norm: 2765.2700, Logit Scale (Grid): 2.0769, (Task): 4.4036
  Layer Mean Squares - Task Emb: 0.999234, Token Emb: 1.007695, Task Proj: 0.020265, Grid Out: 0.001532, Task Out: 0.001184, Transformer: 0.001171
Epoch 6/300 - Train Loss: 1.347143, Test Loss: 1.357088, Time: 87.07s, Weight Norm: 2765.2454, Logit Scale (Grid): 2.0986, (Task): 4.3859
  Layer Mean Squares - Task Emb: 0.999211, Token Emb: 1.007157, Task Proj: 0.020102, Grid Out: 0.001564, Task Out: 0.001174, Transformer: 0.001178