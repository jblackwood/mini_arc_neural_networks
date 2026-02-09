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
 'batch_size': 128,
 'num_epochs': 150,
 'learning_rate': 0.0002,
 'lambd': 0.05,
 'mode': 'train',
 'checkpoint_save_interval': 25,
 'vocab_size': 10,
 'google_drive_dir': '/content/drive/MyDrive/sparse_arc',
 'load_model_path': None,
 'timestamp': '20260209_055632',
 'tensorboard_log_dir': 'output/mini_arc_jepa/runs/20260209_055632_model',
 'model_save_dir': 'output/mini_arc_jepa/models',
 'model_save_path': 'output/mini_arc_jepa/models/20260209_055632_model.pt',
 'checkpoint_dir': 'output/mini_arc_jepa/checkpoints',
 'train_data_dir': 'output/mini_arc_jepa/train',
 'test_data_dir': 'output/mini_arc_jepa/test'}

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
JEPA model has 1,654,144 trainable parameters
Prediction model has 1,588,746 trainable parameters
Total parameters: 3,242,890

Starting training...
Epoch 1/150 - Train JEPA Loss: 0.515991, Train Pred Loss: 1.175712, Test Pred Loss: 1.174907, Time: 306.76s
  JEPA Loss Components - Train Sim: 0.065497, Train SigReg: 9.075383
  Accuracy - Train: 62.27%, Train Perfect: 0.87%, Test: 62.88%, Test Perfect: 0.50%
  Model Norms - JEPA: 88.1220, Pred: 84.2358, JEPA Out Scale: 13.6842, Pred Out Scale: 1.9275
Epoch 2/150 - Train JEPA Loss: 0.385888, Train Pred Loss: 1.030238, Test Pred Loss: 1.054858, Time: 294.51s
  JEPA Loss Components - Train Sim: 0.068894, Train SigReg: 6.408769
  Accuracy - Train: 65.26%, Train Perfect: 1.28%, Test: 65.13%, Test Perfect: 0.33%
  Model Norms - JEPA: 88.2422, Pred: 84.6499, JEPA Out Scale: 13.8001, Pred Out Scale: 2.0329
Epoch 3/150 - Train JEPA Loss: 0.359283, Train Pred Loss: 0.987583, Test Pred Loss: 1.017127, Time: 293.62s
  JEPA Loss Components - Train Sim: 0.068527, Train SigReg: 5.883637
  Accuracy - Train: 66.17%, Train Perfect: 1.36%, Test: 65.86%, Test Perfect: 0.31%
  Model Norms - JEPA: 88.3970, Pred: 84.9683, JEPA Out Scale: 13.9696, Pred Out Scale: 2.1217
Epoch 4/150 - Train JEPA Loss: 0.333790, Train Pred Loss: 0.965568, Test Pred Loss: 0.996093, Time: 294.46s
  JEPA Loss Components - Train Sim: 0.072118, Train SigReg: 5.305561
  Accuracy - Train: 66.69%, Train Perfect: 1.38%, Test: 66.27%, Test Perfect: 0.30%
  Model Norms - JEPA: 88.5696, Pred: 85.2591, JEPA Out Scale: 14.1584, Pred Out Scale: 2.1897
Epoch 5/150 - Train JEPA Loss: 0.311444, Train Pred Loss: 0.949179, Test Pred Loss: 0.980446, Time: 293.50s
  JEPA Loss Components - Train Sim: 0.073938, Train SigReg: 4.824050
  Accuracy - Train: 67.13%, Train Perfect: 1.38%, Test: 66.64%, Test Perfect: 0.32%
  Model Norms - JEPA: 88.7520, Pred: 85.5434, JEPA Out Scale: 14.3656, Pred Out Scale: 2.2521
Epoch 6/150 - Train JEPA Loss: 0.293184, Train Pred Loss: 0.934572, Test Pred Loss: 0.967805, Time: 293.58s
  JEPA Loss Components - Train Sim: 0.077042, Train SigReg: 4.399875
  Accuracy - Train: 67.39%, Train Perfect: 1.36%, Test: 66.82%, Test Perfect: 0.41%
  Model Norms - JEPA: 88.9030, Pred: 85.8097, JEPA Out Scale: 14.5037, Pred Out Scale: 2.3116