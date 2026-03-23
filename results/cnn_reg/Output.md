python CNN_Reg.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774275295.393074  225611 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1774275295.393308  225611 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1774275295.430360  225611 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774275296.250265  225611 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1774275296.250597  225611 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
============================================================
LOADING DATASET...
============================================================

Classes found (11): ['Earth', 'Jupiter', 'MakeMake', 'Mars', 'Mercury', 'Moon', 'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus']
  Earth: 149 images
  Jupiter: 149 images
  MakeMake: 149 images
  Mars: 149 images
  Mercury: 149 images
  Moon: 148 images
  Neptune: 149 images
  Pluto: 149 images
  Saturn: 149 images
  Uranus: 149 images
  Venus: 149 images
Found 1319 images belonging to 11 classes.
Found 319 images belonging to 11 classes.

Total number of classes: 11
Training samples: 1319
Validation samples: 319

============================================================
BUILDING MODEL...
============================================================
E0000 00:00:1774275297.930301  225611 cuda_executor.cc:1737] INTERNAL: CUDA Runtime error: Failed call to cudaGetRuntimeVersion: Error loading CUDA libraries. GPU will not be used.: Error loading CUDA libraries. GPU will not be used.
W0000 00:00:1774275297.930638  225723 cuda_executor.cc:1755] Failed to determine cuDNN version (Note that this is expected if the application doesn't link the cuDNN plugin): INTERNAL: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
W0000 00:00:1774275297.948682  225611 gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
Model: "Planets_CNN_Regularized"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 128, 128, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 128, 128, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation (Activation)              │ (None, 128, 128, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 128, 128, 32)        │           9,248 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 128, 128, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_1 (Activation)            │ (None, 128, 128, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 64, 64, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 64, 64, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 64, 64, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 64, 64, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_2 (Activation)            │ (None, 64, 64, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 64, 64, 64)          │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_3                │ (None, 64, 64, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_3 (Activation)            │ (None, 64, 64, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 32, 32, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 32, 32, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 32, 32, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_4                │ (None, 32, 32, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_4 (Activation)            │ (None, 32, 32, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 32, 32, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_5                │ (None, 32, 32, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_5 (Activation)            │ (None, 32, 32, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_6 (Conv2D)                    │ (None, 16, 16, 256)         │         295,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_6                │ (None, 16, 16, 256)         │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_6 (Activation)            │ (None, 16, 16, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_7 (Conv2D)                    │ (None, 16, 16, 256)         │         590,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_7                │ (None, 16, 16, 256)         │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_7 (Activation)            │ (None, 16, 16, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 256)                 │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │          65,792 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_8                │ (None, 256)                 │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_8 (Activation)            │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_4 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_9                │ (None, 128)                 │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_9 (Activation)            │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_5 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 11)                  │           1,419 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,277,739 (4.87 MB)
 Trainable params: 1,275,051 (4.86 MB)
 Non-trainable params: 2,688 (10.50 KB)

============================================================
TRAINING STARTED...
============================================================
Epoch 1/10
I0000 00:00:1774275298.474785  225611 generator_dataset_op.cc:213] Memory patch applied: M_TRIM_THRESHOLD=128 kb was set.
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 210ms/step - accuracy: 0.2237 - loss: 3.4832 
Epoch 1: val_accuracy improved from None to 0.18182, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 226ms/step - accuracy: 0.3146 - loss: 3.1734 - val_accuracy: 0.1818 - val_loss: 3.7066 - learning_rate: 0.0010
Epoch 2/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 249ms/step - accuracy: 0.5176 - loss: 2.5380 
Epoch 2: val_accuracy did not improve from 0.18182
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 264ms/step - accuracy: 0.5315 - loss: 2.4319 - val_accuracy: 0.1317 - val_loss: 4.1733 - learning_rate: 0.0010
Epoch 3/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.5628 - loss: 2.2004 
Epoch 3: val_accuracy improved from 0.18182 to 0.19436, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 274ms/step - accuracy: 0.5762 - loss: 2.1228 - val_accuracy: 0.1944 - val_loss: 5.2108 - learning_rate: 0.0010
Epoch 4/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.6069 - loss: 1.9487 
Epoch 4: val_accuracy improved from 0.19436 to 0.34169, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.6315 - loss: 1.8880 - val_accuracy: 0.3417 - val_loss: 3.5461 - learning_rate: 0.0010
Epoch 5/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.6295 - loss: 1.8491 
Epoch 5: val_accuracy improved from 0.34169 to 0.42947, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.6194 - loss: 1.8361 - val_accuracy: 0.4295 - val_loss: 3.0688 - learning_rate: 0.0010
Epoch 6/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.6852 - loss: 1.6379 
Epoch 6: val_accuracy did not improve from 0.42947
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.6748 - loss: 1.6364 - val_accuracy: 0.4107 - val_loss: 2.2317 - learning_rate: 0.0010
Epoch 7/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.6553 - loss: 1.6241 
Epoch 7: val_accuracy did not improve from 0.42947
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.6543 - loss: 1.6265 - val_accuracy: 0.4201 - val_loss: 4.3025 - learning_rate: 0.0010
Epoch 8/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.6823 - loss: 1.4831 
Epoch 8: val_accuracy improved from 0.42947 to 0.57053, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 275ms/step - accuracy: 0.6793 - loss: 1.4825 - val_accuracy: 0.5705 - val_loss: 1.7758 - learning_rate: 0.0010
Epoch 9/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.7462 - loss: 1.3621 
Epoch 9: val_accuracy improved from 0.57053 to 0.62696, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 274ms/step - accuracy: 0.7407 - loss: 1.3541 - val_accuracy: 0.6270 - val_loss: 1.5980 - learning_rate: 0.0010
Epoch 10/10
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.7194 - loss: 1.3152 
Epoch 10: val_accuracy improved from 0.62696 to 0.73668, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 274ms/step - accuracy: 0.7301 - loss: 1.3068 - val_accuracy: 0.7367 - val_loss: 1.3339 - learning_rate: 0.0010
Restoring model weights from the end of the best epoch: 10.

============================================================
RESULTS
============================================================
Training history saved: training_history.png

Computing Confusion Matrix...
Confusion matrix saved: confusion_matrix.png

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

       Earth       1.00      1.00      1.00        29
     Jupiter       1.00      0.17      0.29        29
    MakeMake       0.63      1.00      0.77        29
        Mars       1.00      0.41      0.59        29
     Mercury       0.46      1.00      0.63        29
        Moon       0.89      0.59      0.71        29
     Neptune       1.00      1.00      1.00        29
       Pluto       1.00      0.93      0.96        29
      Saturn       0.00      0.00      0.00        29
      Uranus       0.48      1.00      0.65        29
       Venus       1.00      1.00      1.00        29

    accuracy                           0.74       319
   macro avg       0.77      0.74      0.69       319
weighted avg       0.77      0.74      0.69       319

============================================================
SUMMARY
============================================================
  Best Validation Accuracy : 0.7367 (73.67%)
  Best Validation Loss     : 1.3339
  Final Training Accuracy  : 0.7301 (73.01%)
  Total training epochs    : 10
  Model saved at           : best_planets_model.keras
============================================================

REGULARIZATION TECHNIQUES USED:
  1. L2 Weight Regularization (lambda=0.001)
  2. Dropout (rate=0.4)
  3. Batch Normalization (after each Conv block)
  4. Data Augmentation (rotation, shift, zoom, flip, brightness)
  5. Early Stopping (patience=15)
  6. Learning Rate Reduction (patience=7, factor=0.5)
  7. Global Average Pooling (instead of Flatten - fewer parameters)