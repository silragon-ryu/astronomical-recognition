WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774280005.391207  246635 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1774280005.391445  246635 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1774280005.427652  246635 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1774280006.243175  246635 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1774280006.243503  246635 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
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
E0000 00:00:1774280007.952579  246635 cuda_executor.cc:1737] INTERNAL: CUDA Runtime error: Failed call to cudaGetRuntimeVersion: Error loading CUDA libraries. GPU will not be used.: Error loading CUDA libraries. GPU will not be used.
W0000 00:00:1774280007.952826  246748 cuda_executor.cc:1755] Failed to determine cuDNN version (Note that this is expected if the application doesn't link the cuDNN plugin): INTERNAL: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
W0000 00:00:1774280007.971431  246635 gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
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
Epoch 1/100
I0000 00:00:1774280008.526435  246635 generator_dataset_op.cc:213] Memory patch applied: M_TRIM_THRESHOLD=128 kb was set.
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.2491 - loss: 3.4361 
Epoch 1: val_accuracy improved from None to 0.09091, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 26s 278ms/step - accuracy: 0.3654 - loss: 3.0510 - val_accuracy: 0.0909 - val_loss: 6.6103 - learning_rate: 0.0010
Epoch 2/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 270ms/step - accuracy: 0.4932 - loss: 2.5221 
Epoch 2: val_accuracy improved from 0.09091 to 0.13480, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 24s 288ms/step - accuracy: 0.5216 - loss: 2.4265 - val_accuracy: 0.1348 - val_loss: 10.5820 - learning_rate: 0.0010
Epoch 3/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 272ms/step - accuracy: 0.6092 - loss: 2.1070 
Epoch 3: val_accuracy improved from 0.13480 to 0.25078, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 24s 292ms/step - accuracy: 0.6005 - loss: 2.0912 - val_accuracy: 0.2508 - val_loss: 5.3261 - learning_rate: 0.0010
Epoch 4/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 272ms/step - accuracy: 0.6604 - loss: 1.8264 
Epoch 4: val_accuracy improved from 0.25078 to 0.29154, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 24s 293ms/step - accuracy: 0.6687 - loss: 1.7857 - val_accuracy: 0.2915 - val_loss: 4.4913 - learning_rate: 0.0010
Epoch 5/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 272ms/step - accuracy: 0.7127 - loss: 1.6678 
Epoch 5: val_accuracy did not improve from 0.29154
83/83 ━━━━━━━━━━━━━━━━━━━━ 24s 288ms/step - accuracy: 0.7210 - loss: 1.6337 - val_accuracy: 0.2790 - val_loss: 3.7643 - learning_rate: 0.0010
Epoch 6/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 261ms/step - accuracy: 0.7055 - loss: 1.5826 
Epoch 6: val_accuracy improved from 0.29154 to 0.53918, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 279ms/step - accuracy: 0.6937 - loss: 1.5973 - val_accuracy: 0.5392 - val_loss: 2.3471 - learning_rate: 0.0010
Epoch 7/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 258ms/step - accuracy: 0.7339 - loss: 1.5221 
Epoch 7: val_accuracy improved from 0.53918 to 0.66771, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 276ms/step - accuracy: 0.7422 - loss: 1.4712 - val_accuracy: 0.6677 - val_loss: 1.9191 - learning_rate: 0.0010
Epoch 8/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8034 - loss: 1.2859 
Epoch 8: val_accuracy did not improve from 0.66771
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.7635 - loss: 1.3681 - val_accuracy: 0.5047 - val_loss: 3.6506 - learning_rate: 0.0010
Epoch 9/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.7236 - loss: 1.3726 
Epoch 9: val_accuracy improved from 0.66771 to 0.69906, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 274ms/step - accuracy: 0.7339 - loss: 1.3458 - val_accuracy: 0.6991 - val_loss: 1.3236 - learning_rate: 0.0010
Epoch 10/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.7523 - loss: 1.2964 
Epoch 10: val_accuracy did not improve from 0.69906
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.7733 - loss: 1.2297 - val_accuracy: 0.5016 - val_loss: 2.4850 - learning_rate: 0.0010
Epoch 11/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.7509 - loss: 1.3158 
Epoch 11: val_accuracy did not improve from 0.69906
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.7559 - loss: 1.2679 - val_accuracy: 0.4859 - val_loss: 3.1318 - learning_rate: 0.0010
Epoch 12/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.7575 - loss: 1.2320 
Epoch 12: val_accuracy improved from 0.69906 to 0.78683, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 275ms/step - accuracy: 0.7809 - loss: 1.1902 - val_accuracy: 0.7868 - val_loss: 1.1039 - learning_rate: 0.0010
Epoch 13/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.7750 - loss: 1.1743 
Epoch 13: val_accuracy did not improve from 0.78683
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.7870 - loss: 1.1516 - val_accuracy: 0.6991 - val_loss: 1.4237 - learning_rate: 0.0010
Epoch 14/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8010 - loss: 1.0845 
Epoch 14: val_accuracy did not improve from 0.78683
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.7877 - loss: 1.0925 - val_accuracy: 0.3950 - val_loss: 2.2396 - learning_rate: 0.0010
Epoch 15/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.7780 - loss: 1.0708 
Epoch 15: val_accuracy did not improve from 0.78683
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.7801 - loss: 1.0645 - val_accuracy: 0.7116 - val_loss: 1.4733 - learning_rate: 0.0010
Epoch 16/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.7806 - loss: 1.0804 
Epoch 16: val_accuracy did not improve from 0.78683
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.7877 - loss: 1.0961 - val_accuracy: 0.6991 - val_loss: 1.3705 - learning_rate: 0.0010
Epoch 17/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8111 - loss: 1.0645 
Epoch 17: val_accuracy improved from 0.78683 to 0.89655, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.7832 - loss: 1.0949 - val_accuracy: 0.8966 - val_loss: 0.8120 - learning_rate: 0.0010
Epoch 18/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.7993 - loss: 1.0199 
Epoch 18: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8036 - loss: 1.0141 - val_accuracy: 0.6176 - val_loss: 1.6161 - learning_rate: 0.0010
Epoch 19/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 258ms/step - accuracy: 0.8202 - loss: 1.0286 
Epoch 19: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 274ms/step - accuracy: 0.8067 - loss: 1.0207 - val_accuracy: 0.7210 - val_loss: 1.1704 - learning_rate: 0.0010
Epoch 20/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.7859 - loss: 1.0896 
Epoch 20: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.7839 - loss: 1.1095 - val_accuracy: 0.7712 - val_loss: 1.1376 - learning_rate: 0.0010
Epoch 21/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8200 - loss: 0.9835 
Epoch 21: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8150 - loss: 0.9876 - val_accuracy: 0.8056 - val_loss: 1.3179 - learning_rate: 0.0010
Epoch 22/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8240 - loss: 0.8944 
Epoch 22: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8150 - loss: 0.9095 - val_accuracy: 0.5893 - val_loss: 1.5649 - learning_rate: 0.0010
Epoch 23/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8069 - loss: 0.9421 
Epoch 23: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8044 - loss: 0.9409 - val_accuracy: 0.6489 - val_loss: 1.3287 - learning_rate: 0.0010
Epoch 24/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8448 - loss: 0.8212 
Epoch 24: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.

Epoch 24: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8089 - loss: 0.9253 - val_accuracy: 0.6050 - val_loss: 1.7685 - learning_rate: 0.0010
Epoch 25/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8226 - loss: 0.8943 
Epoch 25: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8408 - loss: 0.8329 - val_accuracy: 0.8088 - val_loss: 0.7983 - learning_rate: 5.0000e-04
Epoch 26/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8641 - loss: 0.7313 
Epoch 26: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8741 - loss: 0.7137 - val_accuracy: 0.8903 - val_loss: 0.6936 - learning_rate: 5.0000e-04
Epoch 27/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8557 - loss: 0.7304 
Epoch 27: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8666 - loss: 0.7053 - val_accuracy: 0.7085 - val_loss: 1.2342 - learning_rate: 5.0000e-04
Epoch 28/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8883 - loss: 0.6744 
Epoch 28: val_accuracy did not improve from 0.89655
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8726 - loss: 0.6768 - val_accuracy: 0.8777 - val_loss: 0.6272 - learning_rate: 5.0000e-04
Epoch 29/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8657 - loss: 0.6882 
Epoch 29: val_accuracy improved from 0.89655 to 0.95298, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.8590 - loss: 0.6906 - val_accuracy: 0.9530 - val_loss: 0.6223 - learning_rate: 5.0000e-04
Epoch 30/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8884 - loss: 0.6431 
Epoch 30: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8832 - loss: 0.6367 - val_accuracy: 0.8871 - val_loss: 0.5887 - learning_rate: 5.0000e-04
Epoch 31/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8831 - loss: 0.6133 
Epoch 31: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8870 - loss: 0.6000 - val_accuracy: 0.7994 - val_loss: 0.9704 - learning_rate: 5.0000e-04
Epoch 32/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8693 - loss: 0.6362 
Epoch 32: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8704 - loss: 0.6324 - val_accuracy: 0.8464 - val_loss: 0.6212 - learning_rate: 5.0000e-04
Epoch 33/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8637 - loss: 0.6520 
Epoch 33: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8650 - loss: 0.6582 - val_accuracy: 0.8025 - val_loss: 0.8806 - learning_rate: 5.0000e-04
Epoch 34/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8619 - loss: 0.6653 
Epoch 34: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8734 - loss: 0.6288 - val_accuracy: 0.9498 - val_loss: 0.5431 - learning_rate: 5.0000e-04
Epoch 35/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.8753 - loss: 0.6078 
Epoch 35: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.8696 - loss: 0.6210 - val_accuracy: 0.7022 - val_loss: 1.2178 - learning_rate: 5.0000e-04
Epoch 36/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8558 - loss: 0.6564 
Epoch 36: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8393 - loss: 0.7052 - val_accuracy: 0.6364 - val_loss: 1.4804 - learning_rate: 5.0000e-04
Epoch 37/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8598 - loss: 0.6347 
Epoch 37: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8681 - loss: 0.6094 - val_accuracy: 0.8213 - val_loss: 0.6613 - learning_rate: 5.0000e-04
Epoch 38/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8714 - loss: 0.5987 
Epoch 38: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8620 - loss: 0.6273 - val_accuracy: 0.5486 - val_loss: 1.6559 - learning_rate: 5.0000e-04
Epoch 39/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8640 - loss: 0.6356 
Epoch 39: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8650 - loss: 0.6214 - val_accuracy: 0.8025 - val_loss: 0.7487 - learning_rate: 5.0000e-04
Epoch 40/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8626 - loss: 0.6361 
Epoch 40: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8567 - loss: 0.6594 - val_accuracy: 0.8934 - val_loss: 0.5185 - learning_rate: 5.0000e-04
Epoch 41/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 258ms/step - accuracy: 0.8508 - loss: 0.6377 
Epoch 41: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 274ms/step - accuracy: 0.8711 - loss: 0.5937 - val_accuracy: 0.9248 - val_loss: 0.4861 - learning_rate: 5.0000e-04
Epoch 42/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8548 - loss: 0.6358 
Epoch 42: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8635 - loss: 0.6042 - val_accuracy: 0.9342 - val_loss: 0.5403 - learning_rate: 5.0000e-04
Epoch 43/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8729 - loss: 0.5602 
Epoch 43: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8605 - loss: 0.5946 - val_accuracy: 0.8088 - val_loss: 1.2495 - learning_rate: 5.0000e-04
Epoch 44/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8698 - loss: 0.5785 
Epoch 44: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8650 - loss: 0.5962 - val_accuracy: 0.5361 - val_loss: 1.9503 - learning_rate: 5.0000e-04
Epoch 45/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8605 - loss: 0.5721 
Epoch 45: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8658 - loss: 0.5656 - val_accuracy: 0.8683 - val_loss: 0.5468 - learning_rate: 5.0000e-04
Epoch 46/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8771 - loss: 0.5294 
Epoch 46: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8817 - loss: 0.5279 - val_accuracy: 0.7461 - val_loss: 0.7633 - learning_rate: 5.0000e-04
Epoch 47/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8964 - loss: 0.5202 
Epoch 47: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8779 - loss: 0.5484 - val_accuracy: 0.6771 - val_loss: 2.2298 - learning_rate: 5.0000e-04
Epoch 48/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8606 - loss: 0.5967 
Epoch 48: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.

Epoch 48: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8658 - loss: 0.5724 - val_accuracy: 0.8683 - val_loss: 0.5081 - learning_rate: 5.0000e-04
Epoch 49/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8897 - loss: 0.4913 
Epoch 49: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8946 - loss: 0.4785 - val_accuracy: 0.9091 - val_loss: 0.4381 - learning_rate: 2.5000e-04
Epoch 50/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8749 - loss: 0.5083 
Epoch 50: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8848 - loss: 0.4909 - val_accuracy: 0.9091 - val_loss: 0.4102 - learning_rate: 2.5000e-04
Epoch 51/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9149 - loss: 0.4404 
Epoch 51: val_accuracy did not improve from 0.95298
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9075 - loss: 0.4431 - val_accuracy: 0.9091 - val_loss: 0.4097 - learning_rate: 2.5000e-04
Epoch 52/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9192 - loss: 0.4203 
Epoch 52: val_accuracy improved from 0.95298 to 0.97179, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8916 - loss: 0.4376 - val_accuracy: 0.9718 - val_loss: 0.3848 - learning_rate: 2.5000e-04
Epoch 53/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8819 - loss: 0.4412 
Epoch 53: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.8787 - loss: 0.4441 - val_accuracy: 0.9091 - val_loss: 0.3855 - learning_rate: 2.5000e-04
Epoch 54/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.8729 - loss: 0.4724 
Epoch 54: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.8855 - loss: 0.4579 - val_accuracy: 0.9718 - val_loss: 0.3815 - learning_rate: 2.5000e-04
Epoch 55/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8876 - loss: 0.4710 
Epoch 55: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.8931 - loss: 0.4294 - val_accuracy: 0.9342 - val_loss: 0.3806 - learning_rate: 2.5000e-04
Epoch 56/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8984 - loss: 0.4210 
Epoch 56: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8984 - loss: 0.4186 - val_accuracy: 0.8621 - val_loss: 0.4602 - learning_rate: 2.5000e-04
Epoch 57/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8739 - loss: 0.4625 
Epoch 57: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8779 - loss: 0.4479 - val_accuracy: 0.9028 - val_loss: 0.3701 - learning_rate: 2.5000e-04
Epoch 58/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8935 - loss: 0.4092 
Epoch 58: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8954 - loss: 0.4089 - val_accuracy: 0.9091 - val_loss: 0.3654 - learning_rate: 2.5000e-04
Epoch 59/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8962 - loss: 0.4162 
Epoch 59: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8992 - loss: 0.4189 - val_accuracy: 0.9342 - val_loss: 0.3487 - learning_rate: 2.5000e-04
Epoch 60/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.8988 - loss: 0.3999 
Epoch 60: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9030 - loss: 0.4127 - val_accuracy: 0.9342 - val_loss: 0.3560 - learning_rate: 2.5000e-04
Epoch 61/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.8761 - loss: 0.4359 
Epoch 61: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.8848 - loss: 0.4442 - val_accuracy: 0.8777 - val_loss: 0.5533 - learning_rate: 2.5000e-04
Epoch 62/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9016 - loss: 0.4380 
Epoch 62: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8946 - loss: 0.4368 - val_accuracy: 0.9592 - val_loss: 0.3478 - learning_rate: 2.5000e-04
Epoch 63/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9113 - loss: 0.3939 
Epoch 63: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9030 - loss: 0.4028 - val_accuracy: 0.8088 - val_loss: 0.5928 - learning_rate: 2.5000e-04
Epoch 64/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step - accuracy: 0.8850 - loss: 0.4866 
Epoch 64: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 273ms/step - accuracy: 0.8901 - loss: 0.4537 - val_accuracy: 0.9310 - val_loss: 0.3418 - learning_rate: 2.5000e-04
Epoch 65/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.8948 - loss: 0.3961 
Epoch 65: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8923 - loss: 0.4164 - val_accuracy: 0.9091 - val_loss: 0.3466 - learning_rate: 2.5000e-04
Epoch 66/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9117 - loss: 0.4095 
Epoch 66: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.9075 - loss: 0.4024 - val_accuracy: 0.8997 - val_loss: 0.3575 - learning_rate: 2.5000e-04
Epoch 67/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.8926 - loss: 0.3770 
Epoch 67: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.8908 - loss: 0.3930 - val_accuracy: 0.8401 - val_loss: 0.4896 - learning_rate: 2.5000e-04
Epoch 68/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9006 - loss: 0.4030 
Epoch 68: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9075 - loss: 0.3964 - val_accuracy: 0.9122 - val_loss: 0.3258 - learning_rate: 2.5000e-04
Epoch 69/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.8933 - loss: 0.3797 
Epoch 69: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9014 - loss: 0.3775 - val_accuracy: 0.8997 - val_loss: 0.5739 - learning_rate: 2.5000e-04
Epoch 70/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9023 - loss: 0.3685 
Epoch 70: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9067 - loss: 0.3644 - val_accuracy: 0.9185 - val_loss: 0.3302 - learning_rate: 2.5000e-04
Epoch 71/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9028 - loss: 0.3567 
Epoch 71: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9037 - loss: 0.3687 - val_accuracy: 0.9091 - val_loss: 0.3216 - learning_rate: 2.5000e-04
Epoch 72/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9213 - loss: 0.3530 
Epoch 72: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9037 - loss: 0.3716 - val_accuracy: 0.9091 - val_loss: 0.3448 - learning_rate: 2.5000e-04
Epoch 73/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9092 - loss: 0.3716 
Epoch 73: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.9121 - loss: 0.3752 - val_accuracy: 0.9154 - val_loss: 0.3213 - learning_rate: 2.5000e-04
Epoch 74/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9265 - loss: 0.3553 
Epoch 74: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9075 - loss: 0.3897 - val_accuracy: 0.8495 - val_loss: 0.5215 - learning_rate: 2.5000e-04
Epoch 75/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.8664 - loss: 0.4598 
Epoch 75: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.8840 - loss: 0.4224 - val_accuracy: 0.9091 - val_loss: 0.3188 - learning_rate: 2.5000e-04
Epoch 76/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9041 - loss: 0.3774 
Epoch 76: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9022 - loss: 0.4010 - val_accuracy: 0.9091 - val_loss: 0.3363 - learning_rate: 2.5000e-04
Epoch 77/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9005 - loss: 0.3670 
Epoch 77: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9113 - loss: 0.3657 - val_accuracy: 0.8934 - val_loss: 0.4058 - learning_rate: 2.5000e-04
Epoch 78/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9022 - loss: 0.3509 
Epoch 78: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9060 - loss: 0.3608 - val_accuracy: 0.9561 - val_loss: 0.3167 - learning_rate: 2.5000e-04
Epoch 79/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9020 - loss: 0.3905 
Epoch 79: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.8878 - loss: 0.4023 - val_accuracy: 0.9530 - val_loss: 0.3121 - learning_rate: 2.5000e-04
Epoch 80/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9264 - loss: 0.3575 
Epoch 80: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.9151 - loss: 0.3752 - val_accuracy: 0.9091 - val_loss: 0.3229 - learning_rate: 2.5000e-04
Epoch 81/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9084 - loss: 0.3654 
Epoch 81: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.8984 - loss: 0.3611 - val_accuracy: 0.9091 - val_loss: 0.3066 - learning_rate: 2.5000e-04
Epoch 82/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9255 - loss: 0.3406 
Epoch 82: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9113 - loss: 0.3621 - val_accuracy: 0.9122 - val_loss: 0.3115 - learning_rate: 2.5000e-04
Epoch 83/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9075 - loss: 0.3534 
Epoch 83: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9128 - loss: 0.3512 - val_accuracy: 0.8527 - val_loss: 0.3963 - learning_rate: 2.5000e-04
Epoch 84/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9241 - loss: 0.3406 
Epoch 84: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.9181 - loss: 0.3500 - val_accuracy: 0.8683 - val_loss: 0.4266 - learning_rate: 2.5000e-04
Epoch 85/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9326 - loss: 0.3096 
Epoch 85: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9310 - loss: 0.3213 - val_accuracy: 0.9373 - val_loss: 0.3153 - learning_rate: 2.5000e-04
Epoch 86/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9202 - loss: 0.3283 
Epoch 86: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9174 - loss: 0.3363 - val_accuracy: 0.9091 - val_loss: 0.3068 - learning_rate: 2.5000e-04
Epoch 87/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9238 - loss: 0.3358 
Epoch 87: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9227 - loss: 0.3267 - val_accuracy: 0.6803 - val_loss: 1.7869 - learning_rate: 2.5000e-04
Epoch 88/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9144 - loss: 0.3272 
Epoch 88: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9098 - loss: 0.3334 - val_accuracy: 0.9624 - val_loss: 0.2968 - learning_rate: 2.5000e-04
Epoch 89/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 256ms/step - accuracy: 0.9040 - loss: 0.3384 
Epoch 89: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.9067 - loss: 0.3444 - val_accuracy: 0.6270 - val_loss: 1.2069 - learning_rate: 2.5000e-04
Epoch 90/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9104 - loss: 0.3597 
Epoch 90: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.9060 - loss: 0.3616 - val_accuracy: 0.9060 - val_loss: 0.2965 - learning_rate: 2.5000e-04
Epoch 91/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9034 - loss: 0.3981 
Epoch 91: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9105 - loss: 0.3798 - val_accuracy: 0.9060 - val_loss: 0.3807 - learning_rate: 2.5000e-04
Epoch 92/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9102 - loss: 0.3649 
Epoch 92: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9136 - loss: 0.3484 - val_accuracy: 0.9373 - val_loss: 0.2944 - learning_rate: 2.5000e-04
Epoch 93/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9249 - loss: 0.3449 
Epoch 93: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9204 - loss: 0.3432 - val_accuracy: 0.9154 - val_loss: 0.3057 - learning_rate: 2.5000e-04
Epoch 94/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9452 - loss: 0.3098 
Epoch 94: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9356 - loss: 0.3040 - val_accuracy: 0.8245 - val_loss: 0.5351 - learning_rate: 2.5000e-04
Epoch 95/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9242 - loss: 0.3468 
Epoch 95: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 271ms/step - accuracy: 0.9310 - loss: 0.3357 - val_accuracy: 0.9624 - val_loss: 0.2982 - learning_rate: 2.5000e-04
Epoch 96/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9502 - loss: 0.3075 
Epoch 96: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 271ms/step - accuracy: 0.9310 - loss: 0.3285 - val_accuracy: 0.8652 - val_loss: 0.3676 - learning_rate: 2.5000e-04
Epoch 97/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9167 - loss: 0.3338 
Epoch 97: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9212 - loss: 0.3280 - val_accuracy: 0.8809 - val_loss: 0.3815 - learning_rate: 2.5000e-04
Epoch 98/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9463 - loss: 0.3091 
Epoch 98: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9431 - loss: 0.3109 - val_accuracy: 0.9154 - val_loss: 0.2911 - learning_rate: 2.5000e-04
Epoch 99/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 254ms/step - accuracy: 0.9524 - loss: 0.2785 
Epoch 99: val_accuracy did not improve from 0.97179
83/83 ━━━━━━━━━━━━━━━━━━━━ 22s 270ms/step - accuracy: 0.9454 - loss: 0.2903 - val_accuracy: 0.8213 - val_loss: 0.5930 - learning_rate: 2.5000e-04
Epoch 100/100
83/83 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step - accuracy: 0.9263 - loss: 0.3332 
Epoch 100: val_accuracy improved from 0.97179 to 0.99060, saving model to best_planets_model.keras
83/83 ━━━━━━━━━━━━━━━━━━━━ 23s 272ms/step - accuracy: 0.9136 - loss: 0.3474 - val_accuracy: 0.9906 - val_loss: 0.2491 - learning_rate: 2.5000e-04
Restoring model weights from the end of the best epoch: 100.

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
     Jupiter       1.00      1.00      1.00        29
    MakeMake       1.00      1.00      1.00        29
        Mars       1.00      1.00      1.00        29
     Mercury       0.91      1.00      0.95        29
        Moon       1.00      0.90      0.95        29
     Neptune       1.00      1.00      1.00        29
       Pluto       1.00      1.00      1.00        29
      Saturn       1.00      1.00      1.00        29
      Uranus       1.00      1.00      1.00        29
       Venus       1.00      1.00      1.00        29

    accuracy                           0.99       319
   macro avg       0.99      0.99      0.99       319
weighted avg       0.99      0.99      0.99       319

============================================================
SUMMARY
============================================================
  Best Validation Accuracy : 0.9906 (99.06%)
  Best Validation Loss     : 0.2491
  Final Training Accuracy  : 0.9136 (91.36%)
  Total training epochs    : 100
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