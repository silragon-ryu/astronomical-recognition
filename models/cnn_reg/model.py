import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

import glob
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

DATASET_DIR = "../../dataset/Planets_Moons_Data/Planets and Moons"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.2
L2_LAMBDA = 0.001
DROPOUT_RATE = 0.4
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==============================================================================
# 2. DATASET LOADING & DATA AUGMENTATION
# ==============================================================================

print("=" * 60)
print("LOADING DATASET...")
print("=" * 60)

if os.path.exists(DATASET_DIR):
    classes = sorted(os.listdir(DATASET_DIR))
    classes = [c for c in classes if os.path.isdir(os.path.join(DATASET_DIR, c))]
    print(f"\nClasses found ({len(classes)}): {classes}")
    for cls in classes:
        n = len(glob.glob(os.path.join(DATASET_DIR, cls, "*")))
        print(f"  {cls}: {n} images")
else:
    print(f"\n[WARNING] '{DATASET_DIR}' not found!")
    print("Please download the dataset from Kaggle and update the path.")
    print("https://www.kaggle.com/datasets/emirhanai/planets-and-moons-dataset-ai-in-space")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())
print(f"\nTotal number of classes: {NUM_CLASSES}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")


# ==============================================================================
# 3. CNN MODEL (WITH REGULARIZATION)
# ==============================================================================

print("\n" + "=" * 60)
print("BUILDING MODEL...")
print("=" * 60)

def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = keras.Sequential(name="Planets_CNN_Regularized")

    # ---- BLOCK 1 ----
    model.add(layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        input_shape=input_shape
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(DROPOUT_RATE * 0.5))
    model.add(layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(DROPOUT_RATE * 0.5))
    model.add(layers.Conv2D(
        128, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(
        128, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(DROPOUT_RATE))

    # ---- BLOCK 4 ----
    model.add(layers.Conv2D(
        256, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(
        256, (3, 3), padding='same',
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(
        256,
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(L2_LAMBDA)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(DROPOUT_RATE * 0.75))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


model = build_cnn_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==============================================================================
# 4. CALLBACKS (ADDITIONAL REGULARIZATION)
# ==============================================================================

callbacks = [
    # Early Stopping - prevents overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    # Learning Rate Reduction - reduce LR on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    # Save the best model
    ModelCheckpoint(
        'best_planets_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]


# ==============================================================================
# 5. MODEL TRAINING
# ==============================================================================

print("\n" + "=" * 60)
print("TRAINING STARTED...")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)


# ==============================================================================
# 6. RESULTS AND VISUALIZATION
# ==============================================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('CNN + Regularization - Planets & Moons Classification', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training history saved: training_history.png")

print("\nComputing Confusion Matrix...")
val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix - Planets & Moons CNN', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved: confusion_matrix.png")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])
final_train_acc = history.history['accuracy'][-1]

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"  Best Validation Loss     : {best_val_loss:.4f}")
print(f"  Final Training Accuracy  : {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"  Total training epochs    : {len(history.history['loss'])}")
print(f"  Model saved at           : best_planets_model.keras")
print("=" * 60)

print("\nREGULARIZATION TECHNIQUES USED:")
print(f"  1. L2 Weight Regularization (lambda={L2_LAMBDA})")
print(f"  2. Dropout (rate={DROPOUT_RATE})")
print(f"  3. Batch Normalization (after each Conv block)")
print(f"  4. Data Augmentation (rotation, shift, zoom, flip, brightness)")
print(f"  5. Early Stopping (patience=15)")
print(f"  6. Learning Rate Reduction (patience=7, factor=0.5)")
print(f"  7. Global Average Pooling (instead of Flatten - fewer parameters)")

# Outputs and some fixes codded with AI