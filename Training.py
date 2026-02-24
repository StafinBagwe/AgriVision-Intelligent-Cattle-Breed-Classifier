import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


img_size = (224,224)  
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
class_names = ["Ayrshire", "Brown Swiss","Dangi","Hallikar","Gir",
               "Jaffarabadi buffalo","Holstein friesian crossbreed",
               "Murrah","Alambadi"]

# -----------------------
# Load datasets
# -----------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int",
    class_names=class_names,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "validate",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

# -----------------------
# Data Augmentation
# -----------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="data_augmentation")

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = applications.resnet_v2.preprocess_input(image)
    return image, label

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(1000).prefetch(AUTOTUNE)

val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# -----------------------
# Class Weights
# -----------------------
y_train = np.concatenate([y.numpy() for _, y in tf.keras.preprocessing.image_dataset_from_directory(
    "train", image_size=img_size, batch_size=batch_size, label_mode="int", class_names=class_names, shuffle=False
)], axis=0)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# -----------------------
# Base Model
# -----------------------
base_model = applications.ResNet50V2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Phase 1: feature extraction

# -----------------------
# Model Head (improved)
# -----------------------
inputs = keras.Input(shape=img_size + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)

# -----------------------
# Compile
# -----------------------
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# Callbacks
# -----------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("high_res_best_model.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
]

# -----------------------
# Phase 1: Feature Extraction
# -----------------------
print("Starting feature extraction...")
history_feature_extraction = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# -----------------------
# Phase 2: Fine-Tuning
# -----------------------
print("\nUnfreezing top layers for fine-tuning...")

base_model.trainable = True
for layer in base_model.layers[:-100]:  # Freeze bottom layers
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine_tuning = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    initial_epoch=history_feature_extraction.epoch[-1],
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# -----------------------
# Evaluation
# -----------------------
print("\nLoading best model for evaluation...")
model = keras.models.load_model("high_res_best_model.h5")

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

y_true = np.concatenate([y.numpy() for _, y in tf.keras.preprocessing.image_dataset_from_directory(
    "test", image_size=img_size, batch_size=batch_size, label_mode="int", class_names=class_names, shuffle=False
)], axis=0)

y_pred = np.argmax(model.predict(test_ds), axis=1)

print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(
    y_true, y_pred, target_names=class_names))
