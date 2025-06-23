# i40/ej4/utils.py

import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from roboflow import Roboflow
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURACIÓN GLOBAL ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# --- FUNCIONES DE PRE-PROCESAMIENTO Y PIPELINE DE DATOS ---

def download_dataset(api_key):
    """Descarga el dataset de Roboflow si no existe localmente."""
    if not os.path.exists("Hard-Hat-Detector-1"):
        print("🚚 Descargando dataset desde Roboflow Universe...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("wedothings").project("hard-hat-detector-znysj")
        dataset = project.version(1).download("voc")
        location = dataset.location
    else:
        location = os.path.abspath("Hard-Hat-Detector-1")
    print(f"✅ Dataset original listo en: {location}")
    return location

def filter_and_copy_single_object_dataset(original_location, filtered_location="./dataset_filtrado/"):
    """Filtra un dataset para conservar solo imágenes con un único objeto anotado."""
    print(f"🧹 Filtrando el dataset para obtener imágenes de un solo objeto...")
    if os.path.exists(filtered_location):
        shutil.rmtree(filtered_location)
    os.makedirs(filtered_location, exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        original_split_path = os.path.join(original_location, split)
        filtered_split_path = os.path.join(filtered_location, split)
        os.makedirs(filtered_split_path, exist_ok=True)

        if not os.path.exists(original_split_path): continue

        image_files = [f for f in os.listdir(original_split_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        kept_count = 0
        
        for img_name in tqdm(image_files, desc=f"Filtrando {split}", leave=False):
            xml_name = os.path.splitext(img_name)[0] + ".xml"
            xml_path = os.path.join(original_split_path, xml_name)
            
            if not os.path.exists(xml_path): continue
            try:
                tree = ET.parse(xml_path)
                if len(tree.getroot().findall('object')) == 1:
                    shutil.copy(os.path.join(original_split_path, img_name), filtered_split_path)
                    shutil.copy(xml_path, filtered_split_path)
                    kept_count += 1
            except ET.ParseError: pass
        print(f"  - Conjunto '{split}': Se conservaron {kept_count} de {len(image_files)} imágenes.")

    print(f"\n✅ Proceso de filtrado finalizado. El dataset limpio está en '{filtered_location}'")
    return filtered_location

def parse_annotation(xml_file):
    try:
        tree = ET.parse(xml_file.numpy().decode())
        root = tree.getroot()
        obj = root.find('object')
        if obj is not None:
            bndbox = obj.find('bndbox')
            box = [float(bndbox.find(t).text) for t in ['xmin', 'ymin', 'xmax', 'ymax']]
            return np.array(box, dtype=np.float32), 1
        else:
            return np.array([0, 0, 0, 0], dtype=np.float32), 0
    except:
        return np.array([0, 0, 0, 0], dtype=np.float32), -1

def load_and_preprocess(image_path, annotation_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    img_shape_original = tf.shape(image)
    image_resized = tf.image.resize(image, IMAGE_SIZE)
    box, has_object = tf.py_function(func=parse_annotation, inp=[annotation_path], Tout=[tf.float32, tf.int32])
    box.set_shape([4])
    has_object.set_shape([])
    h_orig = tf.cast(img_shape_original[0], tf.float32)
    w_orig = tf.cast(img_shape_original[1], tf.float32)
    h_orig = tf.maximum(h_orig, 1.0)
    w_orig = tf.maximum(w_orig, 1.0)
    box_normalized = tf.stack([box[0]/w_orig, box[1]/h_orig, box[2]/w_orig, box[3]/h_orig])
    return image_resized, (box_normalized, has_object), image, box, annotation_path

def filter_invalid_data(img_resized, labels, img_orig, box_orig, ann_path):
    return labels[1] >= 0

def create_tf_dataset(dataset_location):
    def get_filepaths(data_dir):
        image_paths, annotation_paths = [], []
        data_dir_path = os.path.join(dataset_location, data_dir)
        if not os.path.exists(data_dir_path): return [], []
        for file_name in sorted(os.listdir(data_dir_path)):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                annotation_name = os.path.splitext(file_name)[0] + ".xml"
                annotation_file = os.path.join(data_dir_path, annotation_name)
                if os.path.exists(annotation_file):
                    image_paths.append(os.path.join(data_dir_path, file_name))
                    annotation_paths.append(annotation_file)
        return image_paths, annotation_paths
    
    train_img, train_ann = get_filepaths("train")
    val_img, val_ann = get_filepaths("valid")
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_ann)).map(
        load_and_preprocess, num_parallel_calls=AUTOTUNE
    ).filter(filter_invalid_data)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_ann)).map(
        load_and_preprocess, num_parallel_calls=AUTOTUNE
    ).filter(filter_invalid_data)
    
    train_ds_model = train_ds.map(lambda img_r, lbls, *args: (img_r, lbls))
    val_ds_model = val_ds.map(lambda img_r, lbls, *args: (img_r, lbls))
    
    train_ds_batched = train_ds_model.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds_batched = val_ds_model.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return train_ds_batched, val_ds_batched, val_ds

# --- CLASE DEL MODELO Y FUNCIONES DE ENTRENAMIENTO ---

class SingleClassDetector(keras.Model):
    def __init__(self, backbone_builder):
        super().__init__()
        self.backbone = backbone_builder(input_shape=IMAGE_SIZE + (3,))
        self.bbox_head = keras.layers.Dense(4, activation='sigmoid', name='bbox_head')
        self.objectness_head = keras.layers.Dense(1, activation=None, name='objectness_head')
        self.bbox_loss_tracker = keras.metrics.Mean(name="bbox_loss")
        self.obj_loss_tracker = keras.metrics.Mean(name="objectness_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.obj_accuracy_tracker = keras.metrics.BinaryAccuracy(name="obj_accuracy")
    
    def call(self, images, training=False):
        features = self.backbone(images, training=training)
        return self.bbox_head(features), self.objectness_head(features)
        
    def compile(self, optimizer, bbox_weight=1.0, obj_weight=1.0):
        super().compile()
        self.optimizer = optimizer
        self.bbox_loss_fn = keras.losses.MeanSquaredError()
        self.obj_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight

    @tf.function
    def train_step(self, data):
        images, (gt_boxes, gt_has_object) = data
        with tf.GradientTape() as tape:
            pred_boxes, pred_objectness_logits = self(images, training=True)
            has_object_float = tf.cast(gt_has_object, tf.float32)
            has_object_mask = tf.reshape(has_object_float, [-1, 1])
            bbox_loss = self.bbox_loss_fn(gt_boxes * has_object_mask, pred_boxes * has_object_mask)
            obj_loss = self.obj_loss_fn(tf.reshape(has_object_float, [-1, 1]), pred_objectness_logits)
            total_loss = self.bbox_weight * bbox_loss + self.obj_weight * obj_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.bbox_loss_tracker.update_state(bbox_loss)
        self.obj_loss_tracker.update_state(obj_loss)
        self.total_loss_tracker.update_state(total_loss)
        pred_objectness_probs = tf.sigmoid(pred_objectness_logits)
        self.obj_accuracy_tracker.update_state(tf.reshape(has_object_float, [-1, 1]), pred_objectness_probs)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        images, (gt_boxes, gt_has_object) = data
        pred_boxes, pred_objectness_logits = self(images, training=False)
        has_object_float = tf.cast(gt_has_object, tf.float32)
        has_object_mask = tf.reshape(has_object_float, [-1, 1])
        bbox_loss = self.bbox_loss_fn(gt_boxes * has_object_mask, pred_boxes * has_object_mask)
        obj_loss = self.obj_loss_fn(tf.reshape(has_object_float, [-1, 1]), pred_objectness_logits)
        total_loss = self.bbox_weight * bbox_loss + self.obj_weight * obj_loss
        
        self.bbox_loss_tracker.update_state(bbox_loss)
        self.obj_loss_tracker.update_state(obj_loss)
        self.total_loss_tracker.update_state(total_loss)
        pred_objectness_probs = tf.sigmoid(pred_objectness_logits)
        self.obj_accuracy_tracker.update_state(tf.reshape(has_object_float, [-1, 1]), pred_objectness_probs)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.bbox_loss_tracker, self.obj_loss_tracker, self.total_loss_tracker, self.obj_accuracy_tracker]

def plot_training_history(history):
    history_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Métricas de Entrenamiento y Validación', fontsize=16)
    
    axes[0, 0].plot(history_df['total_loss'], label='Entrenamiento')
    axes[0, 0].plot(history_df['val_total_loss'], label='Validación')
    axes[0, 0].set_title('Pérdida Total')
    axes[0, 0].set(xlabel='Época', ylabel='Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(history_df['bbox_loss'], label='Entrenamiento')
    axes[0, 1].plot(history_df['val_bbox_loss'], label='Validación')
    axes[0, 1].set_title('Pérdida de Bounding Box (MSE)')
    axes[0, 1].set(xlabel='Época'); axes[0, 1].legend()

    axes[1, 0].plot(history_df['objectness_loss'], label='Entrenamiento')
    axes[1, 0].plot(history_df['val_objectness_loss'], label='Validación')
    axes[1, 0].set_title('Pérdida de Presencia de Objeto (Log Loss)')
    axes[1, 0].set(xlabel='Época'); axes[1, 0].legend()

    axes[1, 1].plot(history_df['obj_accuracy'], label='Entrenamiento')
    axes[1, 1].plot(history_df['val_obj_accuracy'], label='Validación')
    axes[1, 1].set_title('Precisión de Presencia de Objeto')
    axes[1, 1].set(xlabel='Época', ylabel='Accuracy'); axes[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


def visualize_predictions(model, dataset, num_samples=5):
    print("\n🖼️  Visualizando predicciones del modelo...")
    print("Verde = Ground Truth | Rojo = Predicción")
    samples_shown = 0
    for img_resized, (box_normalized, has_object), img_orig, box_orig, ann_path in dataset:
        if samples_shown >= num_samples: break
        pred_boxes_normalized, pred_objectness_logits = model.predict(tf.expand_dims(img_resized, axis=0))
        pred_objectness = tf.sigmoid(pred_objectness_logits[0, 0]).numpy()
        pred_has_object = pred_objectness > 0.5
        img_to_show = img_orig.numpy().astype("uint8")
        h, w, _ = img_to_show.shape
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_to_show)
        if has_object == 1:
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = box_orig.numpy()
            rect_gt = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin, linewidth=2, edgecolor='lime', facecolor='none', label='Ground Truth')
            ax.add_patch(rect_gt)
        if pred_has_object:
            pred_xmin = int(pred_boxes_normalized[0][0] * w)
            pred_ymin = int(pred_boxes_normalized[0][1] * h)
            pred_xmax = int(pred_boxes_normalized[0][2] * w)
            pred_ymax = int(pred_boxes_normalized[0][3] * h)
            rect_pred = patches.Rectangle((pred_xmin, pred_ymin), pred_xmax - pred_xmin, pred_ymax - pred_ymin, linewidth=2, linestyle='--', edgecolor='red', facecolor='none', label=f'Predicción (conf: {pred_objectness:.2f})')
            ax.add_patch(rect_pred)
        status = f"GT: {'Objeto presente' if has_object == 1 else 'Sin objeto'} | Pred: {'Objeto detectado' if pred_has_object else 'No detectado'}"
        ax.set_title(status, fontsize=14)
        ax.legend(loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        samples_shown += 1

# --- FUNCIÓN PRINCIPAL ORQUESTADORA ---

def run_lab(api_key, backbone_builder, epochs=15):
    """Función principal que ejecuta todo el laboratorio."""
    original_dataset_path = download_dataset(api_key)
    filtered_dataset_path = filter_and_copy_single_object_dataset(original_dataset_path)
    
    print("\n📦 Creando los datasets de TensorFlow desde la carpeta filtrada...")
    train_ds, val_ds, val_ds_visualize = create_tf_dataset(filtered_dataset_path)
    print("✅ Datasets listos.")
    
    print("\n🔧 Construyendo el detector de clase única...")
    detector = SingleClassDetector(backbone_builder)
    detector.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), bbox_weight=2.0, obj_weight=1.0)
    print("✅ Modelo compilado.")
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    ]
    
    print("\n🚀 Iniciando entrenamiento...")
    history = detector.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    print("\n📈 Graficando historial de entrenamiento...")
    plot_training_history(history)

    print("\nMostrando predicciones")
    visualize_predictions(detector, val_ds_visualize, num_samples=10)
    
    print("\n🎉 ¡Laboratorio finalizado!")
    return detector



