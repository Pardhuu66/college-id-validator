import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import json
import tf2onnx  # Added for ONNX conversion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths and hyperparameters
DATASET_DIR = 'finalDataset' #Dataset_name/dataset_path # Directory with 'genuine' and 'fake' subfolders
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
FINE_TUNE_EPOCHS = 30
INITIAL_LR = 2e-3
FINE_TUNE_LR = 3e-4
WARMUP_EPOCHS = 5
L2_REG = 1e-4
FOCAL_GAMMA = 2.0
TTA_STEPS = 5

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for imbalanced binary classification."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return focal_loss_fn

def preprocess_image(image):
    """Normalize and standardize image."""
    try:
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0
        mean = tf.reduce_mean(image)
        std = tf.math.reduce_std(image)
        image = (image - mean) / (std + 1e-7)
        return image
    except Exception as e:
        logger.error(f"Preprocess error: {str(e)}")
        return image

def create_data_generators():
    """Create data generators with light augmentation."""
    try:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_image,
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True,
            classes=['fake', 'genuine']  # fake=0, genuine=1
        )

        validation_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            classes=['fake', 'genuine']  # fake=0, genuine=1
        )

        # Compute class weights (for reference, not used with focal loss)
        classes = np.array(train_generator.classes)
        class_counts = np.bincount(classes)
        logger.info(f"Class distribution: Fake (0): {class_counts[0]}, Genuine (1): {class_counts[1]}")
        class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f"Class weights: {class_weight_dict}")

        return train_generator, validation_generator, class_weight_dict
    except Exception as e:
        logger.error(f"Data generator error: {str(e)}")
        raise

def build_model():
    """Build MobileNetV2 model with custom head."""
    try:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0),
                     loss=focal_loss(gamma=FOCAL_GAMMA, alpha=0.25),
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        
        return model
    except Exception as e:
        logger.error(f"Model build error: {str(e)}")
        raise

class MisclassificationLogger(Callback):
    """Log misclassified validation images."""
    def __init__(self, validation_generator):
        super().__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.validation_generator.reset()
            X_val, y_val = next(self.validation_generator)
            y_pred = self.model.predict(X_val, verbose=0)
            y_pred_binary = (y_pred > 0.5).astype(int)
            for i in range(len(y_val)):
                if y_pred_binary[i] != y_val[i]:
                    fname = self.validation_generator.filenames[i % len(self.validation_generator.filenames)]
                    score = y_pred[i][0]
                    true_label = 'genuine' if y_val[i] == 1 else 'fake'
                    pred_label = 'genuine' if y_pred_binary[i] == 1 else 'fake'
                    logger.info(f"Epoch {epoch+1}: Misclassified {fname}: Score={score:.2f}, True={true_label}, Pred={pred_label}")
        except Exception as e:
            logger.error(f"Misclassification logger error: {str(e)}")

def fine_tune_model(model, train_generator, validation_generator):
    """Fine-tune the model with all layers unfrozen."""
    try:
        model.trainable = True
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, clipnorm=1.0),
                     loss=focal_loss(gamma=FOCAL_GAMMA, alpha=0.25),
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        
        history = model.fit(
            train_generator,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=validation_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
                MisclassificationLogger(validation_generator)
            ]
        )
        return history
    except Exception as e:
        logger.error(f"Fine-tune error: {str(e)}")
        raise

def cosine_decay_schedule(epoch, lr):
    """Cosine decay learning rate schedule with warmup."""
    try:
        max_lr = INITIAL_LR if epoch < WARMUP_EPOCHS else FINE_TUNE_LR
        if epoch < WARMUP_EPOCHS:
            return float(max_lr * (epoch + 1) / WARMUP_EPOCHS)
        decay_epochs = EPOCHS - WARMUP_EPOCHS if max_lr == INITIAL_LR else FINE_TUNE_EPOCHS - WARMUP_EPOCHS
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / decay_epochs))
        return float(max_lr * cosine_decay)
    except Exception as e:
        logger.error(f"LR schedule error: {str(e)}")
        return lr

def perform_tta(model, image):
    """Perform test-time augmentation."""
    try:
        images = [image]
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1]
        )
        it = datagen.flow(tf.expand_dims(image, 0), batch_size=1)
        for _ in range(TTA_STEPS - 1):
            aug_image = next(it)[0]
            images.append(aug_image)
        images = tf.stack(images)
        preds = model.predict(images, verbose=0)
        return np.mean(preds, axis=0)[0]
    except Exception as e:
        logger.error(f"TTA error: {str(e)}")
        return model.predict(tf.expand_dims(image, 0), verbose=0)[0][0]

def evaluate_test_cases(model, test_images_dir):
    """Evaluate model on PDF-specified test cases with TTA."""
    try:
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
        test_generator = test_datagen.flow_from_directory(
            test_images_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=1,
            class_mode='binary',
            shuffle=False,
            classes=['fake', 'genuine']
        )
        predictions = []
        for i in range(test_generator.samples):
            image, label = next(test_generator)
            score = perform_tta(model, image[0])
            predictions.append(score)
        results = []
        for i, (score, label, fname) in enumerate(zip(predictions, test_generator.labels, test_generator.filenames)):
            pred_label = 'genuine' if score > 0.85 else 'suspicious' if score > 0.6 else 'fake'
            true_label = 'genuine' if label == 1 else 'fake'
            status = 'approved' if score > 0.85 else 'manual_review' if score > 0.6 else 'rejected'
            reason = 'High confidence' if score > 0.85 else 'Moderate confidence' if score > 0.6 else 'Low confidence or manipulation detected'
            logger.info(f"Test image {fname}: Score={score:.4f}, Predicted={pred_label}, True={true_label}, Status={status}, Reason={reason}")
            results.append({
                'user_id': f'stu_{i}',
                'validation_score': float(score),
                'label': pred_label,
                'status': status,
                'reason': reason,
                'threshold': 0.7
            })
        return results
    except Exception as e:
        logger.error(f"Test case evaluation error: {str(e)}")
        raise

def plot_training_history(history, fine_tune_history):
    """Plot training and validation accuracy/loss."""
    try:
        acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
        val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
        loss = history.history['loss'] + fine_tune_history.history['loss']
        val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
        
        epochs_range = range(len(acc))
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.savefig('training_history.png')
        logger.info("Training history plot saved as training_history.png")
    except Exception as e:
        logger.error(f"Plotting error: {str(e)}")

def main():
    try:
        # Create data generators
        logger.info("Creating data generators...")
        train_generator, validation_generator, class_weight_dict = create_data_generators()
        
        # Build model
        logger.info("Building MobileNetV2 model...")
        model = build_model()
        
        # Initial training
        logger.info("Starting initial training...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                ModelCheckpoint('initial_model.keras', save_best_only=True, monitor='val_accuracy'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
                tf.keras.callbacks.LearningRateScheduler(cosine_decay_schedule),
                MisclassificationLogger(validation_generator)
            ]
        )
        
        # Fine-tuning
        logger.info("Starting fine-tuning...")
        fine_tune_history = fine_tune_model(model, train_generator, validation_generator)
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history, fine_tune_history)
        
        # Evaluate model
        logger.info("Evaluating model...")
        val_loss, val_accuracy, val_auc = model.evaluate(validation_generator)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")
        
        # Evaluate test cases
        logger.info("Evaluating PDF test cases...")
        test_results = evaluate_test_cases(model, DATASET_DIR)  # Use separate test directory if available
        
        # Save test results as JSON
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)
        logger.info("Test results saved as test_results.json")
        
        # Convert and save model as ONNX
        logger.info("Converting model to ONNX...")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=[tf.TensorSpec([None, IMG_HEIGHT, IMG_WIDTH, 3], tf.float32, name='input_1')]
        )
        with open('college_id_classifier.onnx', 'wb') as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Model saved as college_id_classifier.onnx")

    except Exception as e:
        logger.error(f"Main function error: {str(e)}")
        raise

if __name__ == "__main__":
    main()