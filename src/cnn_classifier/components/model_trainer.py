from pathlib import Path
from cnn_classifier.objects.config_object import TrainingConfig
import tensorflow as tf


class Training: 
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.used_model_path
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
    
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.2
        )
        
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = 'nearest'
        )
        
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.validation_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            shuffle=False,
            subset='validation',
            class_mode='categorical',
            **dataflow_kwargs
        )
        print('encode (validation): ', self.validation_generator.class_indices)
        
        if self.config.params_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                horizontal_flip=True,
                zoom_range=0.1,
                **datagenerator_kwargs
                )
        else: 
            train_datagen = valid_datagen
        
        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            subset='training',
            class_mode='categorical',
            **dataflow_kwargs
        ) 
        print('encode (train): ', self.train_generator.class_indices)
        
    def train(self):
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.trained_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        callback = [checkpoint]
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epoch,
            validation_data=self.validation_generator,
            verbose=1,
            callbacks=callback
        )
        
        _, test_acc = self.model.evaluate(self.validation_generator)
        print(f"Test Accuracy: {test_acc:.2f}")
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model): 
        model.save(path)