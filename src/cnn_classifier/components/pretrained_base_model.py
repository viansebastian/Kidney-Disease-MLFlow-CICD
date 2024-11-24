from pathlib import Path 
import tensorflow as tf
from cnn_classifier.objects.config_object import PretrainedModelConfig


class PretrainedBaseModel: 
    def __init__(self, config: PretrainedModelConfig):
        self.config = config
    
    def get_base_model(self): 
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.params_image_size, 
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    def update_base_model(self):
        self.full_model = self.prepare_full_model(
            base_model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate
        )
        
        self.save_model(path=self.config.used_model_path, model=self.full_model)
        
    @staticmethod
    def prepare_full_model(base_model, classes, learning_rate): 
        for layer in base_model.layers:
            layer.trainable = False
        
        flatten = tf.keras.layers.Flatten()(base_model.output)
        dense1 = tf.keras.layers.Dense(256, activation='relu')(flatten)
        batch_norm = tf.keras.layers.BatchNormalization()(dense1)
        dropout = tf.keras.layers.Dropout(0.1)(batch_norm)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout)
        fc = tf.keras.layers.Dense(units=classes, activation='softmax')(dense2)
            
        full_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=fc
        )
        
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        full_model.summary()
        return full_model
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model): 
        model.save(path)