from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def get_model(cfg):
    # TODOÖ add 
    base_model = ResNet50(include_top=False, weights=cfg.weights, input_shape=cfg.input_shape+(cfg.n_channels,), )
    if cfg.freeze_backbone:
        for layer in base_model.layers:
            layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    x = Dense(cfg.model.linear_1, activation='relu')(x)
    x = Dropout(cfg.model.dropout_1)(x)
    output_tensor = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output_tensor)
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', # sparse_categorical_crossentropy
                  metrics=['accuracy'])
    
    return model