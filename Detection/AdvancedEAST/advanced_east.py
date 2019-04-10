import os
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adam

from Detection.AdvancedEAST import cfg
from Detection.AdvancedEAST.network import East
from Detection.AdvancedEAST.losses import quad_loss
from Detection.AdvancedEAST.data_generator import gen

east = East()
east_network = east.east_network()
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,
                                                    # clipvalue=cfg.clipvalue,
                                                    decay=cfg.decay))
if cfg.load_weights:
    east_network.load_weights('D:\py_projects\VTD\model\east_model\saved_model\east_model_weights_3T736.h5')
TB=TensorBoard(log_dir='logs')
RL=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0, verbose=1)
ES=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min', verbose=1)
CK=ModelCheckpoint(filepath=cfg.model_weights_path,save_best_only=True,save_weights_only=True,verbose=3)
his=east_network.fit_generator(generator=gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=100,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[CK,TB,RL,ES]
                           )
print(his)
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
