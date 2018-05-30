#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/30
"""
import os

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from math import ceil
from matplotlib import pyplot as plt

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras.optimizers import Adam
from keras.layers import K

from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.object_detection_2d_data_generator import DataGenerator
from models.keras_ssd7 import build_model
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from workspace.work_dir import DATASET_DIR, DATA_DIR


class SSD7Trainer(object):
    """
    SSD7的训练器
    """

    def __init__(self):
        self.dataset_dir = os.path.join(DATASET_DIR, 'udacity_driving_datasets')  # 数据集
        self.data_dir = DATA_DIR  # 数据

        # 图片高度和宽度, 通道
        self.img_height = 300
        self.img_width = 480
        self.img_channels = 3
        self.scales = [0.08, 0.16, 0.32, 0.64, 0.96]  # 卷积层的缩放尺寸
        self.normalize_coords = True  # 正则化坐标系
        self.aspect_ratios = [0.5, 1.0, 2.0]  # anchor boxes 放大比例
        self.variances = [1.0, 1.0, 1.0, 1.0]  # 4个坐标轴的缩放比例
        self.batch_size = 16  # 批次数
        self.n_classes = 5  # 正例的类别数

    def load_dataset_from_data(self):
        """
        从原始图片和CSV, 加载数据集, 同时写入HDF5文件
        """
        train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

        images_dir = self.dataset_dir  # 图片文件夹

        # Ground truth, 真值, 图片, 物体框(4个点), 物体类别
        train_labels_filename = os.path.join(self.dataset_dir, 'labels_train.csv')
        val_labels_filename = os.path.join(self.dataset_dir, 'labels_val.csv')

        train_dataset.parse_csv(images_dir=images_dir,
                                labels_filename=train_labels_filename,
                                input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                                include_classes='all')  # 解析csv文件

        val_dataset.parse_csv(images_dir=images_dir,
                              labels_filename=val_labels_filename,
                              input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                              include_classes='all')  # 解析csv文件

        # HDF5 文件
        train_dataset_hdf5 = os.path.join(self.data_dir, 'dataset_udacity_traffic_train.h5')
        val_dataset_hdf5 = os.path.join(self.data_dir, 'dataset_udacity_traffic_val.h5')

        train_dataset.create_hdf5_dataset(file_path=train_dataset_hdf5,
                                          resize=False,
                                          variable_image_size=True,
                                          verbose=True)

        val_dataset.create_hdf5_dataset(file_path=val_dataset_hdf5,
                                        resize=False,
                                        variable_image_size=True,
                                        verbose=True)

        # Get the number of samples in the training and validations datasets.
        train_dataset_size = train_dataset.get_dataset_size()
        val_dataset_size = val_dataset.get_dataset_size()

        print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
        print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

        return train_dataset, train_dataset_size, val_dataset, val_dataset_size

    def load_dataset_from_hdf5(self):
        """
        从HDF5, 加载数据集, 速度较快
        """
        # HDF5 文件
        train_dataset_hdf5 = os.path.join(self.data_dir, 'dataset_udacity_traffic_train.h5')
        val_dataset_hdf5 = os.path.join(self.data_dir, 'dataset_udacity_traffic_val.h5')

        train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=train_dataset_hdf5)
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_dataset_hdf5)

        # Get the number of samples in the training and validations datasets.
        train_dataset_size = train_dataset.get_dataset_size()
        val_dataset_size = val_dataset.get_dataset_size()

        print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
        print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

        return train_dataset, train_dataset_size, val_dataset, val_dataset_size

    def build_ssd7_model(self):
        intensity_mean = 127.5  # 像素减去
        intensity_range = 127.5  # 像素除以

        K.clear_session()  # Clear previous models from memory.

        model = build_model(image_size=(self.img_height, self.img_width, self.img_channels),
                            n_classes=self.n_classes,
                            mode='training',
                            l2_regularization=0.0005,
                            scales=self.scales,
                            aspect_ratios_global=self.aspect_ratios,
                            aspect_ratios_per_layer=None,
                            variances=self.variances,
                            normalize_coords=self.normalize_coords,
                            subtract_mean=intensity_mean,
                            divide_by_stddev=intensity_range)

        # model.load_weights('./ssd7_weights.h5', by_name=True)  # 迁移学习，微调参数

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return model

    def create_generator(self, model, train_dataset, val_dataset):
        # 数据扩充链
        data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                                    random_contrast=(0.5, 1.8, 0.5),
                                                                    random_saturation=(0.5, 1.8, 0.5),
                                                                    random_hue=(18, 0.5),
                                                                    random_flip=0.5,
                                                                    random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
                                                                    random_scale=(0.5, 2.0, 0.5),
                                                                    n_trials_max=3,
                                                                    clip_boxes=True,
                                                                    overlap_criterion='area',
                                                                    bounds_box_filter=(0.3, 1.0),
                                                                    bounds_validator=(0.5, 1.0),
                                                                    n_boxes_min=1,
                                                                    background=(0, 0, 0))

        # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

        # The encoder constructor needs the spatial dimensions of the model's
        # predictor layers to create the anchor boxes.
        predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                           model.get_layer('classes5').output_shape[1:3],
                           model.get_layer('classes6').output_shape[1:3],
                           model.get_layer('classes7').output_shape[1:3]]

        ssd_input_encoder = SSDInputEncoder(img_height=self.img_height,
                                            img_width=self.img_width,
                                            n_classes=self.n_classes,
                                            predictor_sizes=predictor_sizes,
                                            scales=self.scales,
                                            aspect_ratios_global=self.aspect_ratios,
                                            variances=self.variances,
                                            matching_type='multi',
                                            pos_iou_threshold=0.5,
                                            neg_iou_limit=0.3,
                                            normalize_coords=self.normalize_coords)

        # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

        train_generator = train_dataset.generate(batch_size=self.batch_size,
                                                 shuffle=True,
                                                 transformations=[data_augmentation_chain],
                                                 label_encoder=ssd_input_encoder,
                                                 returns={'processed_images',
                                                          'encoded_labels'},
                                                 keep_images_without_gt=False)

        val_generator = val_dataset.generate(batch_size=self.batch_size,
                                             shuffle=False,
                                             transformations=[],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

        return train_generator, val_generator

    def train_model(self):
        model = self.build_ssd7_model()
        train_dataset, _, val_dataset, val_dataset_size = self.load_dataset_from_hdf5()
        train_generator, val_generator = self.create_generator(model, train_dataset, val_dataset)

        cp_file = os.path.join(self.data_dir, 'ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5')
        log_file = os.path.join(self.data_dir, 'ssd7_training_log.csv')

        model_checkpoint = ModelCheckpoint(filepath=cp_file,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           period=1)

        csv_logger = CSVLogger(filename=log_file,
                               separator=',',
                               append=True)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0,
                                       patience=10,
                                       verbose=1)

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2,
                                                 patience=8,
                                                 verbose=1,
                                                 epsilon=0.001,
                                                 cooldown=0,
                                                 min_lr=0.00001)

        callbacks = [model_checkpoint,
                     csv_logger,
                     early_stopping,
                     reduce_learning_rate]

        initial_epoch = 0
        final_epoch = 20
        steps_per_epoch = 1000

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=final_epoch,
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=ceil(val_dataset_size / self.batch_size),
                                      initial_epoch=initial_epoch)

        plt.figure(figsize=(20, 12))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(loc='upper right', prop={'size': 24})


if __name__ == '__main__':
    st = SSD7Trainer()
    st.train_model()
