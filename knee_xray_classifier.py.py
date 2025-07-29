"""
Project Title: Attention-Augmented EfficientNetV2-L for Knee Osteoporosis Classification
Author: Brian Trinh
Date: 2025-07-29
Description: Implementation of a two-stage convolutional neural network using EfficientNetV2L
to classify knee X-rays into three categories: Normal, Osteopenia, and Osteoporosis.

"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GaussianNoise,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class EfficientNetModel:
    def __init__(self, path: str) -> None:
        """
        Initializes the EfficientNetV2L model.

        Params:
            path (str): The path to the image folder

        """
        # Set the parameters
        self.early_stopping = EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        self.batch_size = 42
        self.image_size = (224, 224)

        # Process the images from the folders
        (
            self.train_df,
            self.valid_df,
            self.test_df,
            self.train_df_full,
            self.valid_df_full,
            self.test_df_full,
        ) = self.load_images(path)

        # Create the generators from the images in the DataFrames.
        (
            self.train_gen,
            self.valid_gen,
            self.test_gen,
            self.train_gen_full,
            self.valid_gen_full,
            self.test_gen_full,
        ) = self.create_generators()

        # Create and save the model
        self.model = self.create_model()
        self.save_model()

    def load_images(self, path: str) -> list[pd.DataFrame]:
        """
        Processes the images at the provided path, returning a list of
        DataFrames for training, validation, and testing.

        Params:
            path (str): The path to the image folder

        Returns:
            list[pd.DataFrame]: The train, validation, and test DataFrames

        """
        categories = ["Normal", "Osteopenia", "Osteoporosis"]

        data = {"train": [], "val": [], "test": []}
        for split in ["train", "val", "test"]:
            image_paths = []
            labels = []
            for category in categories:
                category_path = os.path.join(path, split, category)
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    image_paths.append(image_path)
                    labels.append(category)

            # Create DataFrame for the split
            df = pd.DataFrame({"image_path": image_paths, "label": labels})

            # Encode labels
            label_encoder = LabelEncoder()
            df["category_encoded"] = label_encoder.fit_transform(df["label"])
            df = df[["image_path", "category_encoded"]]

            # Resample the DataFrame
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(
                df[["image_path"]], df["category_encoded"]
            )
            df_resampled = pd.DataFrame(X_resampled, columns=["image_path"])
            df_resampled["category_encoded"] = y_resampled
            df_resampled["category_encoded"] = df_resampled["category_encoded"].astype(
                str
            )
            data[split] = df_resampled

        # Ignore warnings
        warnings.filterwarnings("ignore")

        # Split train data into train, val, test for initial training and testing
        train_df, temp_df = train_test_split(
            data["train"],
            train_size=0.8,
            shuffle=True,
            random_state=42,
            stratify=data["train"]["category_encoded"],
        )
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            shuffle=True,
            random_state=42,
            stratify=temp_df["category_encoded"],
        )
        return [train_df, valid_df, test_df, data["train"], data["val"], data["test"]]

    def create_generators(self) -> list[ImageDataGenerator]:
        """
        Processes the image DataFrames and creates ImageDataGenerators with them. This
        allows the model to be trained on a more diverse dataset.

        Returns:
            list[ImageDataGenerator]: The training, validation, and test ImageDataGenerators

        """
        # Create the generators to modify the images
        tr_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=180,
            horizontal_flip=True,
            vertical_flip=True,
        )
        ts_gen = ImageDataGenerator(rescale=1.0 / 255)

        print("Using only images from the train folder:")

        # Assign the train, test, and validation DataFrames to the generators
        train_gen = tr_gen.flow_from_dataframe(
            self.train_df,
            x_col="image_path",
            y_col="category_encoded",
            target_size=self.image_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batch_size,
        )
        valid_gen = ts_gen.flow_from_dataframe(
            self.valid_df,
            x_col="image_path",
            y_col="category_encoded",
            target_size=self.image_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batch_size,
        )
        test_gen = ts_gen.flow_from_dataframe(
            self.test_df,
            x_col="image_path",
            y_col="category_encoded",
            target_size=self.image_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=False,
            batch_size=self.batch_size,
        )

        print("\nUsing images from the train, val, and test folders:")

        train_gen_full = tr_gen.flow_from_dataframe(
            self.train_df_full,
            x_col="image_path",
            y_col="category_encoded",
            target_size=self.image_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batch_size,
        )
        valid_gen_full = ts_gen.flow_from_dataframe(
            self.valid_df_full,
            x_col="image_path",
            y_col="category_encoded",
            target_size=self.image_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batch_size,
        )
        test_gen_full = ts_gen.flow_from_dataframe(
            self.test_df_full,
            x_col="image_path",
            y_col="category_encoded",
            target_size=self.image_size,
            class_mode="sparse",
            color_mode="rgb",
            shuffle=False,
            batch_size=self.batch_size,
        )
        return [
            train_gen,
            valid_gen,
            test_gen,
            train_gen_full,
            valid_gen_full,
            test_gen_full,
        ]

    def create_model(
        self, input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 3
    ) -> EfficientNetV2L:
        """
        Creates an instance of the EfficientNetV2L model using imagenet weights, with
        additional noise, convolution, pooling, attention, dropout, and dense layers
        added to enhance accuracy and precision.

        Params:
            input_shape (tuple): The shape of the input layer (default: (224, 224, 3))
            num_classes (int): The number of classes to be classified (default: 3)

        Returns:
            EfficientNetV2L: The initialized and modified EfficientNetV2L model

        """
        inputs = Input(shape=input_shape, name="input_layer")

        # Pretrained model
        base_model = EfficientNetV2L(
            weights="imagenet", include_top=False, input_tensor=inputs
        )

        # Freeze the first 75 layers
        base_model.trainable = False
        for layer in base_model.layers[75:]:
            layer.trainable = True

        x = base_model.output

        # Noise layer
        x = GaussianNoise(0.1, name="gaussian_noise")(x)

        # Convolution layers 64, 128, 256
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_64")(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_128")(x)
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="conv_256")(x)

        # Max Pooling layer
        x = MaxPooling2D((2, 2), padding="same", name="pool_2x2")(x)

        # Multi-Head Attention layers
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        x = Reshape((height * width, channels), name="reshape_to_sequence")(x)
        x = MultiHeadAttention(
            num_heads=16, key_dim=channels, name="multi_head_attention"
        )(x, x)
        x = Reshape((height, width, channels), name="reshape_to_spatial")(x)

        # Convolution layers 512 x2
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="conv_512_1")(x)
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="conv_512_2")(x)

        # Average Pooling layer
        x = GlobalAveragePooling2D(name="average_pooling")(x)

        # Dropout and Dense layers
        x = Dropout(0.3, name="dropout_1")(x)
        x = Dense(
            128,
            activation="relu",
            name="dense_128",
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = Dense(
            512,
            activation="relu",
            name="dense_512_1",
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = Dense(
            1024,
            activation="relu",
            name="dense_1024",
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = Dropout(0.3, name="dropout_2")(x)
        x = Dense(
            512,
            activation="relu",
            name="dense_512_2",
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        outputs = Dense(num_classes, activation="softmax", name="output_layer")(x)

        # Ignore warnings that occur when returning an uncompiled model
        tf.get_logger().setLevel("ERROR")
        return Model(inputs=inputs, outputs=outputs, name="EfficientNet_V2L")

    def save_model(self) -> None:
        """
        Saves the model as a .keras file.

        """
        try:
            self.model.save("EfficientNetV2L_with_Attention.keras")
        except:
            print("Unable to save the model!")

    def train(self, on_full: bool = False, second_iteration: bool = False) -> None:
        """
        Compiles and trains the model using the Adam optimizer and
        sparse_categorical_crossentropy loss. The compiled model is then saved,
        overwriting any previous saves. This method can be used for two-stage
        training.

        Params:
            on_full (bool): Whether or not to train the model on the full train dataset (default: False)
            second_iteration (bool): Whether it's the first or the second iteration for two-stage training (default: False)

        """
        # Default learning rate of 1e-4
        learning_rate = 1e-4

        # Make all the layers trainable and reduce the learning rate if it's the second iteration
        if second_iteration:
            self.model.trainable = True
            learning_rate = 5e-6

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # The weights for the different classes
        class_weights = {0: 1.0, 1: 1.2, 2: 1.5}

        if on_full:
            training_data = self.train_gen_full
            valid_data = self.valid_gen_full
        else:
            training_data = self.train_gen
            valid_data = self.valid_gen

        # Fit the model to the training data
        history = self.model.fit(
            training_data,
            validation_data=valid_data,
            callbacks=[self.early_stopping],
            epochs=200,
            verbose=1,
            class_weight=class_weights,
        )

        # Plot the results post training
        self.plot(history)

        # Save the model post training
        self.save_model()

    def train_on_full_data(self, second_iteration: bool = False) -> None:
        """
        Helper method to call `train` on the full train and valid dataset.

        Params:
            second_iteration (bool): Whether it's the first or the second iteration for two-stage training (default: False)

        """
        # Use a much lower learning rate for the second iteration
        self.train(on_full=True, second_iteration=second_iteration)

    def plot(self, history: tf.keras.callbacks.History) -> None:
        """
        Plots the model's accuracy and loss post-training, comparing accuracy/loss
        on the training and validation datasets over the training epochs.

        Params:
            history (tf.keras.callbacks.History): The results from training the model

        """
        # Plot the model's accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        # Plot the model's loss
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show(block=False)

    def evaluate(self, on_test: bool = False) -> None:
        """
        Evaluates and displays the model's performance using the validation and
        test datasets.

        Params:
            on_test (bool): Whether to evaluate on the full test set or the subset (default: False)

        """
        if on_test:
            # Evaluates and displays the PPO loss on the full validation dataset
            y_pred = self.model.predict(self.valid_gen_full)
            y_true = self.valid_gen_full.labels

            # Evaluates on full test dataset and displays confusion matrix
            predictions = self.model.predict(self.test_gen_full)
            test_labels = self.test_gen_full.classes
            predicted_classes = np.argmax(predictions, axis=1)
            names = list(self.test_gen_full.class_indices.keys())
        else:
            # Evaluates and displays the PPO loss on the validation subset
            y_pred = self.model.predict(self.valid_gen)
            y_true = self.valid_gen.labels

            # Evaluates on the test subset and displays the confusion matrix
            predictions = self.model.predict(self.test_gen)
            test_labels = self.test_gen.classes
            predicted_classes = np.argmax(predictions, axis=1)
            names = list(self.test_gen.class_indices.keys())

        ppo_loss = self.calculate_ppo_loss(y_true, y_pred)
        print("\nPPO Loss on Validation Data:", ppo_loss.numpy())
        report = classification_report(
            test_labels,
            predicted_classes,
            target_names=names,
        )
        print(report)
        self.plot_confusion_matrix(confusion_matrix(test_labels, predicted_classes))

    def evaluate_on_test(self) -> None:
        """
        Helper method to call `evaluate` on the full test dataset.

        """
        self.evaluate(on_test=True)

    def calculate_ppo_loss(self, y_true: np.array, y_pred: np.array) -> int:
        """
        Calculates the PPO loss based on the provided true and predicted values.

        Params:
            y_true (np.array): The true values for the dataset
            y_pred(np.array): The predicted values for the dataset

        Returns:
            int: The calculated PPO loss

        """
        epsilon = 0.2
        y_true_one_hot = tf.one_hot(
            tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1]
        )
        selected_probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        old_selected_probs = tf.reduce_sum(
            tf.stop_gradient(y_pred) * y_true_one_hot, axis=-1
        )
        ratio = selected_probs / (old_selected_probs + 1e-10)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
        loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio))
        return loss

    def plot_confusion_matrix(self, conf_matrix: confusion_matrix) -> None:
        """
        Plots the confusion matrix using matplotlib and seaborn.

        Params:
            conf_matrix (confusion_matrix): The confusion matrix to be plotted

        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.test_gen.class_indices.keys()),
            yticklabels=list(self.test_gen.class_indices.keys()),
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


if __name__ == "__main__":
    # NOTE: Change this to the directory containing the images
    path = "Knee_Osteoarthritis_Classification"
    model = EfficientNetModel(path)

    # NOTE: Do not call `evaluate` when calling `train_on_full_data`
    # NOTE: Only use `train` or `train_on_full_data`, not both in the same run
    # NOTE: Need to call `train` or `train_on_full_data` twice if aiming for a two-stage training process

    # Uncomment some combination of the below depending on the requirements based on the notes above
    # model.train()
    # model.train(second_iteration=True)
    model.train_on_full_data()
    model.train_on_full_data(second_iteration=True)
    # model.evaluate()
    model.evaluate_on_test()
