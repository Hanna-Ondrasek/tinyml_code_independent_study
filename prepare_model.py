# -*- coding: utf-8 -*-
"""
Weather Station ML Model - Local Version
Adapted from prepare_model.ipynb for local execution
"""

import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
import subprocess
import platform

from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import activations
from tensorflow.keras import layers

# Constants
BATCH_SIZE = 64
MIN_SNOW_CM = 5  # Above this value, we consider it as snow
NUM_EPOCHS = 20
OUTPUT_DATASET_FILE = "snow_dataset.csv"
TFL_MODEL_FILE = "snow_model.tflite"
TFL_MODEL_HEADER_FILE = "model.h"
TF_MODEL = "snow_forecast"

# ===== DATA ACQUISITION =====


# Method 1: Using wwo-hist package
def acquire_data_method1():
    from wwo_hist import retrieve_hist_data

    frequency = 1
    api_key = "3e2138ceb3974ff7bd8121652251610"
    location_list = ["canazei"]

    hist_df = retrieve_hist_data(
        api_key,
        location_list,
        "01-JAN-2011",
        "31-DEC-2020",
        frequency,
        location_label=False,
        export_csv=False,
        store_df=True,
    )

    t_list = hist_df[0].tempC.astype(float).to_list()
    h_list = hist_df[0].humidity.astype(float).to_list()
    s_list = hist_df[0].totalSnow_cm.astype(float).to_list()

    return t_list, h_list, s_list


# Method 2: Using Historical Weather API
def acquire_data_method2():
    import calendar
    import requests

    api_key = "3e2138ceb3974ff7bd8121652251610"
    city = "canazei"

    t_list = []
    h_list = []
    s_list = []

    for year in range(2011, 2021):
        for month in range(1, 13):
            num_days_month = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month}-01"
            end_date = f"{year}-{month}-{num_days_month}"

            url_base = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
            api_url = f"{url_base}?key={api_key}&q={city}&format=json&date={start_date}&enddate={end_date}&tp=1"

            print(api_url)

            response = requests.get(api_url)

            if response.status_code == 200:
                json_data = response.json()

                for x in json_data["data"]["weather"]:
                    snow_in_cm = float(x["totalSnow_cm"])
                    for y in x["hourly"]:
                        t = float(y["tempC"])
                        h = float(y["humidity"])
                        t_list.append(t)
                        h_list.append(h)
                        s_list.append(snow_in_cm)

    return t_list, h_list, s_list


# ===== DATA VISUALIZATION =====


def visualize_snow_data(t_list, h_list, s_list):
    t_bin_list = []
    h_bin_list = []

    for snow, t, h in zip(s_list, t_list, h_list):
        if snow > MIN_SNOW_CM:
            t_bin_list.append(t)
            h_bin_list.append(h)

    plt.figure(dpi=100)
    plt.scatter(t_bin_list, h_bin_list, c="#000000", label="Snow")
    plt.grid(color="#AAAAAA", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("Snowfall")
    plt.xlabel("Temperature - °C")
    plt.ylabel("Humidity - %")
    plt.show()


# ===== DATASET PREPARATION =====


def gen_label(snow):
    if snow > MIN_SNOW_CM:
        return "Yes"
    else:
        return "No"


def create_dataset(t_list, h_list, s_list):
    labels_list = []
    for snow, temp in zip(s_list, t_list):
        labels_list.append(gen_label(snow))

    csv_header = ["Temp0", "Temp1", "Temp2", "Humi0", "Humi1", "Humi2", "Snow"]

    dataset_df = pd.DataFrame(
        list(
            zip(
                t_list[:-2],
                t_list[1:-1],
                t_list[2:],
                h_list[:-2],
                h_list[1:-1],
                h_list[2:],
                labels_list[2:],
            )
        ),
        columns=csv_header,
    )

    return dataset_df


def balance_dataset(dataset_df):
    df0 = dataset_df[dataset_df["Snow"] == "No"]
    df1 = dataset_df[dataset_df["Snow"] == "Yes"]

    nosnow_samples_old_percent = round(
        (len(df0.index) / len(dataset_df.index)) * 100, 2
    )
    snow_samples_old_percent = round((len(df1.index) / len(dataset_df.index)) * 100, 2)

    print(f"Before balancing: No Snow={len(df0.index)}, Snow={len(df1.index)}")

    # Random subsampling of the majority class
    if len(df1.index) < len(df0.index):
        df0_sub = df0.sample(len(df1.index))
        dataset_df = pd.concat([df0_sub, df1])
    else:
        df1_sub = df1.sample(len(df0.index))
        dataset_df = pd.concat([df1_sub, df0])

    df0 = dataset_df[dataset_df["Snow"] == "No"]
    df1 = dataset_df[dataset_df["Snow"] == "Yes"]

    print(f"After balancing: No Snow={len(df0.index)}, Snow={len(df1.index)}")

    return dataset_df


def scale_features(dataset_df):
    # Get all values
    t_list = dataset_df["Temp0"].tolist()
    h_list = dataset_df["Humi0"].tolist()
    t_list = t_list + dataset_df["Temp2"].tail(2).tolist()
    h_list = h_list + dataset_df["Humi2"].tail(2).tolist()

    # Calculate mean and standard deviation
    t_avg = mean(t_list)
    h_avg = mean(h_list)
    t_std = std(t_list)
    h_std = std(h_list)

    print("\n=== COPY THESE VALUES FOR YOUR ARDUINO CODE ===")
    print(f"Temperature - [MEAN, STD]  {round(t_avg, 5)} {round(t_std, 5)}")
    print(f"Humidity - [MEAN, STD]     {round(h_avg, 5)} {round(h_std, 5)}")
    print("=" * 50 + "\n")

    # Scaling with Z-score function
    def scaling(val, avg, std):
        return (val - avg) / std

    dataset_df["Temp0"] = dataset_df["Temp0"].apply(lambda x: scaling(x, t_avg, t_std))
    dataset_df["Temp1"] = dataset_df["Temp1"].apply(lambda x: scaling(x, t_avg, t_std))
    dataset_df["Temp2"] = dataset_df["Temp2"].apply(lambda x: scaling(x, t_avg, t_std))
    dataset_df["Humi0"] = dataset_df["Humi0"].apply(lambda x: scaling(x, h_avg, h_std))
    dataset_df["Humi1"] = dataset_df["Humi1"].apply(lambda x: scaling(x, h_avg, h_std))
    dataset_df["Humi2"] = dataset_df["Humi2"].apply(lambda x: scaling(x, h_avg, h_std))

    return dataset_df


# ===== MODEL TRAINING =====


def train_model(x_train, y_train, x_validate, y_validate, f_names):
    model = tf.keras.Sequential()
    model.add(layers.Dense(12, activation="relu", input_shape=(len(f_names),)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_validate, y_validate),
    )

    return model, history


def plot_training_history(history):
    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]
    acc_train = history.history["accuracy"]
    acc_val = history.history["val_accuracy"]
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train, "g", label="Training Loss")
    plt.plot(epochs, loss_val, "b", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_train, "g", label="Training Accuracy")
    plt.plot(epochs, acc_val, "b", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ===== MODEL EVALUATION =====


def evaluate_model(model, x_test, y_test):
    y_test_pred = model.predict(x_test)
    y_test_pred = (y_test_pred > 0.5).astype("int32")

    cm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)

    # Visualize confusion matrix
    index_names = ["Actual No Snow", "Actual Snow"]
    column_names = ["Predicted No Snow", "Predicted Snow"]
    df_cm = pd.DataFrame(cm, index=index_names, columns=column_names)

    plt.figure(dpi=150)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.show()

    # Calculate metrics
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    accur = (TP + TN) / (TP + TN + FN + FP)
    precis = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_score = (2 * recall * precis) / (recall + precis) if (recall + precis) > 0 else 0

    print(f"Accuracy:  {round(accur, 3)}")
    print(f"Recall:    {round(recall, 3)}")
    print(f"Precision: {round(precis, 3)}")
    print(f"F-score:   {round(f_score, 3)}")


# ===== MODEL CONVERSION =====


def convert_to_tflite(model, x_test):
    # Save TensorFlow model
    model.export(TF_MODEL)

    # Representative dataset for quantization
    def representative_data_gen():
        data = tf.data.Dataset.from_tensor_slices(x_test)
        for i_value in data.batch(1).take(100):
            i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
            yield [i_value_f32]

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL)
    converter.representative_dataset = tf.lite.RepresentativeDataset(
        representative_data_gen
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()

    with open(TFL_MODEL_FILE, "wb") as f:
        f.write(tflite_model_quant)

    print(f"\nTFLite model saved: {TFL_MODEL_FILE} ({len(tflite_model_quant)} bytes)")

    return tflite_model_quant


def convert_to_c_array(tflite_model_file):
    """Convert TFLite model to C header file"""

    # Try using xxd command (Unix/Mac/Linux)
    try:
        result = subprocess.run(
            ["xxd", "-i", tflite_model_file], capture_output=True, text=True, check=True
        )

        header_content = result.stdout
        # Modify the output
        header_content = header_content.replace("unsigned char", "const unsigned char")
        header_content = header_content.replace("const", "alignas(8) const", 1)

        with open(TFL_MODEL_HEADER_FILE, "w") as f:
            f.write(header_content)

        print(f"C header file created: {TFL_MODEL_HEADER_FILE}")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nxxd not found. Using Python alternative...")
        return convert_to_c_array_python(tflite_model_file)


def convert_to_c_array_python(tflite_model_file):
    """Python alternative to xxd for Windows or systems without xxd"""

    with open(tflite_model_file, "rb") as f:
        model_data = f.read()

    # Create variable name from filename
    var_name = tflite_model_file.replace(".", "_").replace("-", "_")

    # Generate C array
    hex_array = ", ".join([f"0x{b:02x}" for b in model_data])

    # Format with line breaks every 12 bytes
    hex_lines = []
    hex_values = hex_array.split(", ")
    for i in range(0, len(hex_values), 12):
        hex_lines.append("  " + ", ".join(hex_values[i : i + 12]))

    formatted_array = ",\n".join(hex_lines)

    # Create header content
    header_content = f"""alignas(8) const unsigned char {var_name}[] = {{
{formatted_array}
}};
const unsigned int {var_name}_len = {len(model_data)};
"""

    with open(TFL_MODEL_HEADER_FILE, "w") as f:
        f.write(header_content)

    print(f"C header file created: {TFL_MODEL_HEADER_FILE} (using Python method)")
    return True


# ===== MAIN EXECUTION =====


def main():
    print("=" * 60)
    print("Weather Station ML Model Training - Local Version")
    print("=" * 60)

    # Step 1: Acquire data
    print("\n[1/8] Acquiring weather data...")
    try:
        t_list, h_list, s_list = acquire_data_method1()
    except Exception as e:
        print(f"Method 1 failed: {e}")
        print("Trying Method 2...")
        t_list, h_list, s_list = acquire_data_method2()

    # Step 2: Visualize (optional)
    print("\n[2/8] Visualizing snow data...")
    visualize_snow_data(t_list, h_list, s_list)

    # Step 3: Create dataset
    print("\n[3/8] Creating dataset...")
    dataset_df = create_dataset(t_list, h_list, s_list)

    # Step 4: Balance dataset
    print("\n[4/8] Balancing dataset...")
    dataset_df = balance_dataset(dataset_df)

    # Step 5: Scale features
    print("\n[5/8] Scaling features...")
    dataset_df = scale_features(dataset_df)

    # Save dataset
    dataset_df.to_csv(OUTPUT_DATASET_FILE, index=False)
    print(f"Dataset saved to: {OUTPUT_DATASET_FILE}")

    # Step 6: Prepare data for training
    print("\n[6/8] Preparing data for training...")
    f_names = dataset_df.columns.values[0:6]
    l_name = dataset_df.columns.values[6:7]
    x = dataset_df[f_names]
    y = dataset_df[l_name]

    # Encode labels
    labelencoder = LabelEncoder()
    labelencoder.fit(y.Snow)
    y_encoded = labelencoder.transform(y.Snow)

    # Split dataset
    x_train, x_validate_test, y_train, y_validate_test = train_test_split(
        x, y_encoded, test_size=0.20, random_state=1
    )
    x_test, x_validate, y_test, y_validate = train_test_split(
        x_validate_test, y_validate_test, test_size=0.50, random_state=3
    )

    # Step 7: Train model
    print("\n[7/8] Training model...")
    model, history = train_model(x_train, y_train, x_validate, y_validate, f_names)

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    print("\n[8/8] Evaluating model...")
    evaluate_model(model, x_test, y_test)

    # Convert to TFLite
    print("\nConverting to TensorFlow Lite...")
    tflite_model = convert_to_tflite(model, x_test)

    # Convert to C array
    print("\nConverting to C header file...")
    convert_to_c_array(TFL_MODEL_FILE)

    print("\n" + "=" * 60)
    print("COMPLETE! Generated files:")
    print(f"  - {OUTPUT_DATASET_FILE}")
    print(f"  - {TFL_MODEL_FILE}")
    print(f"  - {TFL_MODEL_HEADER_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
