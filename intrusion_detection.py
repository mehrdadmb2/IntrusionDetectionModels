import os
import subprocess
import sys

def install_requirements():
    ADD = "I:\\IOT\\HW3\\For Me\\requirements.txt"
    if os.path.exists(ADD):
        print("[+] requirements.txt found.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", ADD])
    else:
        print("[-] requirements.txt not found.")

install_requirements()

import zipfile
import platform
import pandas as pd
import numpy as np
import tensorflow as tf
from colorama import Fore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def clear():
    subprocess.run(['cls' if platform.system() == "Windows" else 'clear'], shell=True)

def extract_and_combine_csv(zip_file_path, extracted_folder, output_file_path):
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    csv_files = [os.path.join(extracted_folder, f) for f in os.listdir(extracted_folder) if f.endswith(".csv")]
    print(csv_files)

    dfs = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df.to_csv(output_file_path, index=False)

    for file in csv_files:
        os.remove(file)

    print(f"[+] Combined CSV saved at: {output_file_path}")
    print("[+] All extracted files have been deleted.")

def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    columns = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
               'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
               'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
               'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
               'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
               'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
               'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
               'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
               'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
               'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
               'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
               'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
               'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
               'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
               'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
               'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
               'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
    data.columns = columns
    return data

def preprocess_data(data):
    data = data.replace([np.inf, -np.inf], np.nan)  
    data = data.dropna()  
    features = data.drop('Label', axis=1) 
    labels = data['Label']  # برچسب‌ها

    le_protocol = LabelEncoder()
    labels = le_protocol.fit_transform(labels)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels

def split_data(features, labels):
    return train_test_split(features, labels, test_size=0.3, random_state=42)

def create_time_series(data, labels, timesteps):
    sequences, sequence_labels = [], []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i+timesteps])
        sequence_labels.append(labels[i+timesteps-1])
    return np.array(sequences), np.array(sequence_labels)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    try:
        auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')
    except ValueError:
        auc_score = 0.0
    print("ROC AUC Score:", auc_score)
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))  
    
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


def create_models(timesteps, input_dim):
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(timesteps, input_dim), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    gru_model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, input_shape=(timesteps, input_dim), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    cnn_lstm_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, input_dim)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    bilstm_model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(timesteps, input_dim)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return lstm_model, gru_model, cnn_lstm_model, bilstm_model

def main():
    zip_file = "I:\\IOT\\HW3\\For Me\\CICIDS 2017.zip"
    extracted_folder = "I:\\IOT\\HW3\\For Me\\CICIDS 2017"
    combined_file = "I:\\IOT\\HW3\\For Me\\CICIDS 2017_Marge\\combined_CICIDS2017.csv"
    extract_and_combine_csv(zip_file, extracted_folder, combined_file)

    data = load_data(combined_file)
    data = data.sample(frac=0.008, random_state=42)  # انتخاب 10 درصد از داده‌ها For My Olddd! PC :)

    features, labels = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = split_data(features, labels)
    
    timesteps = 100
    X_train_ts, y_train_ts = create_time_series(X_train, y_train, timesteps)
    X_test_ts, y_test_ts = create_time_series(X_test, y_test, timesteps)
    
    input_dim = X_train_ts.shape[2]
    lstm_model, gru_model, cnn_lstm_model, bilstm_model = create_models(timesteps, input_dim)
    
    print("Training and evaluating LSTM model...")
    train_and_evaluate_model(lstm_model, X_train_ts, y_train_ts, X_test_ts, y_test_ts)
    
    print("Training and evaluating GRU model...")
    train_and_evaluate_model(gru_model, X_train_ts, y_train_ts, X_test_ts, y_test_ts)
    
    print("Training and evaluating CNN+LSTM model...")
    train_and_evaluate_model(cnn_lstm_model, X_train_ts, y_train_ts, X_test_ts, y_test_ts)
    
    print("Training and evaluating BiLSTM model...")
    train_and_evaluate_model(bilstm_model, X_train_ts, y_train_ts, X_test_ts, y_test_ts)

if __name__ == "__main__":
    main()
