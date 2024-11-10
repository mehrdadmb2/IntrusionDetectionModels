# Intrusion Detection Models 🚀
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmehrdadmb2%2FIntrusionDetectionModels&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=visits&edge_flat=false)
![GitHub license](https://img.shields.io/github/license/mehrdadmb2/IntrusionDetectionModels)
![GitHub stars](https://img.shields.io/github/stars/mehrdadmb2/IntrusionDetectionModels?style=social)
![GitHub forks](https://img.shields.io/github/forks/mehrdadmb2/IntrusionDetectionModels?style=social)
![GitHub issues](https://img.shields.io/github/issues/mehrdadmb2/IntrusionDetectionModels)

### 📍 Introduction
This project is a Python-based solution for **network intrusion detection**. We use a variety of deep learning models to analyze and detect anomalies in network traffic. The models used include:
- LSTM
- GRU
- CNN-LSTM
- BiLSTM

The project includes data extraction, preprocessing, transformation to time-series, and evaluation tools for each model, making it a comprehensive framework for analyzing network data.

---

## 📑 Features
- **Automatic Library Installation**: Installs required Python libraries from `requirements.txt`.
- **Data Extraction and Preprocessing**: Combines multiple CSV files from a ZIP archive, removes invalid data, and scales features.
- **Time-Series Data Preparation**: Converts data into sequences suitable for time-series models.
- **Deep Learning Models**: Supports LSTM, GRU, CNN-LSTM, and BiLSTM for comparative analysis.
- **Model Evaluation**: Provides accuracy, classification reports, ROC AUC, and F1 Score for each model.

---

## 📦 Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mehrdadmb2/IntrusionDetectionModels.git
   cd IntrusionDetectionModels
   ```

2. **Install Required Libraries**
   Make sure you have `requirements.txt` in the `IOT/HW3/For Me/` directory. Run the following command:
   ```bash
   python intrusion_detection.py
   ```

3. **Download and Extract CICIDS 2017 Dataset**
   - Place the dataset ZIP file in `IOT/HW3/For Me/` and ensure it's named `CICIDS 2017.zip`.

---

## 🚀 Running the Project

To execute the project:
```bash
python intrusion_detection.py
```

---

## 🛠 Project Structure
- `intrusion_detection.py`: Main file that automates the following tasks:
  - Installing libraries
  - Extracting and merging CSV data from ZIP file
  - Preprocessing data (e.g., handling NaNs, scaling)
  - Training and evaluating each model

---

## 📊 Model Details
| Model    | Layers                               | Notes                                      |
|----------|--------------------------------------|--------------------------------------------|
| LSTM     | 2 LSTM layers, Dropout, Dense        | Sequential LSTM model                      |
| GRU      | 2 GRU layers, Dropout, Dense         | Sequential GRU model                       |
| CNN-LSTM | Conv1D, MaxPooling1D, 2 LSTM, Dense  | Combines CNN and LSTM layers               |
| BiLSTM   | 2 BiLSTM layers, Dropout, Dense      | Bidirectional LSTM for capturing context   |

---

## 🧩 Helper Functions
- **install_requirements**: Automatically installs libraries from `requirements.txt`.
- **extract_and_combine_csv**: Extracts CSV files from ZIP and combines them into a single file.
- **preprocess_data**: Cleans and scales the dataset.
- **create_time_series**: Transforms data for time-series models.
- **train_and_evaluate_model**: Trains and evaluates a model, providing metrics and plots.

---

## 🔍 Example Usage
After setting up and running the script, you can view the performance metrics and comparison charts for each model. Example outputs include:
- **Accuracy**: Displays overall accuracy of each model.
- **Classification Report**: Provides precision, recall, F1 score, and support.
- **ROC AUC Score**: Measures the area under the ROC curve.
- **F1 Score**: Provides a harmonic mean of precision and recall.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙋 Support
For issues or questions, feel free to open an [issue](https://github.com/mehrdadmb2/IntrusionDetectionModels/issues).

---

### 🌟 Acknowledgments
Thanks to [Mehrdad](https://github.com/mehrdadmb2) for creating this project and sharing it with the community!
```

---

