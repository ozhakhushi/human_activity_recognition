# Human Activity Recognition using PyTorch

A multiclass classification project that uses a feedforward neural network (MLP) built with PyTorch to recognize six human activities from pre-extracted smartphone sensor features.

The project demonstrates how deep learning can classify human movement patterns using numerical sensor data.

## Activities

The model classifies the following six activities:

- LAYING
- SITTING
- STANDING
- WALKING
- WALKING_DOWNSTAIRS
- WALKING_UPSTAIRS

## Model Architecture

The model uses a Multi-Layer Perceptron (MLP) with the following architecture:

`561 input features → Linear(128) → ReLU → Linear(6)`

- **Framework:** PyTorch
- **Loss function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Training epochs:** 100

## Evaluation Metrics

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

| Metric | Result |
|---|---:|
| Test Accuracy | ~95.08% |
| Test Loss | ~0.179 |
| Validation Loss | ~0.214 |

The classification report and confusion matrix provide further insights into the model's performance across individual activity classes.

Results may vary depending on the data split, random initialization, and training environment.

## How Is This Model Useful?

Human Activity Recognition (HAR) can help identify a person's physical activity using data collected by smartphone sensors.

Potential applications include:

- **Fitness tracking:** Recognizing activities such as walking and climbing stairs.
- **Health and wellness applications:** Supporting activity monitoring and lifestyle analysis.
- **Smartphone-based activity monitoring:** Using existing mobile sensor data to classify movement without requiring dedicated wearable hardware.
- **Research and prototyping:** Providing a baseline for experimenting with sensor-based classification and deep learning.

This project is a prototype for activity classification, not a production-ready health monitoring system.

## Dataset

This project uses the **Human Activity Recognition with Smartphones** dataset, which contains pre-extracted numerical features derived from smartphone sensor measurements and an `Activity` target column.

**Dataset source:** [Human Activity Recognition with Smartphones — Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)

The training script expects `train.csv` and `test.csv`. These files are not included in this repository because of their size. Download the dataset from Kaggle and update the file paths in the script to match your local setup.

## How to Run

1. Clone or download this repository.
2. Download the dataset from the Kaggle link above.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Update the dataset paths in `model.py`.
5. Run the training script:

   ```bash
   python model.py
   ```

The script trains the model, evaluates its performance, and saves:

- `model.pth` — trained model weights.
- `har.pkl` — fitted label encoder used to convert activity labels.

## Notes

- The model architecture and input feature count must match the data used during training.
- The saved `model.pth` contains the model's state dictionary, not the model architecture. The `MLP` class definition is required to load it.
- The dataset and generated model artifacts are not included in this repository by default.

## Technologies Used

Python · PyTorch · Pandas · Scikit-learn
