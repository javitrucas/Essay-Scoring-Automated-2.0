-----------*Essay Scoring Project*-----------
This project involves developing and evaluating models for automated essay scoring. It utilizes various machine learning models, including LSTM, GRU, and CNN, to predict essay scores based on text input.

-----------*Project Structure*-----------
/home/javitrucas/essay_scoring
│
├── data
│   ├── learning-agency-lab-automated-essay-scoring-2.zip
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── test_prueba.csv
│   ├── tokenizer.json
│   ├── train.csv
│   ├── X_test_pad.npy
│   ├── X_train.csv
│   ├── X_train_pad.npy
│   ├── X_val.csv
│   ├── X_val_pad.npy
│   ├── y_test.npy
│   ├── y_train.csv
│   ├── y_train.npy
│   └── y_val.npy
│
├── models
│   ├── cnn_model.keras
│   ├── essay_scoring_model.h5
│   ├── gru_model.keras
│   ├── lstm_model.keras
│
├── scripts
│   ├── custom_losses.py
│   ├── evaluate.py
│   ├── generate_test_data.py
│   ├── preprocess.py
│   ├── __pycache__
│   └── train_model.py
│
└── README.txt

-----------*Directory Contents*-----------
data/: Contains the datasets and preprocessed data files.
models/: Contains the trained models.
scripts/: Contains the Python scripts for preprocessing data, training models, and evaluating models.

-----------*Scripts*-----------
preprocess.py
This script handles data preprocessing. It cleans the text, performs synonym replacement for data augmentation, tokenizes the text, and saves the preprocessed data.

custom_losses.py
This script defines custom loss functions used during model training.

train_model.py
This script trains three different types of models (LSTM, GRU, CNN) on the preprocessed data and saves the trained models.

evaluate.py
This script evaluates the trained models using various metrics and visualizes the results.

-----------*Usage*-----------
1. Data Preprocessing
Run the preprocess.py script to clean and preprocess the data:

python scripts/preprocess.py

2. Train Models
Run the train_model.py script to train the models:

python scripts/train_model.py

3. Evaluate Models
Run the evaluate.py script to evaluate the trained models:

python scripts/evaluate.py

Evaluation Metrics
The models are evaluated using the following metrics:

	-Mean Absolute Error (MAE)
	-Mean Squared Error (MSE)
	-R2 Score
	-Variance
	-Max Error
	-Median Absolute Error
	-Precision
	-Recall
	-F1 Score
	-Percentage of Correctly Rated Essays

The evaluation script generates several plots:

	-True vs. Predicted Values
	-Bar charts comparing MAE, MSE, R2 Score, Variance, Max Error, Median Absolute Error, 	Precision, Recall, F1 Score, and Correctly Rated Essays Percentage across different models.

-----------*Dependencies*-----------
Python 3.x
NumPy
pandas
scikit-learn
NLTK
TensorFlow
Matplotlib
Ensure all dependencies are installed before running the scripts.
