import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from custom_losses import custom_loss  # Import the custom_loss function

# Load the validation data
X_val_pad = np.load('/home/javitrucas/essay_scoring/data/X_val_pad.npy')
y_val = np.load('/home/javitrucas/essay_scoring/data/y_val.npy')
essay_texts = np.load('/home/javitrucas/essay_scoring/data/essay_texts.npy', allow_pickle=True)

# Load the saved models
lstm_model = load_model('/home/javitrucas/essay_scoring/models/lstm_model.keras', custom_objects={'custom_loss': custom_loss})
gru_model = load_model('/home/javitrucas/essay_scoring/models/gru_model.keras', custom_objects={'custom_loss': custom_loss})
cnn_model = load_model('/home/javitrucas/essay_scoring/models/cnn_model.keras', custom_objects={'custom_loss': custom_loss})
hybrid_model = load_model('/home/javitrucas/essay_scoring/models/hybrid_model.keras', custom_objects={'custom_loss': custom_loss})  # Load the hybrid model

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

def analyze_essays(essays, y_val, predictions, model_name, analysis_type):
    analysis_results = []
    for idx, (essay, true_value, predicted_value) in enumerate(essays):
        doc = nlp(essay)
        analysis_results.append({
            'essay': essay,
            'num_sentences': len(list(doc.sents)),
            'num_words': len(doc),
            'num_unique_words': len(set([token.text for token in doc if token.is_alpha])),
            'num_errors': len([token for token in doc if token.is_alpha and token.tag_ == 'ERR']),
            'model_name': model_name,
            'analysis_type': analysis_type,
            'true_score': true_value,
            'predicted_score': predicted_value,
            'error': abs(predicted_value - true_value)
        })
    return pd.DataFrame(analysis_results)

def evaluate_model(model, X_val, y_val, essay_texts, model_name):
    predictions = model.predict(X_val).flatten()
    rounded_predictions = np.round(predictions).astype(int)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    variance = np.var(predictions)
    max_error = np.max(np.abs(predictions - y_val))
    median_absolute_error = np.median(np.abs(predictions - y_val))
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, rounded_predictions, average='weighted')
    cm = confusion_matrix(y_val, rounded_predictions)
    
    # Correctly rated essays: those within ±1 of the true score
    correctly_rated = np.sum(np.abs(predictions - y_val) <= 1)
    correct_percentage = (correctly_rated / len(y_val)) * 100

    # Exact match precision
    exact_matches = np.sum(rounded_predictions == y_val)
    exact_match_precision = (exact_matches / len(y_val)) * 100

    print(f'--- {model_name} ---')
    print(f'MAE (Mean Absolute Error): {mae}')
    print(f'MSE (Mean Squared Error): {mse}')
    print(f'R2 Score: {r2}')
    print(f'Variance: {variance}')
    print(f'Max Error: {max_error}')
    print(f'Median Absolute Error: {median_absolute_error}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Correctly Rated Essays: {correct_percentage:.2f}%')
    print(f'Exact Match Precision: {exact_match_precision:.2f}%')

    plt.figure(figsize=(10, 6))
    plt.plot(y_val, label='True Values')
    plt.plot(predictions, label='Predicted Values', alpha=0.7)
    plt.title(f'{model_name} - True vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.show()

    residuals = y_val - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='r', linestyles='dashed')
    plt.title(f'{model_name} - Residuals Plot')
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

    for i in range(min(10, len(y_val))):
        print(f'True Value: {y_val[i]}, Predicted Value: {predictions[i]}')

    best_predictions, worst_predictions = get_representative_examples(essay_texts, y_val, predictions, top_n=5)
    
    best_analysis = analyze_essays(best_predictions, y_val, predictions, model_name, 'best')
    worst_analysis = analyze_essays(worst_predictions, y_val, predictions, model_name, 'worst')

    qualitative_analysis = pd.concat([best_analysis, worst_analysis])
    display_qualitative_analysis(qualitative_analysis)

    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'variance': variance,
        'max_error': max_error,
        'median_absolute_error': median_absolute_error,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct_percentage': correct_percentage,
        'exact_match_precision': exact_match_precision,  # Add exact match precision
        'qualitative_analysis': qualitative_analysis,
        'predictions': rounded_predictions  # Add predictions to the return dictionary
    }

def get_representative_examples(essay_texts, y_val, predictions, top_n=5):
    # Calculate the absolute error between the predictions and the true values
    error = np.abs(predictions - y_val)
    
    # Ensure top_n is an integer
    top_n = int(top_n)
    
    # Sort indices based on the error
    sorted_indices = np.argsort(error)
    
    # Get indices for the best and worst predictions
    best_indices = sorted_indices[:top_n]  # Top N indices with the smallest error
    worst_indices = sorted_indices[-top_n:]  # Top N indices with the largest error
    
    # Get the corresponding essays, true values, and predictions for the best and worst examples
    best_predictions = [(essay_texts[i], y_val[i], predictions[i]) for i in best_indices]
    worst_predictions = [(essay_texts[i], y_val[i], predictions[i]) for i in worst_indices]
    
    return best_predictions, worst_predictions

def display_qualitative_analysis(analysis_df):
    for idx, row in analysis_df.iterrows():
        print(f"Model: {row['model_name']}, Type: {row['analysis_type']}")
        print(f"True Score: {row['true_score']}, Predicted Score: {row['predicted_score']}, Error: {row['error']}")
        print(f"Number of Sentences: {row['num_sentences']}, Number of Words: {row['num_words']}, Unique Words: {row['num_unique_words']}, Errors: {row['num_errors']}")
        print(f"Essay Text: {row['essay'][:500]}...")  # Mostrar solo los primeros 500 caracteres para brevedad
        print("="*80)

# Evaluate each model and store the metrics
metrics = {}
metrics['LSTM'] = evaluate_model(lstm_model, X_val_pad, y_val, essay_texts, 'LSTM Model')
metrics['GRU'] = evaluate_model(gru_model, X_val_pad, y_val, essay_texts, 'GRU Model')
metrics['CNN'] = evaluate_model(cnn_model, X_val_pad, y_val, essay_texts, 'CNN Model')
metrics['Hybrid'] = evaluate_model(hybrid_model, X_val_pad, y_val, essay_texts, 'Hybrid Model')  # Evaluate hybrid model

# Extract the metrics
models = list(metrics.keys())
maes = [metrics[model]['mae'] for model in models]
mses = [metrics[model]['mse'] for model in models]
r2_scores = [metrics[model]['r2'] for model in models]
variances = [metrics[model]['variance'] for model in models]
max_errors = [metrics[model]['max_error'] for model in models]
median_absolute_errors = [metrics[model]['median_absolute_error'] for model in models]
precisions = [metrics[model]['precision'] for model in models]
recalls = [metrics[model]['recall'] for model in models]
f1_scores = [metrics[model]['f1'] for model in models]
correct_percentages = [metrics[model]['correct_percentage'] for model in models]
exact_match_precisions = [metrics[model]['exact_match_precision'] for model in models]  # Add exact match precisions

# Plotting Precision, Recall, F1-Score
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precisions, width, label='Precision')
rects2 = ax.bar(x, recalls, width, label='Recall')
rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score')

ax.set_xlabel('Model')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, F1-Score by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()
plt.show()

# Plotting exact match precisions
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(models, exact_match_precisions, width, label='Exact Match Precision')

ax.set_xlabel('Model')
ax.set_ylabel('Exact Match Precision (%)')
ax.set_title('Exact Match Precision by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()
plt.show()

# Plotting pie charts for true scores and predicted scores
def plot_pie_chart(scores, title):
    unique, counts = np.unique(scores, return_counts=True)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.show()

# Plot pie chart for true scores
plot_pie_chart(y_val, 'Distribution of True Scores')

# Plot pie charts for predicted scores of each model
for model in models:
    predictions = metrics[model]['predictions']
    plot_pie_chart(predictions, f'Distribution of Predicted Scores - {model} Model')
