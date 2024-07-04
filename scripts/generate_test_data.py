import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ruta del archivo CSV
test_data_csv_path = '/home/javitrucas/essay_scoring/data/test_prueba.csv'
test_data = pd.read_csv(test_data_csv_path)

# Extraer el texto completo y la puntuación
X_test = test_data['full_text'].values
y_test = test_data['score'].values

# Tokenizar el texto
tokenizer = Tokenizer(num_words=5000)  # Puedes ajustar el número de palabras según sea necesario
tokenizer.fit_on_texts(X_test)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding de las secuencias
X_test_pad = pad_sequences(X_test_seq, padding='post')

# Guardar los datos preprocesados
np.save('/home/javitrucas/essay_scoring/data/X_test_pad.npy', X_test_pad)
np.save('/home/javitrucas/essay_scoring/data/y_test.npy', y_test)

print("Datos de prueba generados y guardados correctamente.")

