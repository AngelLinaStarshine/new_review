import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


max_tokens = 10000
sequence_length = 100


vectorize_layer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=sequence_length
)

def get_batch_data():
    
    (train_data, test_data), info = tfds.load('yelp_polarity_reviews',
                                              split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                              with_info=True, as_supervised=True)

 
    train_text = train_data.map(lambda text, label: text)
    vectorize_layer.adapt(train_text)

    
    train_data = train_data.map(lambda text, label: (vectorize_layer(text), label))
    test_data = test_data.map(lambda text, label: (vectorize_layer(text), label))

    train_batches = train_data.shuffle(1000).padded_batch(10)
    test_batches = test_data.padded_batch(10)
    return train_batches, test_batches

def get_model(embedding_dim=16):
    model = keras.Sequential([
        layers.Embedding(max_tokens, embedding_dim, input_length=sequence_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_data(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5, 1))
    plt.show()

def retrieve_embeddings(model, vectorize_layer):
    out_vectors = io.open('embeddings.tsv', 'w', encoding='utf-8')
    out_metadata = io.open('metadata.tsv', 'w', encoding='utf-8')
    weights = model.layers[0].get_weights()[0]  
    vocab = vectorize_layer.get_vocabulary()  

  
    print("Sample embeddings before saving:")
    for i in range(5):
        if i < len(vocab):
            print(f"Word: {vocab[i]}, Embedding: {weights[i]}")


    for idx, word in enumerate(vocab):
        if idx == 0: 
            continue
        vec = weights[idx]
        out_metadata.write(word + "\n")  
        out_vectors.write('\t'.join([str(x) for x in vec]) + '\n')

    out_vectors.close()
    out_metadata.close()


train_batches, test_batches = get_batch_data()
model = get_model()
history = model.fit(train_batches, epochs=10, validation_data=test_batches, validation_steps=20)

retrieve_embeddings(model, vectorize_layer)


from google.colab import files
files.download('embeddings.tsv)
files.download('metadata.tsv')
