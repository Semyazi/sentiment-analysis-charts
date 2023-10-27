import os
import shutil
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import text_util
import matplotlib.pyplot as plt
import ntpath
import sys

reloaded_model = tf.saved_model.load('./bert')

# Credit: https://youtu.be/3y9GESSZmS0
def exponential_moving_average(x, window=300):
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()

    smoothed = np.convolve(x, weights)[:len(x)]
    smoothed[:window] = smoothed[window]

    return smoothed

def run_model(filepath):
    samples = text_util.preprocess(filepath)
    size = len(samples)
    print(f"Analyzing {size} sentences...")
    if size > 3000:
        print("This may take a while.")

    results = tf.sigmoid(reloaded_model(tf.constant(samples)))

    # Shift in the range of [-1, 1]
    y = tf.reshape(results, (size,))
    y = y.numpy()
    y *= 2
    y -= 1

    rft = np.fft.rfft(y)
    rft[7:] = 0
    y_fourier = np.fft.irfft(rft)

    y_ma = exponential_moving_average(y, int(size*0.1))

    min_size = min(len(y_fourier), len(y_ma))
    y_fourier = y_fourier[:min_size]
    y_ma = y_ma[:min_size]

    x = np.arange(min_size)

    plt.plot(x, y_fourier, label='Fourier Transform')
    plt.plot(x, y_ma, label='Exponential Moving Average')
    plt.xlabel('Sentence')
    plt.ylabel('Sentiment (-1 to 1)')
    plt.title(ntpath.basename(filepath))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_model(sys.argv[1])