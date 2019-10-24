# deep-emotion-sense
Convolutional neural network to predict language-independent emotion on a dataset of English and French speech. 

This model is trained on a merged dataset of natural speech with data provided by [IEMOCAP](https://sail.usc.edu/iemocap/ "The Interactive Emotional Dyadic Motion Capture (IEMOCAP) Database") for english speech samples and [Recola](https://recola.hepforge.org/ "Recola: Recursive Computation of 1-Loop Amplitudes") for french speech samples.

The implementation is a basis for a [research paper on comparison on different optimization algorithms and activation functions](https://github.com/nymvno/Emotion-Recognizer/blob/master/Language-independent%20Emotion%20Recognition%20from%20Speech.pdf "Language-independent Emotion Recognition from Speech").

## Getting Started
You can view the notebook [here](https://github.com/nymvno/EmotionRecognizer/blob/master/CNN%20SpeechRecognition.ipynb).
### Run the notebook
#### Prerequisites
- Python 3
- Tensorflow

#### Starting the notebook
Simply open a new terminal in the directory and type:
```bash
> jupyter notebook
```

## Built With

* [Tensorflow](https://www.tensorflow.org/)

## Contributors

* **F. Strohm** - [StrohmFn](https://github.com/StrohmFn)
* **C. Tasci** - [StraysWonderland](https://github.com/StraysWonderland)
