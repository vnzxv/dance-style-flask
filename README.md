# dance-style-flask
Dance Style Predictor
A model that predicts the most suitable dance style for a song based on its audio features. Given an audio input, the model outputs a predicted dance style along with the confidence probabilities for each of the 6 supported dance style categories.

Overview
Different songs carry different musical characteristics — tempo, energy, and more. This project leverages those audio features to classify which dance style a song most closely aligns with, based on patterns learned from labeled training data.

Output
For each song, the model returns:

Predicted Dance Style — the category with the highest confidence score
Probability Distribution — confidence scores for all 6 dance styles

Interpreting the Results
What the probability MEANS
The predicted dance style (highest probability) indicates the category whose audio feature profile most closely matches the input song, based on patterns learned during training.

Author
Made by Venisse Velandres — feel free to open issues or contribute!
