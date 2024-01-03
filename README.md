# Predicting emotional response to urban sounds  

### What is this Project?
This project is a group project for The London Interdisciplinary School. It focuses on the analysis of urban sounds and their impact on human emotions, specifically in terms of arousal and valence. Utilising audio processing and machine learning techniques, we've trained linear regression models to predict emotional responses based on the characteristics of urban soundscapes.  We tested the model not only on the original data, but on our own collected real-world audio. The reuslts were then compared to an NLP analysis of the participants reactions to the audio. These reactions were recorded in real-time.

### Where are the Results?
The results of the NLP analysis is complied within Research_results.csv, it includes:
- NLP valence/arousal rating based off interviews.
- ML valence/arousal predictions based off linear regression model.

The results of the machine leatning analysis are compiled in the results/final_results directory. Here, you will find:

- 2D Plots: Visual representations of the models' predictions against actual data. These plots show the relationship between the feature extraction of the audio (MFCCs) and the emotional response ratings.
- Statistics: Quantitative analysis of the model's performance, including metrics such as mean squared error and the coefficient of determination (R² score).
- Evaluation: Detailed assessment of both the Valence/MFCC and Arousal/MFCC models. 


#### Data credits to Emo Soundscapes under creative commons - J. Fan, M. Thorogood, and P. Pasquier, “Emo-Soundscapes- A Dataset for Soundscape Emotion Recognition,” Proceedings of the International Conference on Affective Computing and Intelligent Interaction, 2017.
