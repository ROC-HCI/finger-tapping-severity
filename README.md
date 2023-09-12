# finger-tapping-severity

**Data**:
1. fingertapping_features_severity_diagnosis_June13_2023 contains all the extracted features from the study participants.
2. severity_dataset_dropped_correlated_columns drops some features which are highly correlated to one of the remaining features.

**Column Descriptions:**
* Rating1, Rating3, and Rating4 columns are ratings from three experts. 
* Rating2 and Rating5 columns are the ratings from two non-experts. 
* Similarly, Comment1, Comment3, and Comment4 columns are comments from the experts. 
* CheckDifficult1, CheckDifficult3, and CheckDifficult4 columns identify videos that are difficult to rate and thus marked as low-quality.
* Rating is the ground-truth severity rating, based on expert consensus.

**Code**:

1. **feature_extraction.py** is used for extracting features given a filename and specified hand (left or right)
2. **model_training** is used to train the severity assessment model, after the features are extracted and correlated features are removed.
3. model_training_with_neural_network.py was used to report performance of neural network models. 
The setup is exactly the same as the previous one. However this uses standard Pytorch libraries and may require installations of more packages.


**Cite our paper:
**
Islam, M.S., Rahman, W., Abdelkader, A. et al. 
Using AI to measure Parkinson’s disease severity at home. 
npj Digit. Med. 6, 156 (2023). 
https://doi.org/10.1038/s41746-023-00905-9
