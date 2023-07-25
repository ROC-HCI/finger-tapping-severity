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

1. feature_extraction.py is used for extracting features given a filename and specified hand (left or right)
2. model_training.py is used to train the severity assessment model, after the features are extracted and correlated features are removed.
