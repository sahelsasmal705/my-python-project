# Home Credit Default Risk Prediction

This project aims to develop a baseline model for predicting the probability of default in a credit risk competition. The model utilizes LightGBM for training and evaluation.

## Competition Information

- **Competition:** Home Credit Default Risk
- **Organizer/course:** GCI Global 2026
- **Starts:** October 2, 2026
- **Ends:** November 20, 2026 at 1:00 AM UTC
- **Ranking announcement:** December 17, 2026 (Session 14)

## Project Structure

```
my-python-project
├── README.md
├── competition
│   ├── baseline.py
│   ├── input
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── output
│       └── submission.csv
```

## Files Overview

- **README.md**: Documentation for the project, providing an overview, setup instructions, and usage details.
  
- **competition/baseline.py**: Implements a baseline model for predicting the probability of default. It includes data loading, preprocessing, model training using LightGBM, and submission generation.

- **competition/input/train.csv**: Contains the training dataset used to train the model. It includes features and the target variable indicating whether a default occurred.

- **competition/input/test.csv**: Contains the test dataset used to evaluate the model's performance. It includes the same features as the training dataset but without the target variable.

- **competition/input/sample_submission.csv**: Provides a sample submission format, showing the expected structure for the submission file, including the IDs and predicted probabilities for the test dataset.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-python-project
   ```

2. Install the required packages:
   ```
   pip install lightgbm pandas numpy scikit-learn
   ```

## Usage

1. Place your datasets in the `competition/input` directory.
2. Run the baseline model:
   ```
   python competition/baseline.py
   ```
3. Check the `competition/output` directory for the generated submission file.

## How to Submit

1. Run the model as described above. Wait for it to finish and confirm that the
   submission checks at the end show the test IDs match, there are no missing
   values, and the predicted `TARGET` values are between 0 and 1.
2. Open `competition/output/submission.csv`. This is the file to submit; do not
   upload `sample_submission.csv`.
3. Sign in to the platform or class portal used for the **Home Credit Default
   Risk** in-class competition (GCI Global 2026). Open the competition and find
   its submission/upload area.
4. Upload `submission.csv` and submit it. The competition platform may call
   this action “Submit Predictions” or something similar.
5. Check the platform's confirmation and leaderboard/submission history to make
   sure the file was accepted.

The output CSV has the required `SK_ID_CURR` and `TARGET` columns and one
prediction for each row in `test.csv`. The supplied `sample_submission.csv`
contains only example rows, so the generated submission should have the same
number of rows as `test.csv`, not the sample file.

The competition information lists the deadline as **November 20, 2026 at
1:00 AM UTC** and the ranking announcement for **December 17, 2026 (Session
14)**. Submit before the deadline shown on the official competition page,
since it may include updated details.