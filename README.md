# sgCarMart-Kaggle-Competition
## Task 1: Resale Price Prediction:
Utilize the tree-based models, including XGBoost, LightGBM and CatBoost, to predict the resale price. We start with Feature engineering, followed by Model implementation and tunning, and end with evaluation.
## Task 2: Recommendation system: 
This personalized recommendation system will help the new customer to explore their interests and find the best car that they might be interested.
The entire system address the issues of cold-start, overspecialization. It`s robust enough to adjust the recommended items based on the accumulated browsing history and prevent to recommend the same item that has been clicked by the customer.

# Dataset
## File Description
train.csv - the training set
test.csv - the test set

## Data Description 
Please refer to https://www.kaggle.com/competitions/cs5228-2021-semester-1-final-project/data

# Task 1
## Data Cleaning 
Missing Value; Duplicates; Outlier; 
## Feature Engineering
1. Data Transformation: Continous Features; Categorical Feature; Free Text Feature; Date-related Feature
2. Feature selection based on Correlation Matrix 
## Models
1. Baseline Model: XGboost
2. Other Models: LightGBM and CatBoost.

- Data Preprocess: Run "Feature engineering.ipynb"
- Model Selections: Run"Pycaret Models.ipynb" --> Utilize the Pycaret package to test the different models with their default parameters on the training set and the results. The results show that CatBoost was the best, followed by LightGBM and finally XGBoost
- XGBoost model tunning: Run "train_xgb.ipynb" --> Utilize the GridSearchCV package to find the best parameters of the XGBoost Model.


# Task 2 
1. Design for new customer: This recommendation system can do recommendation for new customers by implementing popular-based engines.
2. Create Customer Profile: This recommendation system can remember customers' browsing history and make recommendations based on item-item similarities
3. Prevent Overspecialization: This recommendation system can prevent overspecializa- tion by adding some Popular item into the recommenda- tion system.
4. Prevent Repeating: Prevent to recommend the same item that has been clicked by the customer.
5. Robustness: The recommendation system can adjust the recommended items based on the accumulated browsing history.
Recomemndation System Workflow: 
<img width="728" alt="System_Workflow" src="https://github.com/yichenCY/sgCarMart-Kaggle-Competition/assets/87318317/9462a1f4-5337-45f0-a877-91ba30dd06ea">
