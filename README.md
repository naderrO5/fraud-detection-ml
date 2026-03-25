# Fraud Detection Using Machine Learning

## Project Overview
This project detects fraudulent transactions using machine learning.

## Objectives
- Analyze transaction data
- Identify fraudulent patterns
- Build classification models
- Compare Random Forest and XGBoost

## Dataset
The dataset used for this project is a transaction dataset in CSV format.

## Tools and Libraries
- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- XGBoost
- Jupyter Notebook

## Project Structure
- `data/raw/` → raw dataset
- `notebooks/` → notebook files
- `src/` → Python scripts
- `outputs/` → figures and saved models
## Models Used
- Random Forest
- XGBoost

## Workflow
- Data loading and inspection
- Exploratory data analysis
- Train/test split
- Model training
- Model evaluation
- Model comparison

## Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
## RESULTS
| Model | ROC-AUC | PR-AUC |
|------|--------:|-------:|
| Random Forest | 0.9529 | 0.8542 |
| XGBoost | 0.9770 | 0.8653 |

## Conclusion
Both models performed well, but XGBoost achieved the highest ROC-AUC and PR-AUC scores. Since fraud detection involves highly imbalanced data, PR-AUC is especially important. Based on these results, XGBoost was the best model for this project


## Future Improvements
- Hyperparameter tuning
- Better feature engineering
- Threshold tuning
- Deployment