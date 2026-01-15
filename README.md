# Income Tax Regime Recommendation using Machine Learning

## Overview
This project uses Machine Learning to recommend the most suitable income tax regime (Old or New) for individuals based on their income, age, deductions, and tax values. The objective is to demonstrate how data-driven models can support tax advisory and decision-making.

## Dataset
The dataset contains sample records with the following attributes:
- Annual Income
- Age
- Total Deductions
- Tax under Old Regime
- Tax under New Regime
- Best Tax Regime (target variable)

The target variable is encoded and used for supervised learning.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression

## Methodology
1. Loaded and cleaned the dataset using Pandas.
2. Encoded categorical target values using Label Encoding.
3. Split the dataset into training and testing sets.
4. Trained a Logistic Regression model to classify the optimal tax regime.
5. Evaluated model performance using accuracy score.
6. Generated predictions for sample inputs to demonstrate real-world usage.

## Results
The trained model achieved satisfactory accuracy on the test dataset and successfully recommended the appropriate tax regime for sample inputs.

## Business Insight
The project demonstrates how machine learning models can assist tax professionals by providing data-backed recommendations, reducing manual analysis, and improving decision efficiency.

## How to Run
### Prerequisites
- Python 3.8+

1. Install required libraries:
```bash
pip install pandas scikit-learn
2. Run the model: 
python model.py 


## Author Ushnik Gupta


