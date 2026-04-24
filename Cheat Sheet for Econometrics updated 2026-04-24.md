# Cheat Sheet for Data Analysis #8: Regression Analysis & Prediction with OLS

## If you don't find something you need here, please check: https://lms3.mgimo.ru/mod/page/view.php?id=11555

## Basic Setup and Model Building
Import essential libraries
```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import warnings
warnings.simplefilter(action='ignore', category=Warning)
```
**Meaning:** Imports necessary libraries for data manipulation and Ordinary Least Squares (OLS) regression. `warnings` suppression keeps output clean.

Load and Inspect Data
```python
# Load dataset
df = pd.read_csv('dataset.csv')

# View first few rows
df.head()

# Check column names
df.columns.tolist()
```
**Meaning:** Loads data into a DataFrame, inspects the structure, and retrieves column names to identify potential dependent (`y`) and independent (`x`) variables.

Clean Data (Optional)
```python
# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0'])

# Check for missing values
print(df.isnull().sum())
```
**Meaning:** Removes irrelevant columns (often artifacts from CSV saving) and identifies gaps in data that might affect regression results.

## Building the OLS Model
Specify the Regression Formula
```python
# General syntax: 'dependent_var ~ independent_var1 + independent_var2'
model = ols(formula='y ~ x1 + x2 + x3', data=df).fit()
```
**Meaning:**
- `formula`: Defines the relationship. The variable before `~` is the dependent variable ($Y$). Variables after `~` are independent variables ($X$).
- `data`: The DataFrame containing the variables.
- `.fit()`: Executes the regression calculation.

View Model Summary
```python
model.summary()
```

Extract Coefficients
```python
# Get raw coefficients
params = model.params

# Round for readability
print(params.round(3))
```
**Meaning:** Isolates the intercept and slope coefficients for manual calculation or prediction.

## Working with Transformations in Formulas

```python
# To add log(X) use
np.log(x)
```
```python
# To add X^2 use
x^2 = np.square(x)
```
```python
# To get the exponent of X use
np.exp(x)
```
