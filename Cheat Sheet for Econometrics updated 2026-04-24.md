# Cheat Sheet for Data Analysis #8: Regression Analysis & Prediction with OLS

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

Load and InsData
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
**Meaning:** Displays detailed statistical results, including:
- **R-squared**: Proportion of variance in $Y$ explained by $X$s.
- **Coefficients (coef)**: The estimated impact of each $X$ on $Y$.
- **P-values (P > |t|)**: Statistical significance of each coefficient (typically < 0.05 is significant).
- **F-statistic**: Overall significance of the model.

Extract Coefficients
```python
# Get raw coefficients
params = model.params

# Round for readability
print(params.round(3))
```
**Meaning:** Isolates the intercept and slope coefficients for manual calculation or prediction.

## Working with Transformations in Formulas
Logarithmic Transformations
```python
# Log of dependent variable
model_log_y = ols('np.log(wage) ~ exper + female', data=df).fit()

# Log of independent variables
model_log_x = ols('np.log(output) ~ np.log(capital) + np.log(labour)', data=df).fit()
```
**Meaning:**
- `np.log(y)`: Useful when percentage changes in $X$ affect absolute changes in $Y$, or to normalize skewed data.
- `np.log(x)`: Useful for elasticity calculations (percentage change in $X$ leads to percentage change in $Y$).

Polynomial Terms (Non-Linear Relationships)
```python
# Include squared term for experience (diminishing returns)
# Use I() to perform arithmetic inside the formula
model_poly = ols('wage ~ exper + I(exper**2) + educ', data=df).fit()
```
**Meaning:** `I(exper**2)` tells the formula parser to treat `exper**2` as a mathematical operation rather than a variable name interaction. This captures curvilinear relationships.

- Variable names in the formula must exactly match column names in the DataFrame.

Remember: Correlation does not imply causation. OLS identifies associations based on the provided data. Always consider omitted variable bias and the theoretical basis for your model specification.
