import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 

df = pd.read_csv("performance.csv")
df.head()

x=df[['Hours_Studied','Hours_Slept','Attendance']]
y = df["Final_Score"]


xtrain , xtest , ytrain,ytest = train_test_split(x,y,test_size= 0.30 ,random_state= 42) 


# standardization
# scaler = StandardScaler()
# xtest = scaler.transform(xtest)
# xtrain = scaler.fit_transform(xtrain)

# standardization
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)  # Fit and transform training data
xtest = scaler.transform(xtest)        # Transform test data only



reg = LinearRegression()
reg = reg.fit(xtrain,ytrain)

mse = cross_val_score(reg, xtrain, ytrain, scoring='neg_mean_squared_error', cv=5)
np.mean(mse)

# prediction
reg_prediction = reg.predict(xtest)
reg_prediction

error = ytest - reg_prediction

sns.displot(error,kind='kde')


output = pd.DataFrame({
    'Actual': ytest.values,
    'Predicted': reg_prediction
})
# print(output.head())


# Graph


# plt.scatter(ytest, reg_prediction, color='blue', alpha=0.7)
# plt.xlabel("Actual Final Score")
# plt.ylabel("Predicted Final Score")
# plt.title("Actual vs Predicted Final Score")
# plt.grid(True)
# plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', linestyle='--')  # 45-degree line
# plt.show()


hours_studied = float(input("Enter Hours Studied : "))
hours_slept = float(input("Enter Hours slept : "))
attendance  = float(input("Enter Attendance (0-100%) : "))

user_features = np.array([[hours_studied,hours_slept,attendance]])

user_features_scale = scaler.transform(user_features)

user_prediction = reg.predict(user_features_scale)

print(f"Predicted Final Score: {user_prediction[0]:.2f}")



