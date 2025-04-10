from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# Load the .mat file
data = loadmat('Results_TRF_Grace_cEFR_bEFR.mat')
cEFR_struct = data['cEFR'][0, 0]
bEFR_struct = data['bEFR'][0, 0]
# bEFR_struct.dtype.names
# cEFR_struct.dtype.names

bEFR_p      = bEFR_struct['p']
NumPho      = bEFR_p.shape[1]  
NumS,NumC   = bEFR_p[0,0].shape

DetectionCount = np.zeros((NumS, NumC), dtype=int)
for c in range(NumC):
    for s in range(NumS):
        for i in range(NumPho):
            if ~np.isnan( bEFR_p[0,i][s,c] ) and bEFR_p[0,i][s,c]< 0.05:
                DetectionCount[s,c] += 1
        
                     
SIIVal = [99, 55.7, 18.1, 37.6, 87.5]  # 5 values, one per column

X_list = []
y_list = []
p_list = []
for s in range(NumS):  # for each SII value
    for c in range(NumC):  # for each observation
        X_list.append(DetectionCount[s,c])
        p_list.append(bEFR_p[0,0][s,c])         # to make the mask
        y_list.append(SIIVal[c])


# Remove rows with NaNs
mask    = ~np.isnan(p_list)
X       = np.array(X_list)[mask]
y       = np.array(y_list)[mask]

# linear model
X = X.reshape(-1, 1)  # Reshape to 2D array for sklearn
linreg = LinearRegression()
linreg.fit(X, y)
y_pred = linreg.predict(X)

r2      = r2_score(y, y_pred)
rmse    = np.sqrt(mean_squared_error(y, y_pred))


print(f"Model coefficient: {linreg.coef_[0]:.3f}")
print(f"Intercept: {linreg.intercept_:.3f}")
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Plot
plt.scatter(X, y, label='True SII')
plt.plot(X, y_pred, color='red', label='Linear Fit')
plt.xlabel("Detection Count")
plt.ylabel("SII")
plt.title("Linear Regression: SII vs Detection Count")
plt.grid(True)
plt.legend()
plt.show()

# Add jitter to X for visualization only (not for modeling)
jitter_strength = 0.1
X_jittered = X + np.random.normal(0, jitter_strength, size=X.shape)

# Plot with jittered X
plt.scatter(X_jittered, y, label='True SII', alpha=0.7)
plt.plot(X, y_pred, color='red', label='Linear Fit')  # keep original X for line
plt.xlabel("Detection Count (jittered)")
plt.ylabel("SII")
plt.title("Linear Regression: SII vs Detection Count")
plt.grid(True)
plt.legend()
plt.show()







# = cEFR_struct['p']
SPow = cEFR_struct['SPow']
NPow = cEFR_struct['NPow']

bEFR_struct = data['bEFR'][0, 0]
SMag = bEFR_struct['Unbiased_SMag']
Pho1, Pho2, Pho3, Pho4, Pho5, Pho6 = SMag[0]

# Ensure all inputs are numpy arrays
p_array = np.array(p)
SPow_array = np.array(SPow)
NPow_array = np.array(NPow)
Pho_arrays = [np.array(Pho) for Pho in [Pho1, Pho2, Pho3, Pho4, Pho5, Pho6]]

SIIVal = [99, 55.7, 18.1, 37.6, 87.5]  # 5 values, one per column

# Stack the features
num_rows, num_cols = p_array.shape  # should be (50, 5)

X_list = []
y_list = []

for j in range(num_cols):  # for each SII value
    for i in range(num_rows):  # for each observation
        feature_vector = [p_array[i, j], NPow_array[i, j], SPow_array[i, j]]
        pho_features = [Pho[i, j] for Pho in Pho_arrays]
        feature_vector.extend(pho_features)
        X_list.append(feature_vector)
        y_list.append(SIIVal[j])

X = np.array(X_list)  # shape: (250, 9)
y = np.array(y_list)  # shape: (250,)

# Remove rows with NaNs
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

# Scale features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X_scaled = X

plt.plot(X_scaled)  # identity line
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train MLP
mlp = MLPRegressor(hidden_layer_sizes=(2, ), activation='relu', max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# --- MLP Predictions (already computed) ---
mlp_residuals = y_test - y_pred
var_y = np.var(y_test, ddof=1)
var_res_mlp = np.var(mlp_residuals, ddof=1)
explained_var_mlp = (1 - var_res_mlp / var_y) * 100

print("Var(y_test):", var_y)
print("Var(residuals):", var_res_mlp)

# Plot predictions
plt.scatter(y_test, y_pred)
plt.xlabel("True SIIVal")
plt.ylabel("Predicted SIIVal")
plt.title("MLP Predictions with 9 Features")
plt.grid(True)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # identity line
plt.show()

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.5)  # alpha is the regularization strength
ridge.fit(X_train, y_train)

# Predict using all models
y_pred_lin = linreg.predict(X_test)
y_pred_tree = tree.predict(X_test)
y_pred_ridge = ridge.predict(X_test)

# Compute residuals
res_lin = y_test - y_pred_lin
res_tree = y_test - y_pred_tree
res_ridge = y_test - y_pred_ridge

# Compute variance of residuals
var_y = np.var(y_test, ddof=1)  # total variance in target
var_res_lin = np.var(res_lin, ddof=1)
var_res_tree = np.var(res_tree, ddof=1)
var_res_ridge = np.var(res_ridge, ddof=1)

# Compute explained variance %
expl_var_lin = (1 - var_res_lin / var_y) * 100
expl_var_tree = (1 - var_res_tree / var_y) * 100
expl_var_ridge = (1 - var_res_ridge / var_y) * 100

# Print all
print(f"Explained Variance (Linear Regression): {expl_var_lin:.2f}%")
print(f"Explained Variance (Decision Tree):     {expl_var_tree:.2f}%")
print(f"Explained Variance (Ridge Regression):  {expl_var_ridge:.2f}%")

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

forest = RandomForestRegressor(n_estimators=1000, max_depth=4, random_state=1)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

res_forest = y_test - y_pred_forest
var_res_forest = np.var(res_forest, ddof=1)
expl_var_forest = (1 - var_res_forest / var_y) * 100

print(f"Explained Variance (Random Forest): {expl_var_forest:.2f}%")

# Plot predictions
plt.scatter(y_test, y_pred_forest)
plt.xlabel("True SIIVal")
plt.ylabel("Predicted SIIVal")
plt.title("MLP Predictions with 9 Features")
plt.grid(True)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # identity line
plt.show()


feature_names = ['p', 'NPow', 'SPow', 'Pho1', 'Pho2', 'Pho3', 'Pho4', 'Pho5', 'Pho6']

# Visualize the first tree in the forest
plt.figure(figsize=(20, 10))
plot_tree(forest.estimators_[0], filled=True, feature_names=feature_names)
plt.show()

print(f"Explained Variance (Random Forest): {expl_var_forest:.2f}%")
