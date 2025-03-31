from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the .mat file
data = loadmat('ResultsTRF.mat')
cEFR_struct = data['cEFR'][0, 0]
p = cEFR_struct['p']
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

SIIVal = [0.2, 0.7, 0.5, 0.6, 0.99]  # 5 values, one per column

# Stack the features
num_rows, num_cols = p_array.shape  # should be (50, 5)

X_list = []
y_list = []

for j in range(num_cols):  # for each SII value
    for i in range(num_rows):  # for each observation
        feature_vector = [p_array[i, j], NPow_array[i, j], SPow_array[i, j]]
        # Append the 6 Pho features for the same [i, j] location
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train MLP
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot predictions
plt.scatter(y_test, y_pred)
plt.xlabel("True SIIVal")
plt.ylabel("Predicted SIIVal")
plt.title("MLP Predictions with 9 Features")
plt.grid(True)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # identity line
plt.show()
