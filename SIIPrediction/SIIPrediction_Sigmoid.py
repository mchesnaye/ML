from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sigmoid_functions import sigmoid_basic  # External file

# 
SIIVal = np.array([99, 55.7, 18.1, 37.6, 87.5]) / 100

# Load the .mat file
data        = loadmat('Results_TRF_Grace_cEFR_bEFR.mat')
cEFR_struct = data['cEFR'][0, 0]
bEFR_struct = data['bEFR'][0, 0]
AllCC       = data['AllR']
bEFR_p      = bEFR_struct['p']
NumPho      = bEFR_p.shape[1]  
NumS, NumC  = bEFR_p[0,0].shape

# # # # # #
# Count bEFR detections
# # # # # # 
DetectionCount = np.zeros((NumS, NumC), dtype=int)
for c in range(NumC):
    for s in range(NumS):
        for i in range(NumPho):
            val = bEFR_p[0, i][s, c]
            if (not np.isnan(val)) and (val < 0.05):
                DetectionCount[s, c] += 1

# Build X and y
CC_list     = []
Count_list  = []
SII_list    = []
pval_list   = []
for s in range(NumS):
    for c in range(NumC):
        Count_list.append(DetectionCount[s, c])
        pval_list.append(bEFR_p[0, 0][s, c])  # use p value from first phoneme
        SII_list.append(SIIVal[c])
        CC_list.append(AllCC[s,c])

# Mask NaNs
mask        = ~np.isnan(np.array(pval_list))
Target_SII  = np.array(SII_list)[mask]
Obs_CC      = np.array(CC_list)[mask]
Obs_Counts  = np.array(Count_list)[mask]

# Fit sigmoid
X               = Obs_CC
initial_guess   = [1, np.median(X)]
popt, _         = curve_fit(sigmoid_basic, X, Target_SII, p0=initial_guess)
SII_pred        = sigmoid_basic(X, *popt)                           # Predicted SII values

# Compute R²
r2 = r2_score(Target_SII, SII_pred)

# Smooth curve for plotting
x_fit   = np.linspace(min(X), max(X), 200)
SII_fit = sigmoid_basic(x_fit, *popt)

# Plot: detection count vs SII
jitter_strength = 0.1
X_jittered      = X + np.random.normal(0, jitter_strength, size=X.shape)


# Plot: true vs predicted SII
plt.figure(figsize=(6, 6))
plt.scatter(Target_SII, SII_pred, alpha=0.8)
plt.plot(x_fit, SII_fit, color='red', label='Sigmoid Fit')
plt.xlabel("True SII")
plt.ylabel("Predicted SII")
plt.title("True vs Predicted SII")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print results
print("Fitted sigmoid parameters:")
print(f"k (slope): {popt[0]:.3f}")
print(f"x₀ (midpoint): {popt[1]:.3f}")
print(f"R²: {r2:.3f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_jittered, Target_SII, alpha=0.7, label='True SII')
plt.plot(x_fit, SII_fit, color='red', label='Sigmoid Fit')

#plt.xlabel("Detection Count (jittered)")
#plt.title("Sigmoid Fit: SII vs Detection Count")
plt.title("Sigmoid Fit: SII vs CC value")
plt.xlabel("CC value (jittered)")

plt.ylabel("Scaled SII")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
