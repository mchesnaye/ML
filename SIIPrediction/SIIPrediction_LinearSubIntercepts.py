from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the .mat file
data = loadmat('Results_TRF_Grace_cEFR_bEFR.mat')
AllR = data['AllR']

num_rows, num_cols = AllR.shape 
SIIVal = [99, 55.7, 18.1, 37.6, 87.5] 

R_list = []
SII_list = []
subject_list = []

# Collect data
for j in range(num_cols):  # each SII condition
    for i in range(num_rows):  # each subject
        R_val = AllR[i, j]
        if not np.isnan(R_val):
            R_list.append(float(R_val))
            SII_list.append(SIIVal[j])
            subject_list.append(i)  # subject ID

# Convert to numpy arrays
R = np.array(R_list)
SII = np.array(SII_list)
subject_ids = np.array(subject_list)

# Create design matrix for fixed effect (R) and intercept
X = sm.add_constant(R)  # adds column of ones for intercept
# Create groups for random effects
groups = subject_ids

# Fit Linear Mixed Effects Model: SII ~ R + (1 | subject)
model = sm.MixedLM(SII, X, groups=groups)
result = model.fit()

# Output results
print(result.summary())

# Get predictions
SII_pred = result.fittedvalues

# Compute R² manually
ss_res = np.sum((SII - SII_pred) ** 2)
ss_tot = np.sum((SII - np.mean(SII)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"R²: {r2:.3f}")

# Plot predictions vs true
plt.scatter(SII, SII_pred)
plt.xlabel("True SII Value")
plt.ylabel("Predicted SII Value")
plt.title("LME (Random Intercepts): True vs Predicted SII")
plt.plot([min(SII), max(SII)], [min(SII), max(SII)], 'r--')
plt.grid(True)
plt.show()
