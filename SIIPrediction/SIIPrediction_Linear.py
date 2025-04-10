from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Load the .mat file
data        = loadmat('Results_TRF_Grace_cEFR_bEFR.mat')
cEFR_struct = data['cEFR'][0, 0]
SPow        = cEFR_struct['SPow']
NPow        = cEFR_struct['NPow']
bEFR_struct = data['bEFR'][0, 0]
AllCC       = data['AllR']
bEFR_p      = bEFR_struct['p']
NumPho      = bEFR_p.shape[1]  
NumS, NumC  = bEFR_p[0,0].shape
SIIVal      = [99, 55.7, 18.1, 37.6, 87.5]

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


# organise data
SII_list    = []
CC_list     = []
Det_List    = []
SPow_List   = []
NPow_List   = []
SNR_List    = []
for j in range(NumC):       # each SII condition
    for i in range(NumS):   # each subject
        if not np.isnan( bEFR_p[0, 0][i,j] ):
            Det_List.append(DetectionCount[i,j])
            SII_list.append(SIIVal[j])
            CC_list.append(AllCC[i,j])
            SPow_List.append(SPow[i,j])
            NPow_List.append(NPow[i,j])
            SNR_List.append( SPow[i,j] / NPow[i,j] )  # use p value from first phoneme

# Convert to numpy arrays
SII_Targets = np.array( SII_list ) / 100
Obs_CC      = np.array( CC_list )
Obs_Det     = np.array( Det_List )
Obs_NPow    = np.array( NPow_List )
Obs_SPow    = np.array( SPow_List )

# # #
# model
# # #

# X = sm.add_constant( Obs_CC)  
X       = sm.add_constant( Obs_SPow )  
model   = sm.OLS(SII_Targets, X)
result  = model.fit()

# Output results
print(result.summary())

# Get predictions
SII_pred = result.fittedvalues

# Compute R² manually (optional, already in summary)
ss_res  = np.sum((SII_Targets - SII_pred) ** 2)
ss_tot  = np.sum((SII_Targets - np.mean(SII_pred)) ** 2)
r2      = 1 - ss_res / ss_tot
print(f"R²: {r2:.3f}")




# Plot: add jitter to the predicted values (only when using detection counts)
jitter_strength     = 0.01
SII_pred_jittered   = SII_pred + np.random.normal(0, jitter_strength, size=SII_pred.shape[0] )

# Plot predictions vs true
plt.scatter( SII_Targets, SII_pred_jittered )
plt.xlabel( "True SII Value" )
plt.ylabel( "Predicted SII Value (+jitted)" )
plt.title( "Linear Regression: True vs Predicted SII" )
plt.plot( [min(SII_Targets), max(SII_Targets)], [min(SII_Targets), max(SII_Targets)], 'r--')
plt.grid( True )
plt.show()

# Plot: detection count vs SII
jitter_strength = 0.1
X_jittered      = X[:,1] + np.random.normal(0, jitter_strength, size=X.shape[0] )

# Generate smooth x values for the regression line
x_smooth = np.linspace(min(X[:, 1]), max(X[:, 1]), 200)
X_smooth = sm.add_constant(x_smooth)  # Add intercept term for prediction
y_smooth = result.predict(X_smooth)


plt.figure(figsize=(8, 5))
plt.scatter( X_jittered, SII_Targets, alpha=0.7, label='True SII')
plt.plot( X_smooth[:,1], y_smooth, 'r--')

#plt.title("Linear Regression using bEFR detection counts")
#plt.xlabel("bEFR Detection Count")
plt.title("Linear Regression using cEFR signal power")
plt.xlabel("cEFR signal power")

plt.ylabel("true SII")
plt.grid(True)
plt.tight_layout()
plt.show()
