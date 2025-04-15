from scipy.io import loadmat
import numpy as np
from scipy.optimize import curve_fit
from sigmoid_functions import sigmoid_basic  # External file
from sklearn.metrics import r2_score
import statsmodels.api as sm


# # # # # # # # # #
# Emma ANH data # #
# # # # # # # # # #
data_E          = loadmat('CHL_Results_EMMA.mat')
CC_CHL          = data_E['CHL_CC'].ravel()
SII_CHL         = data_E['CHL_SII'].ravel() / 100
NumS_E, NumC_E  = data_E['CHL_CC'].shape

# sigmoid fit
initial_guess   = [1, np.median(CC_CHL)]
popt, _         = curve_fit(sigmoid_basic, CC_CHL, SII_CHL, p0=initial_guess)
SII_pred        = sigmoid_basic(CC_CHL, *popt)                           # Predicted SII values
r2              = r2_score(SII_CHL, SII_pred)

# linear model fit
X       = sm.add_constant( CC_CHL  )  
model   = sm.OLS(SII_CHL, X)
result  = model.fit()
print( result.summary() )


# # # # # # # # # #
# Grace CHL data  #
# # # # # # # # # #
data_G          = loadmat('Results_TRF_Grace_cEFR_bEFR.mat')
NumS_G, NumC_G  = data_G['AllR'].shape
CC_ANH          = np.array(data_G['AllR'].flatten())
SII_ANH         = np.tile([99, 55.7, 18.1, 37.6, 87.5], (NumS_G, 1)) / 100
SII_ANH         = SII_ANH.flatten()

# apply mask to remove NaNs
mask    = ~np.isnan(np.array(CC_ANH))
CC_ANH  = CC_ANH[mask]
SII_ANH = SII_ANH[mask]

# sigmoid fit
initial_guess   = [1, np.median(CC_ANH)]
popt, _         = curve_fit(sigmoid_basic, CC_ANH, SII_ANH, p0=initial_guess)
SII_pred        = sigmoid_basic(CC_ANH, *popt)      # Predicted SII values
r2              = r2_score(SII_ANH, SII_pred)

# linear model fit
X       = sm.add_constant( CC_ANH  )  
model   = sm.OLS(SII_ANH, X)
result  = model.fit()
print( result.summary() )



# # # # # # # # # # # # # # # # #
# Emma and Grace data combined  #
# # # # # # # # # # # # # # # # #

# Combine 
CC_all      = np.concatenate( [ CC_CHL, CC_ANH ] )
SII_all     = np.concatenate( [ SII_CHL, SII_ANH ] )
group_all   = np.concatenate( [ np.zeros(len( CC_CHL )), np.ones(len( CC_ANH )) ] )

# no distinction between groups
X1      = sm.add_constant(CC_all)  # Shape: (N, 2) — constant + CC
model1  = sm.OLS(SII_all, X1).fit()
print(model1.summary())


# distinction between groups
X2 = np.column_stack([
    np.ones_like(CC_all),       # constant
    CC_all,                     # CC
    group_all,                  # Group indicator
    CC_all * group_all          # interaction term
])
model2 = sm.OLS(SII_all, X2).fit()
print(model2.summary())

