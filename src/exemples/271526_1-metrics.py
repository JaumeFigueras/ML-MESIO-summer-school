
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau
pd.set_option("display.max_rows", 60, "display.max_columns", None)

# https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics


# Rank metrics
################################################################################
ProfA=[3,7,8]
ProfB=[1,5,10]
tau, p_value = kendalltau(ProfA,ProfB)
print(tau)



ProfA=[3,8,7]
ProfB=[1,5,10]
tau, p_value = kendalltau(ProfA,ProfB)
print(tau)


ProfA=[3,7,8]
ProfB=[0,1,1]
tau, p_value = kendalltau(ProfA,ProfB)
print(tau)
print('Gini:',2*roc_auc_score(ProfB,ProfA)-1)


train = pd.read_csv('./DATA/train.csv')
test = pd.read_csv('./DATA/test.csv')


def miss(ds,n_miss):
	miss_list=list()
	for col in list(ds):
		if ds[col].isna().sum()>=n_miss:
			print(col,ds[col].isna().sum())
			miss_list.append(col)
	return miss_list
# Which columns have 1 missing at least...
print('\n################## TRAIN ##################')
m_tr=miss(train,0)


icol='X30'
print('AUC: ',roc_auc_score(train['TARGET'],train[icol]))
print('GINI: ',2*roc_auc_score(train['TARGET'],train[icol])-1)




# Regression metrics
################################################################################
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

TARGET=[50,55,70,90,87,112,96,75,67,72,84,92,101,98] # People weights in Kg



# Model 1: +5 Shift
model1=[i+5 for i in TARGET]

for i in range(0,len(TARGET)):
	print(TARGET[i],'->',model1[i])

print('MAE:',mean_absolute_error(TARGET,model1))
print('RMSE:',mean_squared_error(TARGET,model1)**0.5,'\n')


# Model 2: x2
model2=[i*2 for i in TARGET]

for i in range(0,len(TARGET)):
	print(TARGET[i],'->',model2[i])

print('MAE:',mean_absolute_error(TARGET,model2))
print('RMSE:',mean_squared_error(TARGET,model2)**0.5,'\n')


# Model 3: Outlier
model3=model1.copy()
model3[0]=1000 #

for i in range(0,len(TARGET)):
	print(TARGET[i],'->',model3[i])

print('MAE:',mean_absolute_error(TARGET,model3))
print('RMSE:',mean_squared_error(TARGET,model3)**0.5,'\n')
