from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['rf'] = RandomForestClassifier(random_state=3)
	models['svm'] = SVC()
	models['XGB'] = XGBClassifier(eval_metric='mlogloss')
	return models
def ML(X_train,Y_train,X_test,Y_tes):
	X_train_2 = X_train.reshape(X_train.shape[0], (X_train.shape[1]*X_train.shape[2]))
	X_test2=X_test.reshape(X_test.shape[0], (X_test.shape[1]*X_test.shape[2]))
	models=get_models()
	scoring_all_valid=pd.DataFrame()
	results, names = list(), list()
	for name, model in models.items():
	   	model.fit(X_train_2, Y_train)
		y_pred=model.predict(X_test)
		mes=perf_measure(Y_test,y_pred)
		mcc= matthews_corrcoef(y_true= Y_test, y_pred= y_pred)
		f1=f1_score(y_true= Y_test, y_pred= y_pred)
		acc=accuracy_score(y_true=Y_test, y_pred= y_pred)
		recall=recall_score(y_true=Y_test y_pred= y_pred)
		pre=precision_score(y_true=Y_test, y_pred= y_pred)
		fpr1, tpr1, thresholds = roc_curve( Y_test, y_pred)
		auc1=auc(fpr1, tpr1)
		scoring_all_valid.loc[name,'TP']=mes[0]
		scoring_all_valid.loc[name,'FP']=mes[1]
		scoring_all_valid.loc[name,'TN']=mes[2]
		scoring_all_valid.loc[name,'FN']=mes[3]
		scoring_all_valid.loc[name,'Accuracy']= np.round(acc,4)
		scoring_all_valid.loc[name,'Recall']= np.round(recall,4)
		scoring_all_valid.loc[name,'Precision']= np.round(pre,4)
		scoring_all_valid.loc[name,'F1']= np.round(f1,4)
		scoring_all_valid.loc[name,'AUC']= np.round(auc1,4)
		scoring_all_valid.loc[name,'MCC']= np.round(mcc,4)
		# preds[name]=y_pred

	scoring_all_valid.to_csv('results/'+'ML_results.csv')