from sklearn.metrics import matthews_corrcoef,accuracy_score,auc,precision_score,recall_score,f1_score,roc_curve
from tensorflow import keras
from SplitDataset import get_data_set
from Embedding import embedding
import argparse
import os
import pandas as pd
import numpy as np
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

def main():
	parser=argparse.ArgumentParser(description='PTG-PLM a tool for PTM site prediction using Protein Language Models and CNN')
	parser.add_argument('--BENCHMARKS_DIR',type=str, default='datasets/', help='dataset path')
	parser.add_argument('--benchmark_name', type=str, default='N_gly', help='dataset name')
	parser.add_argument('--site', default='N',type=str,help='PTM site residue(s) for more one residue can write as (\'X\', \'Y\')')
	parser.add_argument('--w', default=12, type=int, help='number of residues that are surrounding the PTM residues')
	parser.add_argument('--PLM', default='ProtBert-BFD',type=str, help='used protein language model (ProtBert-BFD, ProtBert, ProtAlbert, ProtXlnet, ESM-1b, or TAPE)')
	parser.add_argument('--model_path', default='models/',type=str, help='path of model for prediction')
	parser.add_argument('--result_path', default='results/',type=str, help='path to save the prediction results')
	args = parser.parse_args()
	BENCHMARKS_DIR=args.BENCHMARKS_DIR
	benchmark_name=args.benchmark_name
	site=args.site
	w=args.w
	PLM=args.PLM
	model_path=args.model_path
	result_path=args.result_path
	if PLM.upper()=='PROTBERT-BFD':
		model_name='prot_bert_bfd'
	elif PLM.upper()=='PROTBERT':
		model_name='prot_bert'
	elif PLM.upper()=='PROTALBERT':
		model_name='prot_albert'
	elif PLM.upper()=='PROTXLNET':
		model_name='prot_xlnet'
	elif PLM.upper()=='ESM-1B':
		model_name='esm1v_t33_650M_UR90S_1'
	elif PLM.upper()=='TAPE':
		model_name='bert-base'
	else:
		print('PLM must be ProtBert-BFD, ProtBert, ProtAlbert, ProtXlnet, ESM-1b, or TAPE')
		return
	if not os.path.exists(BENCHMARKS_DIR+benchmark_name +'_test.fasta'):
		raise IOError('The protein sequences FASTA file: '+ BENCHMARKS_DIR+benchmark_name +'_test.fasta' + ' does not exist!!!' )
		return
	if not os.path.exists(BENCHMARKS_DIR+benchmark_name +'_pos.csv'):
		raise IOError('The positive sites file: '+ BENCHMARKS_DIR+benchmark_name +'_test_pos.csv' + ' does not exist!!!')
		return
	if (2*w+1)%2==0:
		print('The windw size (2*w+1) value must be odd!!')
		return
		
	test_set =get_data_set(model_name,BENCHMARKS_DIR,benchmark_name+'_test',w,site,balanced=1)
	X_test=embedding(model_name,test_set,w)
	Y_test=test_set['label'].astype('float32')
	scoring=pd.DataFrame()
	model=keras.models.load_model(model_path)
	pred=model.predict(X_test)
	Y_pred1 = [1 if n >= 0.5 else 0 for n in flatten_list(pred)]
	mes=perf_measure(flatten_list(Y_test),Y_pred1)
	mcc= matthews_corrcoef(y_true= flatten_list(Y_test), y_pred= Y_pred1)
	acc=accuracy_score(y_true=flatten_list(Y_test), y_pred= Y_pred1)
	recall=recall_score(y_true=flatten_list(Y_test), y_pred= Y_pred1)
	pre=precision_score(y_true=flatten_list(Y_test), y_pred= Y_pred1)
	fpr1, tpr1, thresholds = roc_curve( flatten_list(Y_test), Y_pred1)
	auc1=auc(fpr1, tpr1)
	scoring.loc['testing_results','Accuracy']= np.round(acc,4)
	scoring.loc['testing_results','Recall']= np.round(recall,4)
	scoring.loc['testing_results','Precision']= np.round(pre,4)
	scoring.loc['testing_results','AUC']= np.round(auc1,4)
	scoring.loc['testing_results','MCC']= np.round(mcc,4)
	scoring.to_csv(result_path+'testing_results.csv')
if __name__ == '__main__':
	main()
