import os
import pandas as pd
import numpy as np
import re
from re import S
from ExtractPeptide import ExtractPeptideforTraining

def balanced_subsample(df):
    trainX = df.values
    train_pos=trainX[np.where(trainX[:,1]==1)]
    train_neg=trainX[np.where(trainX[:,1]!=1)]
    train_pos=pd.DataFrame(train_pos)
    train_neg=pd.DataFrame(train_neg) 
    slength=int(train_pos.shape[0])  
    train_pos_s=train_pos.sample(slength)
    train_neg_s=train_neg.sample(train_neg.shape[0])
    train_neg_ss=train_neg_s[0:slength]
    dfbalanced=pd.concat([train_pos_s,train_neg_ss])
    dfbalanced.columns=df.columns
    return dfbalanced.reset_index()


def get_data_set(model_name,BENCHMARKS_DIR,dsname,w,site,balanced=1):
    unknownChr='<unk>' if model_name=='prot_xlnet' else 'X'
    space=True if model_name.startswith('prot') else False
    fast_path = os.path.join(BENCHMARKS_DIR, '%s.fasta' % dsname)
    pos_path = os.path.join(BENCHMARKS_DIR, '%s_pos.csv' % dsname)
    trainfrag,ids,poses,siteDic=ExtractPeptideforTraining(fast_path,pos_path,w,'X',site=site)
    trainfrag = trainfrag.dropna().drop_duplicates()
    df=pd.DataFrame()
    dfbalanced=pd.DataFrame()
    seq=list()
    label=list()
    sid=list()
    for i in range(trainfrag.shape[0]-1):
        seq.append(''.join(trainfrag.iloc[i,1:w*2+2]) ) 
        label.append(trainfrag.iloc[i,0])
        sid.append((ids[i].replace( '>',''))+'_'+str(poses[i]+1))
    df['seq']=seq
    df['label']=label
    df['sid']=sid
    dfbalanced=balanced_subsample(df)
    
    if balanced==1:
        if not (' ' in dfbalanced['seq'][0]) and space==True:
            dfbalanced['seq']=  [" ".join(sequence)  for sequence in dfbalanced['seq']]
        if (' ' in dfbalanced['seq'][0]) and space ==False:
            dfbalanced['seq']=  [sequence.replace(' ','') for sequence in dfbalanced['seq']]
        dfbalanced['seq']= [re.sub(r"[UZOBX]", unknownChr, seq)  for seq in dfbalanced['seq']]
        return dfbalanced
    if not (' ' in df['seq'][0]) and space==True:
        df['seq']=  [" ".join(sequence)  for sequence in df['seq']]
    if (' ' in df['seq'][0]) and space ==False:
        df['seq']=  [sequence.replace(' ','') for sequence in df['seq']] 
    df['seq']= [re.sub(r"[UZOBX]", unknownChr, seq)  for seq in df['seq']]
    return df

def split_dataset(model_name,BENCHMARKS_DIR,benchmark_name,w,site):
	
	ds=get_data_set(model_name,BENCHMARKS_DIR,benchmark_name,w,site,balanced=1)
	if not os.path.exists(BENCHMARKS_DIR+benchmark_name +'_valid.fasta'):
		ds=ds.values    
		train_pos=ds[np.where(ds[:,1]==1)]
		train_neg=ds[np.where(ds[:,1]!=1)]
		train_pos=pd.DataFrame(train_pos,columns=['seq','label'])
		train_neg=pd.DataFrame(train_neg,columns=['seq','label'])
		a=int(train_pos.shape[0]*0.9);
		b=train_neg.shape[0]-int(train_pos.shape[0]*0.1);
		train_pos_s=train_pos[0:a]
		train_neg_s=train_neg[0:b];

		val_pos=train_pos[(a+1):];
		val_neg=train_neg[b+1:];
		
		train_set=pd.concat([train_pos_s,train_neg_s])
		valid_set=pd.concat([val_pos,val_neg])
	else:
		train_set = ds
		valid_set=get_data_set(model_name,BENCHMARKS_DIR,benchmark_name+'_valid',w,site,balanced=1)

	return train_set,valid_set
	
def split_dataset_glycation(model_name,BENCHMARKS_DIR,benchmark_name,w,site):
	unknownChr='<unk>' if model_name=='prot_xlnet' else 'X'
	space=True if model_name.startswith('prot') else False
	tfrag,ids,poses,focus=ExtractPeptideforTraining(BENCHMARKS_DIR+benchmark_name
													  +'.fasta',BENCHMARKS_DIR+benchmark_name
													  +'_pos.csv',w,'X',site=site)
	pldm_data=pd.DataFrame()
	seq=list()
	label=list()
	sid=list()
	pnames=list()
	locations=list()
	for i in range(tfrag.shape[0]-1):
		seq.append(''.join(tfrag.iloc[i,1:w*2+2]) ) 
		label.append(tfrag.iloc[i,0])
		sid.append((ids[i].replace( '>',''))+'_'+str(poses[i]+1))
		pnames.append(ids[i].replace( '>',''))
		locations.append(poses[i]+1)

	pldm_data['seq']=seq
	pldm_data['label']=label
	pldm_data['sid']=sid
	pldm_data['Pname']=pnames
	pldm_data['location']=locations
	pldm_data=pldm_data.drop_duplicates(subset='seq', keep=False)
	train=pd.read_csv(BENCHMARKS_DIR+benchmark_name+'_train_data.csv',index_col=0)
	test=pd.read_csv(BENCHMARKS_DIR+benchmark_name+'_test_data.csv',index_col=0)
	valid=pd.read_csv(BENCHMARKS_DIR+benchmark_name+'_valid_data.csv',index_col=0)	
	train_set=pd.merge(pldm_data,train,on=['sid','label','Pname','location'],how='inner')
	test_set=pd.merge(pldm_data,test,on=['sid','label','Pname','location'],how='inner')
	valid_set=pd.merge(pldm_data,valid,on=['sid','label','Pname','location'],how='inner')											  
	train_set=train_set.loc[:,['seq_x',  'label', 'sid', 'Pname', 'location']]
	train_set.columns=['seq',  'label', 'sid', 'Pname', 'location']
	test_set=test_set.loc[:,['seq_x',  'label', 'sid', 'Pname', 'location']]
	test_set.columns=['seq',  'label', 'sid', 'Pname', 'location']
	valid_set=valid_set.loc[:,['seq_x',  'label', 'sid', 'Pname', 'location']]
	valid_set.columns=['seq',  'label', 'sid', 'Pname', 'location']		
	if not (' ' in train_set['seq'][0]) and space==True:
	  train_set['seq']=  [" ".join(sequence)  for sequence in train_set['seq']]
	if not (' ' in test_set['seq'][0]) and space==True:
	  test_set['seq']=  [" ".join(sequence)  for sequence in test_set['seq']]  
	if not (' ' in valid_set['seq'][0]) and space==True:
	  valid_set['seq']=  [" ".join(sequence)  for sequence in valid_set['seq']]   
	  
	if (' ' in train_set['seq'][0]) and space ==False:
	  train_set['seq']=  [sequence.replace(' ','') for sequence in train_set['seq']]
	if (' ' in test_set['seq'][0]) and space ==False:
	  test_set['seq']=  [sequence.replace(' ','') for sequence in test_set['seq']]
	if (' ' in valid_set['seq'][0]) and space ==False:
	  valid_set['seq']=  [sequence.replace(' ','') for sequence in valid_set['seq']]  
	
	train_set['seq']= [re.sub(r"[UZOBX]", unknownChr, seq)  for seq in train_set['seq']]
	valid_set['seq']= [re.sub(r"[UZOBX]", unknownChr, seq)  for seq in valid_set['seq']]
	test_set['seq']= [re.sub(r"[UZOBX]", unknownChr, seq)  for seq in test_set['seq']] 
	return train_set,valid_set,test_set