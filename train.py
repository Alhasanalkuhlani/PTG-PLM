from SplitDataset import split_dataset
from Embedding import embedding
from CNNModel import CNNModel
import os
import logging
import argparse




def main():
	parser=argparse.ArgumentParser(description='PTG-PLM a tool for PTM site prediction using Protein Language Models and CNN')
	parser.add_argument('--BENCHMARKS_DIR',type=str, default='datasets/', help='dataset path')
	parser.add_argument('--benchmark_name', type=str, default='N_gly', help='dataset name')
	parser.add_argument('--site', default='N',type=str,help='PTM site residue(s) for more one residue can write as (\'X\', \'Y\')')
	parser.add_argument('--w', default=12, type=int, help='number of residues that are surrounding the PTM residues')
	parser.add_argument('--PLM', default='Esm-1b',type=str, help='used protein language model (ProtBert-BFD, ProtBert, ProtAlbert, ProtXlnet, ESM-1b, or TAPE)')
	parser.add_argument('--config_file', default='CNN_config.ini',type=str, help='CNN parameters config file')
	parser.add_argument('--model_save_path', default='models/',type=str, help='path to save the trained model')
	args = parser.parse_args()
	BENCHMARKS_DIR=args.BENCHMARKS_DIR
	benchmark_name=args.benchmark_name
	site=args.site
	w=args.w
	PLM=args.PLM
	config_file=args.config_file
	model_save_path=args.model_save_path
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
	if not os.path.exists(BENCHMARKS_DIR+benchmark_name +'.fasta'):
		raise IOError('The protein sequences FASTA file: '+ BENCHMARKS_DIR+benchmark_name +'.fasta' + ' does not exist!!!' )
		return
	if not os.path.exists(BENCHMARKS_DIR+benchmark_name +'_pos.csv'):
		raise IOError('The positive sites file: '+ BENCHMARKS_DIR+benchmark_name +'_pos.csv' + ' does not exist!!!')
		return
	if (2*w+1)%2==0:
		print('The windw size (2*w+1) value must be odd!!')
		return
		
	train_set ,valid_set =split_dataset(model_name,BENCHMARKS_DIR,benchmark_name,w,site)
	X_train=embedding(model_name,train_set,w)
	X_valid=embedding(model_name,valid_set,w)
	# if there is error with the model "esm1v_t33_650M_UR90S_1" 
	#please commit the last line in the file /usr/local/lib/python3.7/dist-packages/esm/__init__.py 
	# and run again
	Y_train=train_set['label'].astype('float32')
	Y_valid=valid_set['label'].astype('float32')
	model=CNNModel(X_train,Y_train,X_valid,Y_valid,config_file='CNN_config.ini')
	model.save(model_save_path+'PTG-PLM_'+PLM)
if __name__ == '__main__':
	main()