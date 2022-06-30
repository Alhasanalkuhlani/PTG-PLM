import sys
import pandas as pd
import numpy as np




def read_fasta(fasta_file,pos_file):
        
    try:
        fp = open(fasta_file)
    except IOError:
        print ('cannot open '+fasta_file + ', check if it exist!')
        exit()
    else:
        post_prot=pd.read_csv(pos_file,index_col=0)
        fp = open(fasta_file)
        lines = fp.readlines()
        
        fasta_dict = {} #record seq for one id
        positive_dict={} #record positive positions for one id
        idlist=[] #record id list sorted
        gene_id = ""
        for line in lines:
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id] = seq
                    idlist.append(gene_id)
                seq = ""
                gene_id = line.strip() #  line.split('|')[1] all in > need to be id
            else:
                seq += line.strip().replace(' ','')
        
        fasta_dict[gene_id] = seq #last seq need to be record
        idlist.append(gene_id)
        
        
        for gene_id in fasta_dict:
            posnum=0;
            for i in post_prot.loc[gene_id.replace('>','')].values:
                if(posnum==0):
                    positive_dict[gene_id]=[int(i)-1]
                else:
                    positive_dict[gene_id]+=[int(i)-1]
                posnum+=1;
            
        fasta_dict[gene_id]=fasta_dict[gene_id].replace('#','') #delete all #
    return fasta_dict,positive_dict,idlist



def get_peptide(fasta_dict,positive_dict,idlist,nb_windows, empty_aa,site):
    seq_list_2d = []
    id_list = []
    pos_list = []
    site_list = []
    for id in idlist: #for sort
        seq = fasta_dict[id]
        if(id in positive_dict):
            positive_list=positive_dict[id]
        else:
            positive_list=[]
        for pos in range(len(seq)):
            mid_aa=seq[pos];
            #if mid_aa != "S":
            if not(mid_aa in site):
                continue
            #print(id)
            #print(pos)
            #print(mid_aa)
            start = 0
            if pos-nb_windows>0:
                start = pos-nb_windows 
            left_seq = seq[start:pos]
            
            end = len(seq)
            if pos+nb_windows<end:
                end = pos+nb_windows+1
            right_seq = seq[pos+1:end]
            
            if len(left_seq) < nb_windows:
                if empty_aa is None:
                     continue
                nb_lack = nb_windows - len(left_seq)
                left_seq = ''.join([empty_aa for _count in range(nb_lack) ]) + left_seq
            
            if len(right_seq) < nb_windows:
                if empty_aa is None:
                    continue
                nb_lack = nb_windows - len(right_seq)
                right_seq = right_seq + ''.join([empty_aa for _count in range(nb_lack) ])
            
            final_seq = left_seq + mid_aa + right_seq
            
            if(pos in positive_list):
                final_seq_list = [1] + [ AA for AA in final_seq]
            else:
                final_seq_list = [0] + [ AA for AA in final_seq]  #in process will change all 2 to 0, but keep 0
            
            id_list.append(id)
            pos_list.append(pos)
            site_list.append(str(mid_aa))
            seq_list_2d.append(final_seq_list)
    
    
    
    df = pd.DataFrame(seq_list_2d)
    df2= pd.DataFrame(id_list)
    df3= pd.DataFrame(pos_list)
    df4= pd.DataFrame(site_list)
    
    return df,id_list,pos_list,df4


def ExtractPeptideforTraining(fasta_file,glycation_pos,windows=15,empty_aa="-",site='K'):
    
    
    fasta_dict,positive_dict,idlist = read_fasta(fasta_file,glycation_pos) 
    
    
    peptide,ids,poses,site = get_peptide(fasta_dict,positive_dict,idlist, windows, empty_aa,site)
    return peptide,ids,poses,site

