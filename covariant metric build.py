%timeit collated_APOB.loc['ARIC_230756']['Homo_sigs']
%timeit collated_APOB.loc['ARIC_230756'].Homo_sigs
%timeit collated_APOB.loc['ARIC_230756','Homo_sigs']

#Extract relative position from ACMG and match to patient ID/position - **note there are conflicting significance classifications, that is why that's not extracted also**

#%% Import modules and set directories
import time
start_time = time.time()

import numpy as np
import scipy as sp
from PIL import Image as im
from PIL import ImageDraw
import sympy as smp
import sys
import pandas as pd
pd.set_option('display.max_rows',60)
pd.set_option('display.min_rows',60)
pd.set_option('display.max_columns',60)
import seaborn as sns
import networkx as nx

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import figure
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
plt.style.use('default')
#plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [25,25];#default plot size
plt.rcParams['font.size']=30;
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg-4.3.2-2021-02-27-full_build\\bin\\ffmpeg.exe'
from IPython.display import HTML

# import gpytorch

import itertools as iter
import pyttsx3
engine = pyttsx3.init()
from datetime import datetime,date
import os

#%%

from metric_learn import NCA
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
clf = make_pipeline(NCA(), KNeighborsClassifier())
cross_val_score(clf, X, y)

#%%

from metric_learn import Covariance
from sklearn.datasets import load_iris
test = kriging_output(seen_under10percent,'seen_under10percent_CW_missense-only','RbcL','Kc_normed','KcatC')

# iris = load_iris()['data']

cov = Covariance().fit(test.values).get_mahalanobis_matrix()
x = Covariance().fit_transform(test.values)
x
cov
Covariance().fit(test).get_metric()([0,1,1],[9,0,8])
np.linalg.norm(np.array([0,1,1])-np.array([9,0,8]))

fig,ax = plt.subplots()
ax.scatter(test[:,0],test[:,1],c=test[:,2],cmap='winter')
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm.get_cmap('winter'), norm=mpl.colors.Normalize(vmin=np.min(test[:,2]), vmax=np.max(test[:,2]))))
# ax.set_xlabel('\n\nVarSeqP',fontweight='bold')
# ax.set_ylabel('\n\nk_m (oxygen) (normalised)',fontweight='bold')
# ax.set_zlabel('\n\nk_m (carbon dioxide) (normalised)',fontweight='bold')
# cb.set_label('k_cat (carbon dioxide)')
plt.show()
# fig.savefig('3dkriging_test.png',bbox_inches='tight')

fig,ax = plt.subplots()
ax.scatter(x[:,0],x[:,1],c=test[:,2],cmap='winter')
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm.get_cmap('winter'), norm=mpl.colors.Normalize(vmin=np.min(x[:,2]), vmax=np.max(x[:,2]))))
# ax.set_xlabel('\n\nVarSeqP',fontweight='bold')
# ax.set_ylabel('\n\nk_m (oxygen) (normalised)',fontweight='bold')
# ax.set_zlabel('\n\nk_m (carbon dioxide) (normalised)',fontweight='bold')
# cb.set_label('k_cat (carbon dioxide)')
plt.show()
os.chdir(r'C:\Users\bcalverley\OneDrive - Scripps Research\Documents\0Balch lab\0Covariant metric SCV')
pd.DataFrame(x,columns=['x','y','z']).groupby(['x','y']).mean().to_csv('newmetrictest_rubiscoCW'+mutationNames+'_RbcL_'+Y+'_'+Z+'.csv')

def kriging_mahaMetric_output(mutations,mutationNames,C_L_S,Y,Z,save=False,omit2=[]):
	dfInput = collate_names_and_locations(mutations,C_L_S,omit=omit2)
	df = pd.DataFrame(dfInput[:,1],index=dfInput[:,0],columns=['Loc'])
	df[['Kc_normed','Kc','Ko_normed','Ko','KcatC','Sco','KcatO','KcatC_normed','KcatO_normed']] = rates.reindex(df.index)[['Kc_normed','Kc','Ko_normed','Ko','KcatC','Sco','KcatO','KcatC_normed','KcatO_normed']]
	dfOut = df[['Loc',Y,Z]].copy()
	dfOut = dfOut.dropna()
	dfOut.columns=['x','y','z']
	dfOut.x = dfOut.x.astype('float')
	transformdf = pd.DataFrame(Covariance().fit_transform(dfOut.values),index=dfOut.index,columns=['x','y','z'])
	if save:
		os.chdir('C:\\Users\\bcalverley\\OneDrive - Scripps Research\\Documents\\0Balch lab\\0Covariant metric SCV')
		transformdf.groupby(['x','y']).mean().to_csv('mahaMetric_'+mutationNames+'_'+C_L_S+'_'+Y+'_'+Z+'.csv')
	return transformdf

kriging_mahaMetric_output(seen_under10percent,'seen_under10percent_CW_missense-only','RbcL','Kc','KcatC',save=True)

#%%

import skgstat as skg

xx, yy = np.mgrid[0:0.5 * np.pi:500j, 0:0.8 * np.pi:500j]
np.random.seed(42)

# generate a regular field
_field = np.sin(xx)**2 + np.cos(yy)**2 + 10

# add noise
np.random.seed(42)

z = _field + np.random.normal(0, 0.15, (500,  500))

plt.imshow(z, cmap='RdYlBu_r')
plt.close()

np.random.seed(42)

coords = np.random.randint(0, 500, (300, 2))
values = np.fromiter((z[c[0], c[1]] for c in coords), dtype=float)
V = skg.Variogram(coords, values)

V.plot()

#%%

os.chdir(r'C:\Users\bcalverley\OneDrive - Scripps Research\Documents\0Balch lab')

x = np.array([62,57,62,59])
y = np.array([121,121,132,112])
colour = np.array(['black','#FF0066','#FF0066','#FF0066'])
label = np.array(['A','B','C','D'])

fig,ax = plt.subplots()
ax.scatter(x,y,s=800,c=colour,marker='x',linewidths=10,zorder=2)
ax.plot([57,62,59],[121,121,112],c='k')
ax.plot([62,62],[121,132],c='k')
ax.grid(True)
ax.axis('scaled')
ax.set_xticks(np.arange(np.min(x)-1,np.max(x)+1,1))
ax.set_yticks(np.arange(np.min(y)-1,np.max(y)+1,1))
for tic in (ax.get_xticklabels() + ax.get_yticklabels()):
	tic.set_fontsize(40)
for i,lab in enumerate(label):
    ax.annotate(lab,(x[i]+.1,y[i]-.4),fontsize=48,fontweight='bold')
ax.set_xlabel('Lifespan (years)',fontsize=40,style='italic')
ax.set_ylabel('Mass (lb)',fontsize=40,style='italic')
fig.savefig('krige_eg1.png',bbox_inches='tight')


#%%Allele frequency to pathogenicity correlation

acmg_APOB.Significance

set(acmg_APOB.Significance.values)
[i for i in patientByMutation_APOB]

patientByMutation_APOB.Patient_count
#%%

r_values = pd.read_csv(os.getcwd()+'\\Kriging\\pearson_R_values.csv')
fig,ax = plt.subplots()
ax.plot([int(st.split('B')[1].split('_')[0]) for st in r_values.iloc[89:156].filename],r_values.iloc[89:156]['pearson.estimate'],'bo',label='APOB')
ax.plot([int(st.split('R')[1].split('_')[0]) for st in r_values.iloc[227:301].filename],r_values.iloc[227:301]['pearson.estimate'],'kx',label='LDLR')
ax.plot([int(st.split('K')[1].split('_')[0])-900 for st in r_values.iloc[302:399].filename],r_values.iloc[302:399]['pearson.estimate'],'r+',label='PCSK9')
ax.legend()
ax.grid(True)
ax.set_xlabel('Age')
ax.set_ylabel("Pearson's R")
plt.show()
fig.savefig('age-r_values.png',bbox_inches='tight')

#%%
# def collate_by_mutation(acmg,geno,coll):
#     data = [[loca,np.unique(acmg.loc[loca,'Relative_position'])[0],list(geno[geno[int(loca)]=='0/1'].index.values),list(geno[geno[int(loca)]=='1/1'].index.values),list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),coll.loc[list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),'TCHOL_HDL_rat'].values,coll.loc[list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),'LDL_HDL_rat'].values,coll.loc[list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),'BMI'].values,coll.loc[list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),'AGE'].values,coll.loc[list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),'TRIG'].values,coll.loc[list(geno[(geno[int(loca)]=='0/1')|(geno[int(loca)]=='1/1')].index.values),'TCHOL'].values] for loca in np.unique(acmg.index) if int(loca) in geno.columns]
#     result = pd.DataFrame(data).set_index(0)
#     result.index.name = None
#     result.columns = ['Relative_position','Het_patient_ID','Homo_patient_ID','All_patient_ID','All_TCHOL_HDL_rat','All_LDL_HDL_rat','All_BMI','All_AGE','All_TRIG','All_TCHOL']
#     return result
len(collate_names_and_locations(seenMost,'RbcL'))
len(collate_names_and_locations(seenMost,'RbcL')[collate_names_and_locations(seenMost,'RbcL')[:,0]!='Cyanidium caldarium'])
len(collate_names_and_locations(seenMost,'RbcL',omit='Cyanidium caldarium'))
rbcLSCollater(seenMost,'Sco',save=True,omit2='Cyanidium caldarium')

chaoClade = ['Atrichum undulatum','Marchantia polymorpha','Pteridium aquilinum','Platycerium superbum','Cycas panzhihuaensis','Metasequoia glyptostroboides','Nymphaea alba','Agave victoriae','Carpobrotus edulis','Echeveria elegans','Drosera capensis','Drosera venusta','Sarracenia flava','Arabidopsis thaliana','Ceratophyllum demersum','Chenopodium album','Crithmum maritimum','Dactylis glomerata','Flaveria pringlei','Flaveria floridana','Iris douglasiana','Limonium latebracteatum','Limonium stenophyllum','Limonium virgatum','Nicotiana tabacum','Pallenis maritima','Sideritis cretica','Spinacia oleracea','Teucrium heterophyllum','Trachycarpus fortunei','Triticum aestivum','Flaveria bidentis','Zea mays','Synechococcus 7002','Synechococcus 6301','Thermosynechococcus elongatus BP-1','Chlamydomonas reinhardtii','Chromatium vinosum']

kriging_output(allMutations,'allMutations_CWCladev2','RbcL','Ko_normed','Sco',save=True)
rates
np.all(np.array(['1','2'])!=None)
len(collate_names_and_locations(seenMost,'RbcL')[~np.isin(collate_names_and_locations(seenMost,'RbcL')[:,0],'Marchantia polymorpha')])
len([])
['Borophytes', 'Atrichum undulatum', 'Marchantia polymorpha', 'Ferns',
       'Pteridium aquilinum', 'Platycerium superbum', 'Gymnosperms',
       'Cycas panzhihuaensis', 'Metasequoia glyptostroboides',
       'Basal Angiosperms', 'Nymphaea alba', 'CAM plants', 'Agave victoriae',
       'Carpobrotus edulis', 'Carnivorous plants', 'Echeveria elegans',
       'Drosera capensis', 'Drosera venusta', 'Sarracenia flava', 'C3 Plants',
       'Arabidopsis thaliana', 'Ceratophyllum demersum', 'Chenopodium album',
       'Crithmum maritimum', 'Dactylis glomerata', 'Flaveria pringlei',
       'Flaveria floridana', 'Helianthus annuus', 'Iris douglasiana',
       'Limonium latebracteatum', 'Limonium stenophyllum',
       'Limonium virgatum ', 'Nicotiana tabacum', 'Oryza sativa',
       'Pallenis maritima', 'Sideritis cretica', 'Spinacia oleracea',
       'Teucrium heterophyllum', 'Trachycarpus fortunei', 'Triticum aestivum',
       'C4 Plants', 'Flaveria bidentis', 'Neurachne munroi',
       'Paspalum dilatatum ', 'Portulaca oleracea', 'Sorghum bicolor',
       'Zea mays', 'Cyanobacteria', 'Synechococcus 7002', 'Synechococcus 6301',
       'Synechococcus 7002 (SE49V,D82G)', 'Synechococcus 6301(LF137I)',
       'Synechococcus 6301(LV186I)', 'Synechococcus 6301(LS395C)',
       'Synechococcus 6301(LI462V)', 'Thermosynechococcus elongatus BP-1',
       'Synechocystis 6803', 'Diatoms', 'Phaeodactylum tricornutum',
       'Thalassiosira weissflogii', 'Thalassiosira oceanica',
       'Chaetoceros calcitrans', 'Chaetoceros muellerii',
       'Bellerochea horologicalis', 'Fragilariopsis cylindrus', 'Red algae',
       'Galdieria sulfuraria', 'Griffithsia monilis', 'Porphyridium purpureum',
       'Cyanidium caldarium', 'Cylindrotheca N1', 'Green algae',
       'Chlamydomonas reinhardtii', 'Olisthodiscus luteus', 'Euglena gracilis',
       'Proteobacteria', 'Rhodobacter sphaeroides', 'Chromatium vinosum']
rates.index.values
#%%
# LDLR_data = pd.concat([pd.read_table(data_directories[0]+ff,delimiter="\t") for ff in file_names[0] if os.path.exists(data_directories[0]+ff)])
# #LDLR_data = pd.read_csv('BCC LDLR_data.csv')
#
# LDLR_patient_ID = np.array(LDLR_data.columns[4:])
# if np.all(np.array([LDLR_data[patient].value_counts().index [0] for patient in LDLR_patient_ID])=='0/0'):
#     LDLR_mutationsPerPatient = np.array([sum(LDLR_data[patient].value_counts()[1:]) for patient in LDLR_patient_ID])
#     print('All patients contain majority 0/0 genotypes.')
# else:
#     print('Not all patients contain majority 0/0 genotypes.')
# LDLR_genotype_counts = [np.array([LDLR_data[patient].value_counts().index,LDLR_data[patient].value_counts()]) for patient in LDLR_patient_ID]

# def collate_names_and_locations(mutations,C_L_S):
#     testdf = variationPositions[C_L_S].loc[variationPositions[C_L_S][variationPositions[C_L_S][mutations[C_L_S]].any(1)].index,mutations[C_L_S]]
#     testSO = testdf.where(testdf==1).stack().index#.values
#     # import re
#     # x = np.copy(mutations[C_L_S])
#     # os.chdir('C:\\Users\\bcalverley\\OneDrive - Scripps Research\\Documents\\0Balch lab\\0Rubisco\\2016 Sequence and activity files\\'+C_L_S)
#     # with open('All_sequences.txt','r') as seq_load:
#     #     stringSeqs = np.array(re.split('\n',seq_load.read()))
#     # boolNames = [st.startswith('>') for st in stringSeqs]
#     # stringNames = stringSeqs[boolNames]
#     # stringNamesSplit = [re.split(' OS=| GN=',st) for st in stringNames]
#     # out = []
#     # for pos in testSO:
#     #     for stri in stringNamesSplit:
#     #         if stri[0].startswith('>'+pos[0]):
#     #             if len(stri)>1:
#     #                 out.append([stri[1],pos[1]/len(wt[C_L_S])])
#     #             else:
#     #                 out.append([stri[0],pos[1]/len(wt[C_L_S])])
#     # out = np.array(out)
#     SOarray = np.copy(np.array([list(ii) for ii in testSO]))
#     SOarray[:,1] = SOarray[:,1].astype('float64') / len(wt[C_L_S])
#     return SOarray#out

def duplicate_finder(list):
    seen = {}
    dupes = []
    for x in list:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes
duplicate_finder(set(pheno_data['dbGaP_Subject_ID']))

engine.say('Run complete.')
engine.runAndWait()
print("Cell completed:",datetime.now())

np.set_printoptions(threshold=sys.maxsize)

#%% Rubisco

# rubisco_seqs = pd.DataFrame(columns=['Sequence'])
# for fi in os.listdir():
#     rubisco_seqs.loc[fi] = open(fi).read().split('\n',1)[1].replace('\n','')
from Bio.Seq import Seq
from Bio import SeqIO
rubisco_seqs = []
for filename in os.listdir():
    for seq_record in SeqIO.parse(filename, "genbank"):
        rubisco_seqs.append(seq_record)
[len(sequ.seq) for sequ in rubisco_seqs]
from Bio.Align import MultipleSeqAlignment,substitution_matrices,AlignInfo
test = MultipleSeqAlignment(rubisco_seqs[14:17])
from Bio import Align,AlignIO

aligner = Align.PairwiseAligner()
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

matched_alignments = {}
for sequ in rubisco_seqs:
     alignments = aligner.align(rubisco_seqs[0].seq,sequ.seq)
     y = [len(a) for a in alignments[0].aligned][0]
     result = alignments[0]
     for ali in alignments:
         # print([len(a) for a in ali.aligned][0])
         x = [len(a) for a in ali.aligned][0]
         if x < y:
             y = x
             result = ali
     matched_alignments[sequ.description] = result
[print(x) for x in matched_alignments.values()]

from Bio.Blast import NCBIWWW as ncw
help(ncw.qblast)
#%% Import modules and set directories
import time
start_time = time.time()

import numpy as np
import scipy as sp
from PIL import Image as im
from PIL import ImageDraw
import sympy as smp
import sys
import pandas as pd
pd.set_option('display.max_rows',60)
pd.set_option('display.min_rows',60)
pd.set_option('display.max_columns',60)
import seaborn as sns

import matplotlib as mpl
from matplotlib import figure
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import ArtistAnimation
#plt.style.available
# plt.style.use('default')
# plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [25,25];#default plot size
plt.rcParams['font.size']=30;
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg-4.3.2-2021-02-27-full_build\\bin\\ffmpeg.exe'
from IPython.display import HTML

# import gpytorch
# import Bio

import itertools as iter
import pyttsx3
engine = pyttsx3.init()
from datetime import datetime
import os

#os.chdir('')
#file_name = ''
#data_directory = os.getcwd()+"\\"+file_name
#if not os.path.exists(data_directory):
#    os.mkdir(data_directory)
