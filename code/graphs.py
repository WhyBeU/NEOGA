#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
import matplotlib as mpl
import numpy as np
import pandas as pd
import math
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from math import sqrt, log10
DATADIR = ".\\data\\graph\\"
def save_obj(obj, folder, name ):
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(folder, name ):
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(folder + name + '.pkl', 'rb') as f:
            return pickle.load(f)
class OOMFormatter(mpl.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mpl.ticker._mathdefault(self.format)
# %%-
# %%--  Matplotlib style sheet
mpl.style.use('seaborn-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] ='STIXGeneral'
mpl.rcParams['mathtext.default'] = 'rm'
mpl.rcParams['mathtext.fallback_to_cm'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.grid.which']='both'
mpl.rcParams['grid.linewidth']= 0.5
mpl.rcParams['axes.xmargin']=0.05
mpl.rcParams['axes.ymargin']=0.05
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['figure.figsize'] = (16.18,10)
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['image.cmap'] = "viridis"
mpl.rcParams['figure.dpi'] = 75
mpl.rcParams['savefig.dpi'] = 150
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.viridis(np.linspace(0.1,0.9,10)))
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Graph-A : Dataset distribution
#///////////////////////////////////////////
figA_prod = pd.read_csv(DATADIR+"production-distribution.csv", index_col=None)
# %%--  Plot
bins2 = np.int((figA_prod["efficiency"].max()-figA_prod["efficiency"].min())/0.01)

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(x=figA_prod["efficiency"],bins=bins2,alpha=0.85,color = 'Navy')

ax.annotate(
    " Mean : %.2F \n Std : %.2F \n Min : %.2F \n Max : %.2F "%(figA_prod['efficiency'].mean(),figA_prod['efficiency'].std(),figA_prod['efficiency'].min(),figA_prod['efficiency'].max()),
    xy=(math.floor(figA_prod['efficiency'].min())*1.01,4500),
    xycoords='data',
    fontsize=15,
    color = "black",
    ha='left',
    )
ax.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax.set_axisbelow(True)
ax.set_xlabel("Cell efficiency [%]")
ax.set_ylabel("Counts")
ax.set_xlim(left=math.floor(figA_prod['efficiency'].min()),right=math.ceil(figA_prod['efficiency'].max()))
ax.set_ylim(bottom=0,top=6000)
plt.show()
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Graph-B : ML Comparison on dataset
#///////////////////////////////////////////
# %%--  Load Data
res_tab = []
for file in os.listdir(DATADIR+"CV-results//"):
    res_dic = load_obj(DATADIR+"CV-results//",file)
    N=int(file.split('-')[-1].split('.')[0])
    for key in res_dic:
        if key=='struc':    continue
        for rmse,r2 in zip(res_dic[key][1]['test_RMSE'],res_dic[key][1]['test_r2']):
            res_tab.append([key,N,rmse,r2,'Validation set'])
        for rmse,r2 in zip(res_dic[key][1]['train_RMSE'],res_dic[key][1]['train_r2']):
            res_tab.append([key,N,rmse,r2,'Training set'])
FigB_df = pd.DataFrame(res_tab)
FigB_df.columns = ['Model','Data set size','RMSE [%]','$R^2$ score','Set']
FigB_df = FigB_df.loc[FigB_df['Set']=='Validation set']
# %%-
# %%--  Plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,5))
palette = sns.color_palette("viridis_r", 4)
sns.lineplot(
    x='Data set size',
    y='RMSE [%]',
    hue='Model',
    style='Model',
    data=FigB_df,
    estimator='mean',
    ci=95,
    ax=ax1,
    palette=palette,
    err_style='band',
    markers=True,
    dashes=False,
    hue_order = ['SV','RF','AB','NN'],
    style_order = ['SV','RF','AB','NN'],
    legend=False,
    )
ax1.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax1.loglog()
ax1.set_xlim(left=0.8e2,right=1e6)
ax1.set_ylim(bottom=0.01,top=1)
ax1.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
ax1.annotate("(a)",xy=(0.9,0.04),xycoords='axes fraction', fontsize=14)

sns.lineplot(
    x='Data set size',
    y='$R^2$ score',
    hue='Model',
    style='Model',
    data=FigB_df,
    estimator='mean',
    ci=95,
    ax=ax2,
    palette=palette,
    err_style='band',
    markers=True,
    dashes=False,
    legend='brief',
    hue_order = ['SV','RF','AB','NN'],
    style_order = ['SV','RF','AB','NN'],
    )
ax2.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax2.semilogx()
ax2.set_xlim(left=0.8e2,right=1e6)
ax2.set_ylim(bottom=0,top=1)
ax2.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
fig.legend(bbox_to_anchor=(0.5, 1), loc=9, borderaxespad=0., ncol=5)
ax2.annotate("(b)",xy=(0.9,0.04),xycoords='axes fraction', fontsize=14)
ax2.get_legend().remove()
plt.tight_layout()
plt.show()
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Graph-C : ML training on dataset
#///////////////////////////////////////////
# %%--  Load Data
FC_prod_dict = load_obj(DATADIR,'NN_prod_2020-02-03-11-31_N-399738')
FigC_prod = {
    'Actual':FC_prod_dict['actual'].flatten(),
    'Predicted':FC_prod_dict['predicted'].flatten(),
    'R2':r2_score(FC_prod_dict['actual'].flatten(),FC_prod_dict['predicted'].flatten()),
    'RMSE': np.sqrt(mean_squared_error(FC_prod_dict['actual'].flatten(),FC_prod_dict['predicted'].flatten())),
}
# %%-
# %%--  Plot predicted vs actual with stastistic
fig, (ax2) = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
#   Prod
ax2.set_aspect('equal')
ax2.annotate("$R^2=$%.3f \nRMSE = %.3f"%(FigC_prod['R2'],FigC_prod['RMSE']),xy=(0.05,0.85),xycoords='axes fraction', fontsize=14)
ax2.set_title("Production data set", fontsize=16)
ax2.scatter(FigC_prod['Actual'],FigC_prod['Predicted'],marker=".",alpha=0.1,s=4,color = 'Navy')
ax2.set_xlabel('True efficiency [%]')
ax2.set_ylabel('Predicted efficiency [%]')
ax2.set_ylim(bottom=15.8, top=19.2)
ax2.set_xlim(left=15.8, right=19.2)
ax2.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax2.locator_params(nbins = 4)
plt.tight_layout()
plt.show()
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Graph-D : GA optimization
#///////////////////////////////////////////
# %%--  Load Data
FigD_log = load_obj(DATADIR, 'GA-100_2020-02-12-14-55')
FD_prod_dict = load_obj(DATADIR,'NN_prod_2020-02-03-11-31_N-399738')
FigD_prod = {
    'Actual':FD_prod_dict['actual'].flatten(),
    'Predicted':FD_prod_dict['predicted'].flatten(),
    'R2':r2_score(FD_prod_dict['actual'].flatten(),FD_prod_dict['predicted'].flatten()),
    'RMSE': np.sqrt(mean_squared_error(FD_prod_dict['actual'].flatten(),FD_prod_dict['predicted'].flatten())),
}
res_tab=[]
for g,a,p in zip(FigD_log['pop_gen'],FigD_log['pop_eff_act'],FigD_log['pop_eff_pred']):
    res_tab.append([g,'Predicted',p])
    res_tab.append([g,'True',a])
FigD_df = pd.DataFrame(res_tab)
FigD_df.columns = ['Generation','Model','Efficiency [%]']
max_tab=[]
for g in FigD_df['Generation'].unique():
    pred=FigD_df.loc[FigD_df['Generation']==g].loc[FigD_df['Model']=='Predicted']['Efficiency [%]'].max()
    true=FigD_df.loc[FigD_df['Generation']==g].loc[FigD_df['Model']=='True']['Efficiency [%]'].max()
    max_tab.append([g,pred,true])
FigD_max = pd.DataFrame(max_tab)
FigD_max.columns = ['Gen','Pred','True']
# %%-
# %%--  Plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
ax2min=17.75
ax2max=19.75
bins=np.int((np.max(FigD_prod['Actual'])-np.min(FigD_prod['Actual']))/0.01)

#   By generation
ax1.plot([0,np.max(FigD_log['gen'])+1],[FigD_prod['Actual'].max(),FigD_prod['Actual'].max()],'--',c='Navy')
palette = plt.cm.ocean(np.linspace(0.1,0.6,2))
# palette = sns.color_palette("ocean", 2)
sns.lineplot(
    x='Generation',
    y='Efficiency [%]',
    hue='Model',
    style='Model',
    data=FigD_df,
    estimator='mean',
    ci=None,
    ax=ax1,
    palette=palette,
    err_style='band',
    markers=True,
    dashes=False,
    hue_order = ['True','Predicted'],
    style_order = ['True','Predicted'],
    legend=False,
    lw=0.5,
    )
ax1.plot(FigD_max['Gen'],FigD_max['True'],c=palette[0], label='True population mean')
ax1.plot(FigD_max['Gen'],FigD_max['Pred'],c=palette[1], label='Predicted population mean')
ax1.annotate("(a)",xy=(0.9,0.05),xycoords='axes fraction', fontsize=14)
ax1.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax1.set_ylim(bottom=ax2min, top=ax2max)
ax1.set_xlim(left=0, right=np.max(FigD_log['gen'])+1)
ax1.locator_params(nbins = 5)

#   Compared to data
ax2.annotate("(b)",xy=(0.9,0.05),xycoords='axes fraction', fontsize=14)

ax2.plot([ax2min,ax2max],[ax2min,ax2max],'k--')
ax2.plot([FigD_prod['Actual'].max(),FigD_prod['Actual'].max()],[ax2min,ax2max],'--',c='Navy')
ax2.plot([np.max(FigD_log['pop_eff_act']),np.max(FigD_log['pop_eff_act'])],[ax2min,ax2max],'--',c='goldenrod')
ax2.scatter(FigD_log['pop_eff_act'],FigD_log['pop_eff_pred'],c=FigD_log['pop_gen'][::-1],marker="+",alpha=0.6,s=5)
ax2.set_xlabel('True efficiency [%]')
ax2.set_ylabel('Predicted efficiency [%]')
ax2.set_ylim(bottom=ax2min, top=ax2max)
ax2.set_xlim(left=ax2min, right=ax2max)
ax2.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax2.locator_params(nbins = 4)
plt.tight_layout()
plt.show()
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Graph-E : NEO-GA Loop
#///////////////////////////////////////////
# %%--  Load Data
FigE_log = load_obj(DATADIR,'LogObj_2020-02-11_16-40N10000_L5_NGA25_ValT_ClipFF_47param_tight')  #   Current used
res_tab=[]
loop_length = len(FigE_log['GA'][-2]['pop_gen'])
for l in FigE_log['Parameters']['loopRange']:
    for g,a,p in zip(FigE_log['GA'][l]['pop_gen'],FigE_log['GA'][l]['pop_eff_act'],FigE_log['GA'][l]['pop_eff_pred']):
        for act,pred in zip(a,p):
            res_tab.append([g,'Predicted',pred,l])
            res_tab.append([g,'True',act,l])
FigE_df = pd.DataFrame(res_tab)
FigE_df.columns = ['Generation','Model','Efficiency [%]','Loop #']
Nloop = len(FigE_log['Parameters']['loopRange'])
# %%-
# %%--  Plot
fig = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(nrows=1,
                       ncols=2*Nloop+1,
                       figure=fig,
                       width_ratios= [1,2]*(Nloop)+[1],
                       height_ratios=[1],
                       wspace=0, hspace=0.05,
                      )
color = plt.cm.viridis(np.linspace(0,0.8,Nloop+2))
color[0] = np.array(mpl.colors.to_rgba(mpl.colors.CSS4_COLORS['navy']))
ax = [None]*(2*Nloop +1)
features = [' Initial']+['Iteration #'+str(i+1) for i in range(Nloop)]+['Final']
y_min = 17
y_max = 20.5
x_min_dist=0
x_max_dist=1500
x_min_GA=0
x_max_GA=loop_length

#   Distributions
for l in range(2*Nloop+1):
    ax[l]=fig.add_subplot(gs[0,l])
    ax[l].set_ylim(bottom=y_min,top=y_max)

    if l%2==0:  #Distribution
        ax[l].set_xlim(x_min_dist,x_max_dist)
        ax[l].set_xticks([])
        if l!=0:
            ax[l].set_yticks([])
            ax[l].spines['left'].set_visible(False)
        ax[l].spines['right'].set_visible(False)
        ax[l].spines['top'].set_visible(False)
        eff = FigE_log['Dataset'][int(l/2)]['Efficiencies']
        bins = np.int((np.max(eff)-np.min(eff))/0.01)
        ax[l].hist(x=eff,bins=bins,alpha=0.85,orientation="horizontal",color=color[int(l/2)])
        for k in range(l+1):
            if k%2==0: ax[k].plot([x_min_dist,x_max_dist],[np.max(eff),np.max(eff)],'--',color=color[int(l/2)])
            if k%2==1: ax[k].plot([x_min_GA,x_max_GA],[np.max(eff),np.max(eff)],'--',color=color[int(l/2)])
        ax[l].annotate(
            r"$\bar{\eta}$=%.2F $\pm$ %.2F"%(np.mean(eff),np.std(eff)),
            xy=(x_max_dist/3,np.mean(eff)-0.6*np.std(eff)),
            xycoords='data',
            fontsize=14,
            color = color[int(l/2)],
            ha='left',
            rotation=-90,
            )
        ax[0].annotate(
            "%.2F "%(np.max(eff)),
            xy=(x_min_dist,np.max(eff)),
            xycoords='data',
            fontsize=14,
            color = color[int(l/2)],
            ha='right',
            rotation=0,
            )
        feature_ha = 'left' if l==0 else 'right'
        ax[l].annotate(
            features[int(l/2)],
            xy=(x_min_dist,y_max),
            xycoords='data',
            fontsize=14,
            color = color[int(l/2)],
            ha=feature_ha,
            rotation=0,
            )
    else:  #GA
        ax[l].set_xlim(x_min_GA,x_max_GA)
        ax[l].set_yticks([])
        ax[l].spines['left'].set_visible(False)
        ax[l].spines['right'].set_visible(False)
        ax[l].spines['top'].set_visible(False)
        locDF = FigE_df.loc[FigE_df['Loop #']==int((l-1)/2)]
        sns.lineplot(
            x='Generation',
            y='Efficiency [%]',
            style='Model',
            data= locDF,
            estimator='mean',
            ci=None,
            ax=ax[l],
            color=color[int((l+1)/2)],
            markers=True,
            err_style='band',
            dashes=False,
            style_order = ['Predicted','True'],
            legend=False,
            lw=1,
            )
        if l!=Nloop :ax[l].set_xlabel("")
        ax[l].locator_params(nbins = 4)

ax[0].locator_params(nbins = 3)
ax[0].set_ylabel('Efficiency [%]', labelpad=25)
plt.show()
# %%-
