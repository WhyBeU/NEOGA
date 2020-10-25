#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import datetime
import seaborn as sns
import math
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from math import sqrt, pi, exp, pow
import subprocess
from scipy.integrate import quad
from io import StringIO, BytesIO
import sys
import pyDOE2 as pyDOE
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate,cross_val_predict
import pickle
from math import sqrt, log10
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

DATADIR = ".\\data\\"
def save_obj(obj, folder, name ):
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(folder, name  ):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)
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
# %%--  Genetic algorithm
class geneticAlgorithm():
    def __init__(self):
        pass

    def evaluate(self,ind):
        """
        This is the evaluation function. The this function contains the estimator
        that was loaded beforehand. The population is then passed in the .predict()
        method of the estimator which returns the efficiency. The population
        consists of a set of recipes that were selected from the previous
        generation.
        """
        fitness = np.around(self.estimator.predict(ind.reshape(1, -1)), 5)
        return fitness

    def initES(self,icls, MIN, MAX, MEAN, bf, features, rand_dist):
        """
        I create a numpy array of recipes with the same function that was used to
        make the recipes for PC1D to evaluate.
        """
        if rand_dist=='uniform': a = np.array(object=[np.random.uniform(MIN[col].values.tolist()[0],MAX[col].values.tolist()[0]) for col in features])
        if rand_dist=='gaussian': a = np.array(object=[np.random.normal(MEAN[col].values.tolist()[0],np.sqrt(bf)*np.abs(MEAN[col].values.tolist()[0])) for col in features])
        ind = icls(a)
        return ind

    def cxTwoPointCopy(self,ind1, ind2):
        """Execute a two points crossover with copy on the input individuals. The
        copy is required because the slicing in numpy returns a view of the data,
        which leads to a self overwritting in the swap operation. It prevents
        ::

            >>> import numpy
            >>> a = numpy.array((1,2,3,4))
            >>> b = numpy.array((5.6.7.8))
            >>> a[1:3], b[1:3] = b[1:3], a[1:3]
            >>> print(a)
            [1 6 7 4]
            >>> print(b)
            [5 6 7 8]
        """
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

        return ind1, ind2

    def checkBounds(self,min, max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max[i]:
                            child[i] = max[i]
                        elif child[i] < min[i]:
                            child[i] = min[i]
                return offspring
            return wrapper
        return decorator

    def abc(self,model,NGEN,features,bf):
        standardPPC = pd.read_csv('standardPPC.csv',sep=',')
        df_boundaries = pd.read_csv("boundariesPPC.csv",sep=",")
        MIN = standardPPC.loc[:,features]*(1-bf)
        MAX = standardPPC.loc[:,features]*(1+bf)
        for key in features:
            MIN[key] = np.clip(a=MIN[key].values,a_min=df_boundaries.loc[0,key],a_max=df_boundaries.loc[1,key])
            MAX[key] = np.clip(a=MAX[key].values,a_min=df_boundaries.loc[0,key],a_max=df_boundaries.loc[1,key])
        MIN["bulkFactor"] = 1
        MAX["bulkfactor"] = 1
        self.estimator = model
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.initES, creator.Individual, MIN, MAX, features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.cxTwoPointCopy)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.2)
        toolbox.decorate("mate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.decorate("mutate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = tools.Logbook()  # number of generations
        CXPB = 0.6
        MUTPB = 0.10
        pop = toolbox.population(n=100)

        for g in range(NGEN):
            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))
            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            record = stats.compile(pop)
            logbook.record(gen=g, **record)
        listy = []
        for ind in pop:
            listy.append(list(ind))
        # df_GA = pd.DataFrame(listy)
        # df_GA.columns = features
        # DS = dataSim(path=r"C:\Users\z5236065\Desktop\tempPC1D1")
        # df0 = DS.GAresultschecker(df_GA)
        # actualMAX = max(df0.loc[:,"efficiency"])
        # actualAVG = np.mean(df0.loc[:,"efficiency"])
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_maxs = logbook.select("max")
        size_avgs = logbook.select("avg")
        size_std = logbook.select("std")
        return gen, fit_maxs, size_avgs, pop #actualMAX, actualAVG,
    #  Add something to prevent the algorithm from tweaking certain points.

    def indu(self,model,NGEN,features,bf):
        industryPPC = pd.read_csv('industryPPC.csv',sep=',')
        df_boundaries = pd.read_csv("boundariesPPC.csv",sep=",")
        MIN = industryPPC.loc[:,features]*(1-bf)
        MAX = industryPPC.loc[:,features]*(1+bf)
        for key in features:
            MIN[key] = np.clip(a=MIN[key].values,a_min=df_boundaries.loc[0,key],a_max=df_boundaries.loc[1,key])
            MAX[key] = np.clip(a=MAX[key].values,a_min=df_boundaries.loc[0,key],a_max=df_boundaries.loc[1,key])
        self.estimator = model
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.initES, creator.Individual, MIN, MAX, features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.cxTwoPointCopy)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.2)
        toolbox.decorate("mate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.decorate("mutate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = tools.Logbook()  # number of generations
        CXPB = 0.6
        MUTPB = 0.10
        pop = toolbox.population(n=100)

        for g in range(NGEN):
            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))
            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            record = stats.compile(pop)
            logbook.record(gen=g, **record)
        listy = []
        for ind in pop:
            listy.append(list(ind))
        # df_GA = pd.DataFrame(listy)
        # df_GA.columns = features
        # DS = dataSim(path=r"C:\Users\z5236065\Desktop\tempPC1D1")
        # df0 = DS.GAresultschecker(df_GA)
        # actualMAX = max(df0.loc[:,"efficiency"])
        # actualAVG = np.mean(df0.loc[:,"efficiency"])
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_maxs = logbook.select("max")
        size_avgs = logbook.select("avg")
        size_std = logbook.select("std")
        return gen, fit_maxs, size_avgs, pop #actualMAX, actualAVG,

    def optim(self,model,scaler_col,scaler,features,bf=0.5,NGEN=100, clip=True, PPCfile='industryPPC.csv',CXPB=0.6,MUTPB=0.1,popsize=100,SELECTSIZE=3,rand_dist='uniform'):
        industryPPC = pd.read_csv(PPCfile,sep=',')
        df_boundaries = pd.read_csv("boundariesPPC.csv",sep=",")
        MIN = industryPPC.loc[:,features]*(1-bf)
        MAX = industryPPC.loc[:,features]*(1+bf)
        MEAN = industryPPC.loc[:,features]
        for col in features:
            if clip: MIN[col] = np.clip(a=MIN[col].values,a_min=df_boundaries.loc[0,col],a_max=df_boundaries.loc[1,col])
            if clip: MAX[col] = np.clip(a=MAX[col].values,a_min=df_boundaries.loc[0,col],a_max=df_boundaries.loc[1,col])
            MIN[col]=scaler_col[col].transform(MIN[col].values.reshape(-1,1))
            MAX[col]=scaler_col[col].transform(MAX[col].values.reshape(-1,1))
            MEAN[col]=scaler_col[col].transform(MEAN[col].values.reshape(-1,1))
        self.estimator = model
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.initES, creator.Individual, MIN, MAX, MEAN, bf, features,rand_dist)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.cxTwoPointCopy)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.2)
        toolbox.decorate("mate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.decorate("mutate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.register("select", tools.selTournament, tournsize=SELECTSIZE)
        toolbox.register("evaluate", self.evaluate)
        stats = tools.Statistics(key=lambda ind: scaler.inverse_transform([ind.fitness.values]))
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = tools.Logbook()  # number of generations
        pop = toolbox.population(n=popsize)

        listy = []
        for g in range(NGEN):
            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))
            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            listy.append({'gen':g,'pop':[list(ind) for ind in pop],'eff':[toolbox.evaluate(ind) for ind in pop]})
            pop[:] = offspring
            record = stats.compile(pop)
            logbook.record(gen=g, **record)
        # df_GA = pd.DataFrame(listy)
        # df_GA.columns = features
        # DS = dataSim(path=r"C:\Users\z5236065\Desktop\tempPC1D1")
        # df0 = DS.GAresultschecker(df_GA)
        # actualMAX = max(df0.loc[:,"efficiency"])
        # actualAVG = np.mean(df0.loc[:,"efficiency"])
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_maxs = logbook.select("max")
        size_avgs = logbook.select("avg")
        size_std = logbook.select("std")
        return logbook,listy

    def simpleOptimization(self,model,NGEN,features,checkpoint):
        MIN = pd.read_csv("minPPC.csv",sep=",")
        MAX = pd.read_csv("maxPPC.csv",sep=",")
        self.estimator = model
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.initES, creator.Individual, MIN, MAX, features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.cxTwoPointCopy)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.2)
        toolbox.decorate("mate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.decorate("mutate", self.checkBounds(MIN.values[0].tolist(), MAX.values[0].tolist()))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = tools.Logbook()  # number of generations
        CXPB = 0.6
        MUTPB = 0.10
        pop = toolbox.population(n=100)
        counter = 0
        df_GA = pd.DataFrame(columns=features)
        df_GA.loc[0,:] = np.zeros((1,features.shape[0]))
        C = 0
        df_GA['counter'] = C
        for g in range(NGEN):
            # Select and clone the next generation individuals
            offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))
            # Apply crossover and mutation on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            record = stats.compile(pop)
            logbook.record(gen=g, **record)
            if counter is checkpoint:
                listy = []
                for ind in pop:
                    listy.append(list(ind))
                df0 = pd.DataFrame(listy)
                df0.columns = features
                df0.drop_duplicates(inplace=True)
                df0['counter'] = C
                C += 1
                df_GA = pd.concat([df_GA,df0],axis=0,ignore_index=True)
                counter = 0
            counter += 1
        return gen, fit_maxs, size_avgs, pop, df_GA

# %%-
# %%--  Logger
class Logger():
    """Write to a log file everything passed to print().

    Parameters
    ----------
    logfile : str
        path to the log file.

    Examples
    -------
    Define the logger
    >>> logger = Logger(logfile)

    Start logging sys.stdout
    >>> logger.open()

    End logging sys.stdout
    >>> logger.close()

    Attributes
    ----------
    terminal :
        local storage of original sys.stdout.
    log :
        file handler for logfile.

    """
    def __init__(self, logfile: str):
        self.terminal=None
        self.log = None
        self.logfile=logfile
    def __getattr__(self, attr):
            return getattr(self.terminal, attr)
    def write(self, message):
        """Overwrite write method to enable writing in both sys.stdout and log file."""
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
    def close(self):
        """Close log file and restore sys.stdout"""
        self.log.close()
        sys.stdout = self.terminal
    def open(self):
        """Open log file and save sys.stdout"""
        self.terminal = sys.stdout
        self.log = open(self.logfile, "a+")
        sys.stdout = self
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Machine learning on distributions
#///////////////////////////////////////////
# %%--  Load data and scale the target output
df_broad = pd.read_csv(DATADIR+"broad-distribution.csv", index_col=None)
df_prod = pd.read_csv(DATADIR+"production-distribution.csv", index_col=None)

scaler_broad = MinMaxScaler().fit(df_broad.loc[:,"efficiency"].values.reshape(-1,1))
df_broad['Eff_norm'] = scaler_broad.transform(df_broad['efficiency'].values.reshape(-1,1))
scaler_prod = MinMaxScaler().fit(df_prod.loc[:,"efficiency"].values.reshape(-1,1))
df_prod['Eff_norm'] = scaler_prod.transform(df_prod['efficiency'].values.reshape(-1,1))

X_col=['cellTh0', 'tau0', 'bulkFactor', 'sawDmg0', 'etchT', 'etch_t',
       'etchPrev', 'NaOH', 'propanol', 'textT', 'text_t', 'exhaust', 'HFconc',
       'HClconc', 'clean_t', 'rinse_t', 'Pconc', 'diffV', 'diffTdry', 'diffT2',
       'plasmaP', 'plasma_t', 'PECVD_t', 'PECVD_T', 'SiH4', 'NH4', 'dAl',
       'posAl', 'negAl', 'phiAl', 'pAl', 'muAl', 'vAl', 'dAg', 'posAg',
       'negAg', 'phiAg', 'pAg', 'muAg', 'vAg', 'wAg0', 'pitchAg', 'O2', 'N2',
       'fireV', 'fireTdry', 'fireT2']
y_col='Eff_norm'

scale_broad_col ={}
scale_prod_col ={}
for col in X_col:
    column = df_broad[col]
    if column.mean()>1e5 and column.min()>0: df_broad[col]=np.log10(column)
    scale_broad_col[col]=StandardScaler().fit(column.values.reshape(-1,1))
    df_broad[col] = scale_broad_col[col].transform(column.values.reshape(-1,1))
    column = df_prod[col]
    if column.mean()>1e5 and column.min()>0: df_prod[col]=np.log10(column)
    scale_prod_col[col]=StandardScaler().fit(column.values.reshape(-1,1))
    df_prod[col] = scale_prod_col[col].transform(column.values.reshape(-1,1))

# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Cross-Validation on ML algorithms
#///////////////////////////////////////////
# %%--  Define algorithms and setting
Nrange=np.logspace(2, np.log10(len(df_broad)),50)
sca_b_std = scaler_broad.scale_[0]
sca_b_min = scaler_broad.data_min_[0]
sca_b_max = scaler_broad.data_max_[0]
Ncv=5

# %%-
# %%--  Perform cross-validation training
for N in Nrange:
    N=int(N)
    print('*'*100)
    print(N)
    df_ML = df_broad.sample(N).copy(deep=True)
    MLpipeline = {
        'struc':['model','CV_Results','Mean_RMSE','Std_RMSE'],
        'RF':[RandomForestRegressor(n_estimators=150, bootstrap=True)],
        'AB':[AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=150, learning_rate=0.9, loss='square')],
        'SV':[SVR(kernel='rbf', C=1, epsilon = 0.1, gamma='scale')],
        'NN':[MLPRegressor(alpha=1e-5, activation='logistic',tol=1e-6,hidden_layer_sizes=(300,150,75),solver='adam',epsilon=1e-9,beta_1=0.9,learning_rate='adaptive',learning_rate_init=0.001, max_iter=300,)],
        'LR':[Lasso()]
    }
    filename = "CV-"+str(Ncv)+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_N-"+str(N)
    X=df_ML[X_col]
    y=df_ML[y_col]
    for key in MLpipeline:
        if key=='struc':    continue
        print('Cross validation of model :',key)
        ML=MLpipeline[key]
        res = cross_validate(ML[0],X,y, cv = Ncv, scoring=('r2','neg_mean_squared_error'),n_jobs=-1,verbose=0,return_train_score=True)
        res['test_RMSE']=(sca_b_max-sca_b_min)*np.sqrt(-res['test_neg_mean_squared_error'])
        res['train_RMSE']=(sca_b_max-sca_b_min)*np.sqrt(-res['train_neg_mean_squared_error'])
        ML.append(res)
        print(res['test_RMSE'].mean(),"+/-",res['test_RMSE'].std())
        print(res['train_RMSE'].mean(),"+/-",res['train_RMSE'].std())
        print(res['test_r2'].mean(),"+/-",res['test_r2'].std())
        print('-'*50)

    save_obj(MLpipeline,DATADIR+"CV-results\\",filename)
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Cross-Validation on ML algorithms - Prod
#///////////////////////////////////////////
# %%--  Define algorithms and setting
Nrange=np.logspace(2, np.log10(len(df_prod)),50)
sca_p_std = scaler_prod.scale_[0]
sca_p_min = scaler_prod.data_min_[0]
sca_p_max = scaler_prod.data_max_[0]
Ncv=5

# %%-
# %%--  Perform cross-validation training
for N in Nrange:
    N=int(N)
    print('*'*100)
    print(N)
    df_ML = df_prod.sample(N).copy(deep=True)
    MLpipeline = {
        'struc':['model','CV_Results','Mean_RMSE','Std_RMSE'],
        'RF':[RandomForestRegressor(n_estimators=150, bootstrap=True)],
        'AB':[AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=150, learning_rate=0.9, loss='square')],
        'SV':[SVR(kernel='rbf', C=1, epsilon = 0.1, gamma='scale')],
        'NN':[MLPRegressor(alpha=1e-5, activation='logistic',tol=1e-6,hidden_layer_sizes=(300,150,75),solver='adam',epsilon=1e-9,beta_1=0.9,learning_rate='adaptive',learning_rate_init=0.001, max_iter=300,)],
        'LR':[Lasso()]
    }
    filename = "CV-prod-"+str(Ncv)+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_N-"+str(N)
    X=df_ML[X_col]
    y=df_ML[y_col]
    for key in MLpipeline:
        if key=='struc':    continue
        print('Cross validation of model :',key)
        ML=MLpipeline[key]
        res = cross_validate(ML[0],X,y, cv = Ncv, scoring=('r2','neg_mean_squared_error'),n_jobs=-1,verbose=0,return_train_score=True)
        res['test_RMSE']=(sca_p_max-sca_p_min)*np.sqrt(-res['test_neg_mean_squared_error'])
        res['train_RMSE']=(sca_p_max-sca_p_min)*np.sqrt(-res['train_neg_mean_squared_error'])
        ML.append(res)
        print(res['test_RMSE'].mean(),"+/-",res['test_RMSE'].std())
        print(res['train_RMSE'].mean(),"+/-",res['train_RMSE'].std())
        print(res['test_r2'].mean(),"+/-",res['test_r2'].std())
        print('-'*50)

    save_obj(MLpipeline,DATADIR+"CV-results\\",filename)
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Neural Network training on Narrow dataset
#///////////////////////////////////////////
# %%--  Define algorithm
ML_NN = {
    'model':MLPRegressor(alpha=1e-5, activation='logistic',tol=1e-6,hidden_layer_sizes=(300,150,75),solver='adam',epsilon=1e-9,beta_1=0.9,learning_rate='adaptive',learning_rate_init=0.001, max_iter=300,verbose=2),
    }
N=len(df_prod)-1
# N=1000
sca_p_std = scaler_prod.scale_[0]
sca_p_min = scaler_prod.data_min_[0]
sca_p_max = scaler_prod.data_max_[0]
df_ML = df_prod.sample(N).copy(deep=True)
filename = "NN_prod"+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_N-"+str(N)
# %%-
# %%--  Train NN
df_train, df_test = train_test_split(df_ML, test_size = 0.2)
X_train, y_train = df_train[X_col],df_train[y_col]
X_test, y_test = df_test[X_col],df_test[y_col]

ML_NN['model'].fit(X_train, y_train)

ML_NN['train_MSE']=(sca_p_max-sca_p_min)*np.sqrt(mean_squared_error(y_train,ML_NN['model'].predict(X_train)))
ML_NN['train_R2']= r2_score(y_train,ML_NN['model'].predict(X_train))
ML_NN['test_MSE']= (sca_p_max-sca_p_min)*np.sqrt(mean_squared_error(y_test,ML_NN['model'].predict(X_test)))
ML_NN['test_R2']= r2_score(y_test,ML_NN['model'].predict(X_test))
ML_NN['actual'] = scaler_prod.inverse_transform(y_test.values.reshape(-1,1))
ML_NN['predicted'] = scaler_prod.inverse_transform(ML_NN['model'].predict(X_test).reshape(-1,1))


print(ML_NN['model'])
print('Test MSE',ML_NN['test_MSE'])
print('Train MSE',ML_NN['train_MSE'])
print('R2',ML_NN['test_R2'])
print('-'*100)


save_obj(ML_NN,DATADIR,filename)
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Neural Network training on Broad dataset
#///////////////////////////////////////////
# %%--  Define algorithm
ML_NN = {
    'model':MLPRegressor(alpha=1e-5, activation='logistic',tol=1e-6,hidden_layer_sizes=(300,150,75),solver='adam',epsilon=1e-9,beta_1=0.9,learning_rate='adaptive',learning_rate_init=0.001, max_iter=300,verbose=2),
    }
N=len(df_broad)-1
# N=1000
sca_b_std = scaler_broad.scale_[0]
sca_b_min = scaler_broad.data_min_[0]
sca_b_max = scaler_broad.data_max_[0]
df_ML = df_broad.sample(N).copy(deep=True)
filename = "NN_broad"+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_N-"+str(N)
# %%-
# %%--  Train NN
df_train, df_test = train_test_split(df_ML, test_size = 0.2)
X_train, y_train = df_train[X_col],df_train[y_col]
X_test, y_test = df_test[X_col],df_test[y_col]

ML_NN['model'].fit(X_train, y_train)

ML_NN['train_MSE']=(sca_b_max-sca_b_min)*np.sqrt(mean_squared_error(y_train,ML_NN['model'].predict(X_train)))
ML_NN['train_R2']= r2_score(y_train,ML_NN['model'].predict(X_train))
ML_NN['test_MSE']= (sca_b_max-sca_b_min)*np.sqrt(mean_squared_error(y_test,ML_NN['model'].predict(X_test)))
ML_NN['test_R2']= r2_score(y_test,ML_NN['model'].predict(X_test))
ML_NN['actual'] = scaler_broad.inverse_transform(y_test.values.reshape(-1,1))
ML_NN['predicted'] = scaler_broad.inverse_transform(ML_NN['model'].predict(X_test).reshape(-1,1))


print(ML_NN['model'])
print('Test MSE',ML_NN['test_MSE'])
print('Train MSE',ML_NN['train_MSE'])
print('R2',ML_NN['test_R2'])
print('-'*100)


save_obj(ML_NN,DATADIR,filename)
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    NN on narrow dataset full train
#///////////////////////////////////////////
# %%--  Define algorithm
ML_NN = {
    'model':MLPRegressor(alpha=1e-5, activation='logistic',tol=1e-6,hidden_layer_sizes=(300,150,75),solver='adam',epsilon=1e-9,beta_1=0.9,learning_rate='adaptive',learning_rate_init=0.001, max_iter=300,verbose=2),
    }
N=len(df_prod)-1
# N=None
sca_p_std = scaler_prod.scale_[0]
sca_p_min = scaler_prod.data_min_[0]
sca_p_max = scaler_prod.data_max_[0]
df_ML = df_prod.sample(N).copy(deep=True)
filename = "NN_prod_fulltrain"+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_N-"+str(N)
# %%-
# %%--  Train NN
df_train = df_ML
X_train, y_train = df_train[X_col],df_train[y_col]

ML_NN['model'].fit(X_train, y_train)

ML_NN['train_MSE']=(sca_p_max-sca_p_min)*np.sqrt(mean_squared_error(y_train,ML_NN['model'].predict(X_train)))
ML_NN['train_R2']= r2_score(y_train,ML_NN['model'].predict(X_train))

print(ML_NN['model'])
print('Train MSE',ML_NN['train_MSE'])
print('Train R2',ML_NN['train_R2'])
print('-'*100)


save_obj(ML_NN,DATADIR,filename)
# %%-
# %%--  Load NN
ANN = load_obj(DATADIR,'NN_prod_fulltrain_2020-02-07-09-37_N-399738')['model']
ANN = ML_NN['model']
features = X_col
# %%-
# %%--  Genetic Algorithm Optimizer
GA = geneticAlgorithm()
print([""",
    model=ANN,
    scaler_col=scale_prod_col,
    scaler=scaler_prod,
    features=X_col,
    bf=0.15,
    NGEN=100,
    clip=True,
    PPCfile='industryPPC.csv',
    CXPB=0.6,
    MUTPB=0.01,
    popsize=100,
    rand_dist='gaussian',
    SELECTSIZE=5,
    """]
    )
log,listy=GA.optim(
    model=ANN,
    scaler_col=scale_prod_col,
    scaler=scaler_prod,
    features=X_col,
    bf=0.15,
    NGEN=100,
    clip=False,
    PPCfile='industryPPC.csv',
    CXPB=0.5,
    MUTPB=0.15,
    popsize=100,
    rand_dist='gaussian',
    SELECTSIZE=20,
)
log_res={
    'gen':[],
    'avg':[],
    'std':[],
    'min':[],
    'max':[],
    'pop_gen':[],
    'pop_eff_act':[],
    'pop_eff_pred':[],
}
for GEN in log:
    log_res['gen'].append(GEN['gen']+1)
    log_res['avg'].append(GEN['avg'])
    log_res['std'].append(GEN['std'])
    log_res['min'].append(GEN['min'])
    log_res['max'].append(GEN['max'])
# %%-
# %%-- Predict efficiency of population generation
check_gen=range(len(log_res['gen']))
for p in listy:
    if p['gen'] not in check_gen: continue
    df_GA = pd.DataFrame(p['pop'])
    df_GA.columns = X_col
    for col in X_col: df_GA[col]=scale_prod_col[col].inverse_transform(df_GA[col].values.reshape(-1,1))
    DS = dataSim(path=r"PC1Dfiles")
    df0 = DS.GAresultschecker(df_GA)
    if len(df0)!=len(df_GA):
        continue
    for pred, act in zip(df0['efficiency'].values.flatten(),scaler_prod.inverse_transform(p['eff']).flatten()):
        log_res['pop_gen'].append(p['gen']+1)
        log_res['pop_eff_act'].append(act)
        log_res['pop_eff_pred'].append(pred)
# df_GA = pd.DataFrame(listy)
# df_GA.columns = features
# DS = dataSim(path=r"C:\Users\z5236065\Desktop\tempPC1D1")
# df0 = DS.GAresultschecker(df_GA)
# actualMAX = max(df0.loc[:,"efficiency"])
# actualAVG = np.mean(df0.loc[:,"efficiency"])
# %%-
# %%-- Save algorithm
filename = "GA-"+str(len(log_res['gen']))+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
save_obj(log_res,DATADIR,filename)
print(filename)
print('Max predic efficiency', np.max(log_res['pop_eff_pred']))
print('Max actual efficiency', np.max(log_res['pop_eff_act']))
print("--"*50)

# %%-
# %%--  plot 1
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(log_res['gen'], log_res['max'], label="GA maximum",c='C0')
ax.plot(log_res['gen'], log_res['min'], label="GA minimum",c='C4')
ax.plot(log_res['gen'], log_res['avg'], label="GA average",c='C8')
ax.set_xlabel("Generation")
ax.set_ylabel("Cell Efficiency [%]")
ax.legend()
# %%-
# %%--  plot 2
fig, ax = plt.subplots(figsize=(5,5))
cdict = {}
for g,c in zip(np.unique(log_res['pop_gen']),plt.cm.viridis(np.linspace(0.1,0.9,len(np.unique(log_res['pop_gen']))))):
    cdict[g]="Gen %d"%(g)
    iix = np.where(log_res['pop_gen'] == g)[0]
    first=True
    for ix in iix:
        if first:
            ax.scatter(log_res['pop_eff_act'][ix],log_res['pop_eff_pred'][ix],c=c,label=cdict[g])
            first=False
        else:
            ax.scatter(log_res['pop_eff_act'][ix],log_res['pop_eff_pred'][ix],c=c)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.legend()
# %%-
# %%--  Genetic Algorithm Optimizer - hyper-parameter validation
parameter = ['uniform']
print('distribution')
for x in parameter: #Save best for now
    GA = geneticAlgorithm()
    log,listy=GA.optim(
        model=ANN,
        scaler_col=scale_prod_col,
        scaler=scaler_prod,
        features=X_col,
        bf=0.1,
        NGEN=100,
        clip=False,
        PPCfile='industryPPC.csv',
        CXPB=0.6,
        MUTPB=0.1,
        popsize=100,
        rand_dist='gaussian',
        SELECTSIZE=50,
        )
    log_res={
        'gen':[],
        'avg':[],
        'std':[],
        'min':[],
        'max':[],
        'pop_gen':[],
        'pop_eff_act':[],
        'pop_eff_pred':[],
    }
    for GEN in log:
        log_res['gen'].append(GEN['gen']+1)
        log_res['avg'].append(GEN['avg'])
        log_res['std'].append(GEN['std'])
        log_res['min'].append(GEN['min'])
        log_res['max'].append(GEN['max'])

    check_gen=[24]
    for p in listy:
        if p['gen'] not in check_gen: continue
        df_GA = pd.DataFrame(p['pop'])
        df_GA.columns = X_col
        for col in X_col: df_GA[col]=scale_prod_col[col].inverse_transform(df_GA[col].values.reshape(-1,1))
        DS = dataSim(path=r"PC1Dfiles")
        df0 = DS.GAresultschecker(df_GA)
        if len(df0)!=len(df_GA):
            continue
        for pred, act in zip(df0['efficiency'].values.flatten(),scaler_prod.inverse_transform(p['eff']).flatten()):
            log_res['pop_gen'].append(p['gen']+1)
            log_res['pop_eff_act'].append(act)
            log_res['pop_eff_pred'].append(pred)

    print('GA-parameters', x, 'Max efficiency', np.max(log_res['pop_eff_act']))

# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    GA-Loop on Dataset
#   Please note this section is not available
#   as the python + PC1D simulation is non-shareable IP
#///////////////////////////////////////////
# %%--  Loop parameter
name ="N10000_L10_NGA20_ClipTT_47p_bf5_bfGA10_ContinousTraining"

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
save=True
logObj = {}
logObj['Parameters']={
    "name":name,
    "timestamp":timestamp,
    "loopRange":np.arange(0,10,1),
    "Ndata":10000,
    "bf":0.05,
    "clip":True,
    "Initial PPC file":"industryPPC.csv",
    "NNmodel": MLPRegressor(alpha=1e-5, activation='logistic',tol=1e-6,
                            hidden_layer_sizes=(300,150,75),solver='adam',epsilon=1e-9,
                            beta_1=0.9,learning_rate='adaptive',learning_rate_init=0.001,
                            max_iter=300,verbose=0, warm_start=False),
    "ValidFrac":0.01,
    "bf_GA":0.1,
    "N_GA":20,
    'clip_GA':True,
    'CXPB_GA':0.5,
    'MUTPB_GA':0.15,
    'popsize_GA':100,
    'select_GA':20,
    'validate_GA':True,
    'X_col':['etchT', 'etch_t',  'cellTh0','tau0', 'bulkFactor','sawDmg0',
           'etchPrev', 'NaOH', 'propanol', 'textT', 'text_t', 'exhaust', 'HFconc',
           'HClconc', 'clean_t', 'rinse_t', 'Pconc', 'diffV', 'diffTdry', 'diffT2',
           'plasmaP', 'plasma_t', 'PECVD_t', 'PECVD_T', 'SiH4', 'NH4', 'dAl',
           'posAl', 'negAl', 'phiAl', 'pAl', 'muAl', 'vAl', 'dAg', 'posAg',
           'negAg', 'phiAg', 'pAg', 'muAg', 'vAg', 'wAg0', 'pitchAg', 'O2', 'N2',
           'fireV', 'fireTdry', 'fireT2'],
    'y_col':'Eff_norm',
}
X_col=logObj['Parameters']['X_col']
y_col=logObj['Parameters']['y_col']
logObj['Timelog']={'Dataset':[],'Training':[],'GA':[]}
logObj['Dataset']=[{'File':logObj['Parameters']['Initial PPC file']}]
logObj['GA']=[{}]
logObj['Training']=[{}]
if save: logger = Logger(DATADIR+"loop\\log\\"+timestamp+"_"+name+".txt")
if save: logger.open()

title = "HYPERPARAMETERS"
print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
toprint = logObj['Parameters']
for k in toprint:
    print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
print("\n")

# %%-
# %%--  Execute loop
df_VPL= None
for l in logObj['Parameters']['loopRange']:
    title = "LOOP #"+str(l+1)
    print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))

    #  <subcell>    Generate data set
    print("-"*5,"Dataset Generation")
    print("Started at {0}".format(datetime.datetime.now().strftime("%m-%d   %H:%M:%S")))
    start_time = datetime.datetime.now()

    #   Generate dataset from VPL
    ''' Not available'''

    if  df_VPL is None:
        df_VPL = df_VPL_temp.copy(deep=True)
    else:
        df_VPL = pd.concat([df_VPL_temp,df_VPL])
    logObj['Dataset'][l]['Mean']=df_VPL_temp['efficiency'].mean()
    logObj['Dataset'][l]['Std']=df_VPL_temp['efficiency'].std()
    logObj['Dataset'][l]['Min']=df_VPL_temp['efficiency'].min()
    logObj['Dataset'][l]['Max']=df_VPL_temp['efficiency'].max()
    toprint=logObj['Dataset'][l]
    for k in toprint:
        print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
    logObj['Dataset'][l]['Efficiencies']=df_VPL_temp['efficiency']

    #   Normalize generated dataset
    df_VPL_NN = df_VPL.copy(deep=True)
    scaler_VPL = MinMaxScaler().fit(df_VPL_NN.loc[:,"efficiency"].values.reshape(-1,1))
    df_VPL_NN['Eff_norm'] = scaler_VPL.transform(df_VPL_NN['efficiency'].values.reshape(-1,1))
    scale_VPL_col ={}
    for col in X_col:
        column = df_VPL_NN[col]
        if column.mean()>1e5 and column.min()>0: df_VPL_NN[col]=np.log10(column)
        scale_VPL_col[col]=StandardScaler().fit(column.values.reshape(-1,1))
        df_VPL_NN[col] = scale_VPL_col[col].transform(column.values.reshape(-1,1))
    logObj['Dataset'][l]['scalerEff'] =  scaler_VPL
    logObj['Dataset'][l]['scalerCol'] = scale_VPL_col

    logObj['Timelog']['Dataset'].append(datetime.datetime.now()-start_time)
    print("Finished in {0}".format(datetime.datetime.now()-start_time))
    print("\n")
    #  </subcell>
    #  <subcell>    Train NN
    print("-"*5,"NN Training")
    print("Started at {0}".format(datetime.datetime.now().strftime("%m-%d   %H:%M:%S")))
    start_time = datetime.datetime.now()

    #   NN training
    NN = logObj['Parameters']['NNmodel']
    df_train, df_test = train_test_split(df_VPL_NN, test_size = logObj['Parameters']['ValidFrac'])
    X_train, y_train = df_train[X_col],df_train[y_col]
    X_test, y_test = df_test[X_col],df_test[y_col]
    NN.fit(X_train,y_train)

    #   Log results
    sca_min = scaler_VPL.data_min_[0]
    sca_max = scaler_VPL.data_max_[0]

    logObj['Training'][l]['Train R2']= r2_score(y_train,NN.predict(X_train))
    logObj['Training'][l]['Train RMSE']= (sca_max-sca_min)*np.sqrt(mean_squared_error(y_train,NN.predict(X_train)))
    logObj['Training'][l]['Valid R2']= r2_score(y_test,NN.predict(X_test))
    logObj['Training'][l]['Valid RMSE']= (sca_max-sca_min)*np.sqrt(mean_squared_error(y_test,NN.predict(X_test)))
    toprint = logObj['Training'][l]
    for k in toprint:
        print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
    logObj['Training'][l]['NN']= NN
    logObj['Training'][l]['lossCurve']= NN.loss_curve_

    logObj['Timelog']['Training'].append(datetime.datetime.now()-start_time)
    print("Finished in {0}".format(datetime.datetime.now()-start_time))
    print("\n")
    #  </subcell>
    #  <subcell>    GA optimizer
    print("-"*5,"GA")
    print("Started at {0}".format(datetime.datetime.now().strftime("%m-%d   %H:%M:%S")))
    start_time = datetime.datetime.now()

    #   Optimize
    GA = geneticAlgorithm()
    logGA,listy=GA.optim(model=NN, scaler_col=scale_VPL_col, scaler=scaler_VPL, features=X_col,
        bf=logObj['Parameters']['bf_GA'], NGEN=logObj['Parameters']['N_GA'], clip=logObj['Parameters']['clip_GA'],
        PPCfile=logObj['Dataset'][l]['File'], CXPB=logObj['Parameters']['CXPB_GA'], MUTPB=logObj['Parameters']['MUTPB_GA'],
        popsize=logObj['Parameters']['popsize_GA'], rand_dist='gaussian',  SELECTSIZE=logObj['Parameters']['select_GA'],
    )
    logObj['GA'][l]['Avg']=[GEN['avg'] for GEN in logGA]
    logObj['GA'][l]['Std']=[GEN['std'] for GEN in logGA]
    logObj['GA'][l]['Min']=[GEN['min'] for GEN in logGA]
    logObj['GA'][l]['Max']=[GEN['max'] for GEN in logGA]

    #   Verify
    logObj['GA'][l]['pop_gen']=[]
    logObj['GA'][l]['pop_eff_act']=[]
    logObj['GA'][l]['pop_eff_pred']=[]
    for p in listy:
        if not logObj['Parameters']['validate_GA']: continue
        df_GA = pd.DataFrame(p['pop'])
        df_GA.columns = X_col
        for col in X_col: df_GA[col]=scale_VPL_col[col].inverse_transform(df_GA[col].values.reshape(-1,1))
        DS = dataSim(path=r"PC1Dfiles")
        df0 = DS.GAresultschecker(df_GA)
        if len(df0)!=len(df_GA): continue
        logObj['GA'][l]['pop_gen'].append(p['gen']+1)
        logObj['GA'][l]['pop_eff_pred'].append([pred for pred in scaler_VPL.inverse_transform(p['eff']).flatten()])
        logObj['GA'][l]['pop_eff_act'].append([act for act in df0['efficiency'].values.flatten()])

    #   Select new recipe
    pop_last = listy[-1]
    pop_last['eff_array'] = [eff for eff in scaler_VPL.inverse_transform(pop_last['eff']).flatten()]
    indices = np.argsort(pop_last['eff_array'])[-logObj['Parameters']['select_GA']:][::-1]
    df_pop = pd.DataFrame(p['pop']).ix[indices]
    df_pop.columns = X_col
    df_mean = pd.read_csv(logObj['Dataset'][l]['File'],sep=',')
    for col in X_col:
        df_pop[col]=scale_VPL_col[col].inverse_transform(df_pop[col].values.reshape(-1,1))
        df_mean[col]=[df_pop[col].mean()]
    if save:
        filename = DATADIR+"loop\\recipe\\"+timestamp+name+"_PPC_loop"+str(l)+".csv"
        df_mean.to_csv(filename,encoding='utf-8', index=False)
        logObj['Dataset'].append({'File':filename})
    else:
        df_mean.to_csv('tempPPC.csv',encoding='utf-8', index=False)
        logObj['Dataset'].append({'File':'tempPPC.csv'})
    logObj['Training'].append({})
    logObj['GA'].append({})
    toprint = {
        'Avg efficiency(selected) [GA]': np.mean(pop_last['eff_array']) ,
        'Avg efficiency(selected) [PC1D]': np.mean(logObj['GA'][l]['pop_eff_act'][-1]) if logObj['Parameters']['validate_GA'] else "NA",
    }
    for k in toprint:
        print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])

    logObj['Timelog']['GA'].append(datetime.datetime.now()-start_time)
    print("Finished in {0}".format(datetime.datetime.now()-start_time))
    print("\n")
    #  </subcell>

#  <subcell>    Final Dataset Generation
print("-"*5," Final Dataset Generation")
print("Started at {0}".format(datetime.datetime.now().strftime("%m-%d   %H:%M:%S")))
start_time = datetime.datetime.now()

#   Generate final dataset from VPL
''' Not Available '''
logObj['Dataset'][l+1]['Mean']=df_VPLf['efficiency'].mean()
logObj['Dataset'][l+1]['Std']=df_VPLf['efficiency'].std()
logObj['Dataset'][l+1]['Min']=df_VPLf['efficiency'].min()
logObj['Dataset'][l+1]['Max']=df_VPLf['efficiency'].max()
toprint=logObj['Dataset'][l+1]
for k in toprint:
    print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
logObj['Dataset'][l+1]['Efficiencies']=df_VPLf['efficiency']

print("Finished in {0}".format(datetime.datetime.now()-start_time))
print("\n")
#  </subcell>
save_obj(logObj,DATADIR+"loop\\","LogObj_"+timestamp+"_"+name)
if save: logger.close()
# %%-

# %%--  Plot distributions
fig, ax = plt.subplots(figsize=(8,8))
color = plt.cm.viridis(np.linspace(0.1,0.8,len(logObj['Parameters']['loopRange'])+1))

for l in range(len(logObj['Parameters']['loopRange'])+1):
    eff = logObj['Dataset'][l]['Efficiencies']
    bins = np.int((np.max(eff)-np.min(eff))/0.02)
    ax.hist(x=eff,bins=bins,alpha=0.85,label="Loop %s; Mean: %.2F; Max: %.2F"%(l,np.mean(eff),np.max(eff)),color=color[l])

ax.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax.set_axisbelow(True)
ax.set_xlabel("Cell efficiency [%]")
ax.set_ylabel("Counts")
ax.set_xlim(left=15,right=20)
ax.set_ylim(bottom=0)
ax.legend()

# %%-
# %%--  Plot learning curves
fig, ax = plt.subplots(figsize=(8,8))
res_tab=[]
loop_length = len(logObj['GA'][-2]['pop_gen'])
for l in logObj['Parameters']['loopRange']:
    for g,a,p in zip(logObj['GA'][l]['pop_gen'],logObj['GA'][l]['pop_eff_act'],logObj['GA'][l]['pop_eff_pred']):
        for act,pred in zip(a,p):
            res_tab.append([g+loop_length*l,'Predicted',pred,l])
            res_tab.append([g+loop_length*l,'True',act,l])
Fig_df = pd.DataFrame(res_tab)
Fig_df.columns = ['Generation','Model','Efficiency [%]','Loop #']

sns.lineplot(
    x='Generation',
    y='Efficiency [%]',
    hue='Loop #',
    style='Model',
    data=Fig_df,
    estimator='mean',
    ci=None,
    ax=ax,
    palette=sns.color_palette('viridis', n_colors=len(logObj['Parameters']['loopRange'])),
    err_style='band',
    markers=True,
    dashes=False,
    # hue_order = ['True','Predicted'],
    # style_order = ['Validation set','Training set'],
    # style_order = ['True','Predicted'],
    # legend=False,
    lw=0.5,
    )

ax.tick_params(axis='both',direction='in', which='both',top=True,right=True)
ax.set_axisbelow(True)
ax.set_xlabel("Generation")
ax.set_ylabel("Cell efficiency [%]")
# ax.set_xlim(left=0,right=100)
ax.set_ylim(bottom=17.5)
# %%-
