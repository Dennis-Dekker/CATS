# packages
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from statistics import mean
import math

# enter raw data
baseline = np.array([0.636, 0.476, 0.684, 0.684, 0.789])
baseline_2 = np.array([0.636, 0.476, 0.684, 0.684, 0.789, 0.773, 0.619, 0.684, 0.684, 0.684, 0.636, 0.619, 0.789, 0.684, 0.789])
anova = np.array([0.864, 0.667, 0.842, 0.895, 0.842])
recursive = np.array([0.75, 0.85, 0.80, 0.75, 0.9])
recursive_2 = np.array([0.8,0.85,0.85,0.75,0.8,0.9,0.7,0.9,0.75,0.7,0.7,0.75,0.85,0.8,0.85])
l1_regularization = np.array([0.773, 0.762, 0.842, 0.842, 0.842])
anova_2 = np.array([0.818, 0.762, 1.0, 0.842, 0.842, 0.773, 0.667, 0.895, 0.842, 0.737, 0.773, 0.667, 0.895, 0.789, 0.737])
l1_regularization_2 = np.array([0.727, 0.762, 0.842, 0.842, 0.842, 0.727, 0.762, 0.947, 0.842, 0.842, 0.773, 0.667, 0.947, 0.895, 0.842])

# calculate the average
baseline_mean = np.mean(baseline)
baseline_2_mean = np.mean(baseline_2)
anova_mean = np.mean(anova)
recursive_mean = np.mean(recursive)
recursive_2_mean = np.mean(recursive_2)
l1_regularization_mean = np.mean(l1_regularization)
anova_2_mean = np.mean(anova_2)
l1_regularization_2_mean = np.mean(l1_regularization_2)

# calculate the standard deviation
baseline_std = np.std(baseline)
baseline_2_std = np.std(baseline_2)
anova_std = np.std(anova)
recursive_std = np.std(recursive)
recursive_2_std = np.std(recursive_2)
l1_regularization_std = np.std(l1_regularization)
anova_2_std = np.std(anova_2)
l1_regularization_2_std = np.std(l1_regularization_2)

# calculate the standard error
baseline_se = stats.sem(baseline)
baseline_2_se = stats.sem(baseline_2)
anova_se = stats.sem(anova)
recursive_se = stats.sem(recursive)
recursive_2_se = stats.sem(recursive_2)
l1_regularization_se = stats.sem(l1_regularization)
anova_2_se = stats.sem(anova_2)
l1_regularization_2_se = stats.sem(l1_regularization_2)


menMeans   = (baseline_2_mean, anova_2_mean, recursive_2_mean, l1_regularization_2_mean)
menStd     = (baseline_2_se, anova_2_se, recursive_2_se, l1_regularization_2_se)

print(menStd)
ind  = np.arange(4)    # the x locations for the groups
width= 0.7
labels = ('Baseline', 'ANOVA', 'RFS-SVM', 'L1')

# Pull the formatting out here
bar_kwargs = {'width':width,'color':['r', 'blue','g','y'],'linewidth':2,'zorder':5,"alpha":0.5}
err_kwargs = {'zorder':0,'fmt':"none",'linewidth':1,'ecolor':'k',"capsize":10}  #for matplotlib >= v1.4 use 'fmt':'none' instead

fig, ax = plt.subplots()
ax.p1 = plt.bar(ind, menMeans, **bar_kwargs)
ax.errs = plt.errorbar(ind, menMeans, yerr=menStd, **err_kwargs)


# Custom function to draw the diff bars

def label_diff(i,j,text,X,Y):
    x = (X[i]+X[j])/2
    y = 1.1*mean([Y[i], Y[j]])
    dx = abs(X[i]-X[j])-1

    props = {'connectionstyle':matplotlib.patches.ConnectionStyle("Bar", fraction=0.2),'arrowstyle':'-',
                 'shrinkA':10,'shrinkB':10,'linewidth':1}
    ax.annotate(text, xy=(X[i]+0.1,y+0.09 + dx*0.075), zorder=10, annotation_clip=False)
    ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)

# Call the function
label_diff(0,1,'p=0.0028',ind,menMeans)
label_diff(0,2,'p=0.0002',ind,menMeans)
label_diff(0,3,'p=0.0006',ind,menMeans)


plt.ylim(top=1.15)
plt.xticks(ind, labels, color='k')
ax.yaxis.grid(True)
plt.show()
