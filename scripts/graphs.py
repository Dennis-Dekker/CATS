# packages
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math

# enter raw data
baseline = np.array([0.727, 0.809, 0.684, 0.789, 0.684])
anova = np.array([0.864, 0.667, 0.842, 0.895, 0.842])
recursive = np.array([0.75, 0.85, 0.80, 0.75, 0.9])
l1_regularization = np.array([0.773, 0.762, 0.842, 0.842, 0.842])
anova_2 = np.array([0.818, 0.762, 1.0, 0.842, 0.842, 0.773, 0.667, 0.895, 0.842, 0.737, 0.773, 0.667, 0.895, 0.789, 0.737])
l1_regularization_2 = np.array([0.727, 0.762, 0.842, 0.842, 0.842, 0.727, 0.762, 0.947, 0.842, 0.842, 0.773, 0.667, 0.947, 0.895, 0.842])

# calculate the average
baseline_mean = np.mean(baseline)
anova_mean = np.mean(anova)
recursive_mean = np.mean(recursive)
l1_regularization_mean = np.mean(l1_regularization)
anova_2_mean = np.mean(anova_2)
l1_regularization_2_mean = np.mean(l1_regularization_2)

# calculate the standard deviation
baseline_std = np.std(baseline)
anova_std = np.std(anova)
recursive_std = np.std(recursive)
l1_regularization_std = np.std(l1_regularization)
anova_2_std = np.std(anova_2)
l1_regularization_2_std = np.std(l1_regularization_2)

# calculate the standard error
baseline_se = stats.sem(baseline)
anova_se = stats.sem(anova)
recursive_se = stats.sem(recursive)
l1_regularization_se = stats.sem(l1_regularization)
anova_2_se = stats.sem(anova_2)
l1_regularization_2_se = stats.sem(l1_regularization_2)


"""
# do t-test
print(anova.size)
t_baseline_anova = abs((baseline_mean - anova_mean)) / (math.sqrt(math.pow(baseline_std,2)/baseline.size +
                                                   math.pow(anova_std,2)/anova.size))
print(t_baseline_anova)
"""

print(anova_mean, anova_std, anova_se)
print(anova_2_mean, anova_2_std, anova_2_se)


# create lists for the plot
methods = ['Baseline', 'ANOVA', 'ANOVA_2', 'RFS-SVM', 'L1', 'L1_2']
x_pos = np.arange(len(methods))
CTEs = [baseline_mean, anova_mean, anova_2_mean, recursive_mean, l1_regularization_mean, l1_regularization_2_mean]
error = [baseline_se, anova_se, anova_2_se, recursive_se, l1_regularization_se, l1_regularization_2_se]

# build the plot
fig, ax = plt.subplots()
barlist=ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
barlist[0].set_color('r')
barlist[2].set_color('g')
barlist[3].set_color('y')
ax.set_ylabel('Accuracy')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()