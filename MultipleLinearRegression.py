import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import itertools as itr

import scipy.optimize as spo

from scipy.stats import f


def draw_scatters(df, degree=1, eq=1, vline=pd.DataFrame({'A' : []}), solo=0):
    # Plot scatter diagrams for each pair Xi & Y with regression degree 2
    end = df.shape[1]-1
    esymbol = df.columns[end]
    ecol = df[df.columns[end]]
    for i in range(0, end):
        symbol = df.columns[i]
        icol = df[symbol]

        x = icol
        y = ecol

        z = np.polyfit(x, y, degree)
        f = np.poly1d(z)

        x_new = np.linspace(x.min(), x.max(), 50)
        y_new = f(x_new)

        plt.scatter(x, y)
        plt.plot(x_new, y_new, 'r', linewidth=3)
        if (degree == 1 & eq == 1):
            plt.title(symbol + ' vs  ' + esymbol + "  y=(%.6f)x + (%.6f)" % (z[0], z[1]), fontsize=12)
        elif (degree == 2 & eq == 1):
            plt.title(symbol + ' vs  ' + esymbol + "  y=(%.6f)x^2 + (%.6f)x + (%.6f)" % (z[0], z[1], z[2]), fontsize=12)
        elif (degree == 3 & eq == 1):
            plt.title(symbol + ' vs  ' + esymbol + "  y=(%.6f)x^3 + (%.6f)x^2 + (%.6f)x + (%.6f)" % (z[0], z[1], z[2], z[3]), fontsize=12)

        # Plot optimized values
        if (vline.empty != True):
            plt.axvline(vline[symbol], color='green', linestyle='-', linewidth=3)

        # Plot one factor analysis
        if (solo == 1):
            if (symbol == 'Perlite'):
                tmp = df.loc[(df['Foam_glass'] == 0) & (df['Ufapore'] == 0)]
                plt.plot(tmp['Perlite'], tmp[esymbol], 'black', linestyle='--', linewidth=3)
            elif (symbol == 'Foam_glass'):
                tmp = df.loc[(df['Perlite'] == 0) & (df['Ufapore'] == 0)]
                plt.plot(tmp['Foam_glass'], tmp[esymbol], 'black', linestyle='--', linewidth=3)
            elif (symbol == 'Ufapore'):
                tmp = df.loc[(df['Perlite'] == 0) & (df['Foam_glass'] == 0)]
                plt.plot(tmp['Ufapore'], tmp[esymbol], 'black', linestyle='--', linewidth=3)

        plt.xlabel(symbol, fontsize=26)
        plt.ylabel(df.columns[end], fontsize=26)
        plt.grid(True)
        plt.show()

def x_value(i, j):
    # Using codes, calculate X values
    if j != 2:
        return i*0.5+0.5
    else:
        return i*0.1+0.1

def experiment(prod, report):
    # Calculate parameters and functions
    for i in list(prod):
        x1 = x_value(i[0], 0)
        x2 = x_value(i[1], 1)
        x3 = x_value(i[2], 2)
        params = [x1, x2, x3]

        report.loc[len(report.index)] = [x1, x2, x3, fr.get_model_value(0, params), fr.get_model_value(1, params), fr.get_model_value(2, params)]

class FactorResearch:
    def __init__(self):
        # this class can contain up to 3 models
        self.N = 0
        self.models = [0, 0, 0]

    def add_model(self, model):
        self.models[self.N] = model
        self.N += 1

    def get_model_value(self, i, params):
        return self.models[i].predict([params])[0]

    def regression_third(self, df):
        # Save factors and function names
        header = df.columns
        x1_name = header[0]
        x2_name = header[1]
        x3_name = header[2]
        y_name = header[3]

        # draw_scatters(df, 2)

        # Polynomial regression
        X = df.loc[:, x1_name:x3_name]
        Y = df[y_name]

        model = Pipeline([('poly', PolynomialFeatures(degree=3, interaction_only=False)),
                          ('linear', LinearRegression(fit_intercept=False))])
        model = model.fit(X, Y)
        # Save model
        self.add_model(model)

        print '\nCoefficients:\n', model.named_steps['linear'].coef_

        # Regression model visualization
        df.loc[:, 'model'] = np.NaN
        for i in range(0, df.shape[0]):
            df.loc[i, 'model'] = model.predict([df.loc[i, x1_name:x3_name]])[0]

        print '\n', df

        df[y_name].plot(label=y_name, linewidth=2)
        df['model'].plot(label='regression model', linewidth=2, ls='--')
        plt.xlabel('combination of factors', fontsize=26)
        plt.ylabel('function', fontsize=26)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Hypothesis Testing - Comparing Two Variances
        MVTR_std_dev, model_std_dev = df.loc[:, y_name:'model'].std()
        print '\n', y_name, 'standard deviation is: ', MVTR_std_dev
        print 'model standard deviation is: ', model_std_dev

        test_statistic = (MVTR_std_dev / model_std_dev) ** 2
        print "F Test statistic: ", test_statistic

        df1 = len(df.loc[:, y_name]) - 1
        df2 = len(df.loc[:, 'model']) - 1
        upper_crit_value = f.ppf(0.975, df1, df2)
        lower_crit_value = f.ppf(0.025, df1, df2)
        print '\nUpper critical value at a = 0.05 with df1 = {0} and df2 = {1}: '.format(df1, df2), upper_crit_value
        print 'Lower critical value at a = 0.05 with df1 = {0} and df2 = {1}: '.format(df1, df2), lower_crit_value


if __name__ == "__main__":
    # Import Excel data
    readExcel = pd.read_excel("./exp.xlsx")
    df = pd.DataFrame(readExcel)

    fr = FactorResearch()

    # Models generation
    end = df.shape[1]
    for i in range(3, end):
        iname = df.columns[i]
        tmp = df.iloc[:, 0:3]
        tmp.loc[:, iname] = df.iloc[:, i]

        fr.regression_third(tmp)

    # Get all products of [-1, -0.5, 0, +0.5, +1] - precised experiment
    codes = np.arange(-1, 1.1, 0.5)
    prod = itr.product(codes, repeat=3)
    report = pd.DataFrame(columns=df.columns)
    experiment(prod, report)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print '\n', report

    print '\n', report.loc[(report['MVTR'] >= 0.1) & (8 <= report['R']) & (report['R'] <= 12) & (report['C'] >= 7)]

    # Super-precised experiment
    codes = np.arange(-1, 1.01, 0.1)
    prod = itr.product(codes, repeat=3)

    report2 = pd.DataFrame(columns=df.columns)
    experiment(prod, report2)

    for i in range(3, 6):
        iname = report2.columns[i]
        tmp = report2.iloc[:, 0:3]
        tmp.loc[:, iname] = report2.iloc[:, i]

        draw_scatters(tmp, 3, 0, report.loc[84, :], 1)