# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:34:59 2018

@author: Ithier
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:24:28 2018

@author: Ithier
"""

import pandas as pd
import numpy as np
import scipy
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.from_csv.html
event_key = '2017txlu'
matches_file = event_key + '_matches.csv'
stats_file = event_key + '_stats.csv'
experience_file = event_key + '_rookieyear.csv'
matches = pd.DataFrame.from_csv(matches_file)
stats = pd.DataFrame.from_csv(stats_file)
experience = pd.DataFrame.from_csv(experience_file)
teams = pd.merge(experience, stats, on = "team")
teams = experience["team"].tolist()
teams.append("score")
df_teams = pd.DataFrame(columns = teams)
qm = matches.loc[matches["level"] == "qm"]

# https://erikrood.com/Python_References/rows_cols_python.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html
# http://book.pythontips.com/en/latest/enumerate.html
# # https://stackoverflow.com/questions/22963263/creating-a-zero-filled-pandas-data-frame
m = np.zeros((len(teams), len(teams)))
for counter, team in enumerate(teams):
    # create dataframe of only games played by one team
    blue_team = qm.loc[(qm["b1"] == team) | (qm["b2"] == team) | (qm["b3"] == team)]
    red_team = qm.loc[(qm["r1"] == team) | (qm["r2"] == team) | (qm["r3"] == team)]
    indiv_team = pd.concat([blue_team, red_team])
    indiv_team.reset_index(inplace = True)
    matrix = pd.DataFrame(0, index = np.arange(len(indiv_team)), columns = teams)
    # update dataframe with scores and who played in the match
    for index, row in indiv_team.iterrows():
        b1 = row["b1"]
        b2 = row["b2"]
        b3 = row["b3"]
        blue = []
        blue.extend([b1, b2, b3])
        bscore = row["bscore"]
        r1 = row["r1"]
        r2 = row["r2"]
        r3 = row["r3"]
        red = []
        red.extend([r1, r2, r3])
        rscore = row["rscore"]
        if team in blue:
            matrix.loc[index, b1] = 1
            matrix.loc[index, b2] = 1
            matrix.loc[index, b3] = 1
            matrix.loc[index, "score"] = bscore
        else:
            matrix.loc[index, r1] = 1
            matrix.loc[index, r2] = 1
            matrix.loc[index, r3] = 1
            matrix.loc[index, "score"] = rscore
    new_row = matrix.sum(axis = 0)
    m[counter, :] = new_row
M = m.copy()
M = M[:, 0:len(teams)-1]
sM = scipy.sparse.csr_matrix(M)
s = m.copy()
s = s[:, -1]
OPR = np.linalg.lstsq(M,s)[0]
OPR_s = scipy.sparse.linalg.lsqr(M,s)

#https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.linalg.lsqr.html
#https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix

