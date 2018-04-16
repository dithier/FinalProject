# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:29:12 2018

@author: Ithier
"""
import pandas as pd
import numpy as np
import scipy.sparse.linalg
import scipy.sparse
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
df_teams = pd.DataFrame(columns = teams)
qm = matches.loc[matches["level"] == "qm"]

def OPR(teams, qm):
    """ 
        @param teams: list of team names
        @param qm: dataframe with qualification match information
        @return OPR: list with OPR scores for each team
    """
    teams.append("score")
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
    OPR = scipy.sparse.linalg.lsqr(sM,s)[0]
    return OPR

def DPR(teams, qm):
    """ 
        @param teams: list of team names
        @param qm: dataframe with qualification match information
        @return DPR: list with DPR scores for each team
    """
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
                matrix.loc[index, "score"] = rscore
            else:
                matrix.loc[index, r1] = 1
                matrix.loc[index, r2] = 1
                matrix.loc[index, r3] = 1
                matrix.loc[index, "score"] = bscore
        new_row = matrix.sum(axis = 0)
        m[counter, :] = new_row
    M = m.copy()
    M = M[:, 0:len(teams)-1]
    #https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.linalg.lsqr.html
    #https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix
    sM = scipy.sparse.csr_matrix(M)
    s = m.copy()
    s = s[:, -1]
    DPR = scipy.sparse.linalg.lsqr(sM, s)[0]
    return DPR



OPR_scores = OPR(teams, qm)
DPR_scores = DPR(teams, qm)