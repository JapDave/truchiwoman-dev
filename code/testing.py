#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:47:43 2024

@author: mukelembe
"""

import regex, os
from random import sample, shuffle
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from langdetect import detect_langs
from unidecode import unidecode

import uuid
import time
from datetime import datetime
import sys


def legibility_checker(x):
    if ((regex.search(r'([aeiou]\w|\w[aeiou]){2,}', x) and len(x)>3) or 
        (len(x)<=3 and regex.search(r'(si|no)', unidecode(x)))):
        return True
    else:
        return False



data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
sessions_folder = os.path.join(data_path, 'sessions')
configs_folder = os.path.join(data_path, 'configs')
responses_folder = os.path.join(data_path, 'responses')
feedback_folder = os.path.join(data_path, 'feedback')

#data_path = os.path.abspath('/Users/mukelembe/Documents/truchiwoman/data')
onto_path = os.path.join(data_path, 'truchiontologia_translations.csv')
explain_path = os.path.join(data_path, 'truchiontologia_explanations.csv')

session = {}
session['user'] = "94efd246-4890-4983-951f-2ad86724c017"

resppath = os.path.join(responses_folder, f"{session['user']}.csv")


init_time = [float(regex.sub(r"\.csv", "", e.split("__")[-1])) for e in os.listdir(configs_folder) 
             if regex.match(regex.compile(session['user']), e.split("__")[0])]

outpath = f"{session['user']}__{max(init_time)}.csv"
# config_df = pd.read_csv(os.path.join(configs_folder, outpath), index_col=None)

color_codes_wanted = {"human":"#9828bd", "chatgpt":"#ffcc33"}

user_dfs = []
for file_name in os.listdir(responses_folder):
    if regex.search(r"\d+.csv", file_name):
        prev_explored_df0 = pd.read_csv(os.path.join(responses_folder, file_name), index_col=None)
        prev_explored_df = prev_explored_df0.drop("Unnamed: 0", axis=1).set_index(pd.Series([file_name]*prev_explored_df0.shape[0], name="user")).reset_index()
        user_dfs.append(prev_explored_df)


users_df = pd.concat(user_dfs, axis=0).reset_index(drop=False).rename(columns={"index": "n_obs"})


color_codes_wanted = {"human":"#9828bd", "chatgpt":"#ffcc33"}

cond_time = users_df.time_elapsed<7
cond_resp = users_df[["answer1", "answer2"]].map(legibility_checker).apply(sum, axis=1)

users_df = users_df.assign(weighted_times = users_df.time_elapsed/users_df.len_orig,
                           too_quick = cond_time,
                           too_sloppy_resp = np.where(cond_resp == 2, False, True))   

boxplot_df = users_df[(users_df.too_quick==False) & (users_df.too_sloppy_resp==False) & (users_df.n_obs>0)]
barplot_df0 = boxplot_df.groupby("agent")["n_obs"].count().reset_index()
barplot_df1 = boxplot_df.groupby("agent")["user"].nunique().reset_index()

sns.set_theme()
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(8.5, 5))

left   =  0.1  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1   # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  1     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
        left    =  left, 
        bottom  =  bottom, 
        right   =  right, 
        top     =  top, 
        wspace  =  wspace, 
        hspace  =  hspace
)
y_title_margin = .95
x_title_margin = .25

sns.barplot(ax=axes[0], data=barplot_df0, x="agent", y="n_obs", hue="agent", palette=color_codes_wanted, legend=False, gap=.1)
sns.barplot(ax=axes[1], data=barplot_df1, x="agent", y="user", hue="agent", palette=color_codes_wanted, legend=False, gap=.1)
sns.boxplot(ax=axes[2], data=boxplot_df, x="agent", y="weighted_times", hue="agent", palette=color_codes_wanted, legend=False, gap=.1)


axes[0].set_title("""
           La cantidad total de azulejos 
           que se han resuelto
           """, x=x_title_margin, y=y_title_margin) 
           
axes[1].set_title("""
           La cantidad de usuarios  
           distintos que han resuelto azulejos
           """, x=x_title_margin, y=y_title_margin) 
                  
axes[2].set_title("""
           Segundos divididos por el número   
           de caracteres del texto original
           """, x=x_title_margin, y=y_title_margin) 


for j in range(3):
    axes[j].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    axes[j].set(xlabel=None)
    axes[j].set(ylabel=None)
    
    
    
    