#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:35:58 2024

@author: mukelembe
"""

import os, regex
import pandas as pd

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
configs_folder = os.path.join(data_path, "configs")


for file in os.listdir(configs_folder):
    if regex.search(r"\.csv$", file):
        df = pd.read_csv(file)
        df = df.assign(relevancy=pd.Series(), abstraction=pd.Series())
        print(df)
        print()