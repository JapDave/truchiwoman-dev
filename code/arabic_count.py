#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:35:58 2024

@author: mukelembe
"""

import requests
import regex
from bs4 import BeautifulSoup as bs

import os
import time
from unidecode import unidecode

import random
from tqdm import tqdm
from synonyms_extractor import Synonyms_and_lemmas_saver

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
configs_folder = os.path.join(data_path, "configs")
resources_path = os.path.join(data_path, "linguistic_resources")

nov_trad_path = os.path.join(resources_path, "novela_traducida.txt")
class_path = os.path.join(resources_path, "synonyms_and_lemmas_class.joblib")

    
paths = {
    "configs_folder": configs_folder,
    "nov_trad_path": nov_trad_path,
    "class_path": class_path
         }
    
    
alifat = "ا	ب	ت	ث	ج	ح	خ	د	ذ	ر	ز	س	ش	ص	ض	ط	ظ	ع	غ	ف	ق	ك	ل	م	ن	ه	و	ي".split()
print(alifat)
print(f"El alifato contiene {len(alifat)} consonantes.")

print(len(alifat) * 3)

combinations = []
for a in alifat:
    for b in alifat:
        if a != b:
            for c in alifat:
                combinations.append(a+b+c)
                    
print(len(combinations))


syn_lem_inst = Synonyms_and_lemmas_saver(paths.get("class_path"), paths.get("nov_trad_path"))
syn_lem_inst.raw_prox = None

ar_word_dict = {}
for word in tqdm(combinations):
    word_root = ""
    syn_lem_inst._get_proxies(f'https://www.dict.com/arabic-english/{word}')
    core_soup = bs(syn_lem_inst.r_obj, "html.parser")
    no_content = core_soup.find_all("span", class_="no_entry_found")

    if not no_content:
        ar_word_dict[word] = {}
        lemma = core_soup.find_all("span", class_="lex_ful_entr l1")
        senses = core_soup.find_all("span", class_="lex_ful_coll2")
        ar_word_dict[word]["lemma"] = lemma
        ar_word_dict[word]["senses"] = senses

        if len(ar_word_dict) >= 50:
            break









