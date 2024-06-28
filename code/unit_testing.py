#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:06:21 2024

@author: mukelembe
"""

import regex
import re
from itertools import chain
from pprint import pprint

import pandas as pd
import numpy as np
import os
import pickle
import time

import random
from random import sample
from collections import Counter

from synonyms_extractor import Synonyms_and_lemmas_saver
from insight_retriever import meaning_extractor, real_time_processor, response_grader


if __name__ == "__main__":
    # Variables
    increase_syn_dict = 500
    iterations_for_unfound_syns = 0
    save_increase_step = 500
    save_class = False
    
    # Paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    responses_folder = os.path.join(data_path, "responses")
    configs_folder = os.path.join(data_path, "configs")
    resources_path = os.path.join(data_path, "linguistic_resources")
    
    nov_trad_path = os.path.join(resources_path, "novela_traducida.txt")
    class_path = os.path.join(resources_path, "synonyms_and_lemmas_class.joblib")
    
    meaningful_df_path = os.path.join(data_path, "meaningful_df.csv")

        
    paths = {
        "configs_folder": configs_folder,
        "responses_folder": responses_folder,
        "nov_trad_path": nov_trad_path,
        "class_path": class_path
             }
    
    word = "casa"
    url = f'https://www.wordreference.com/sinonimos/{word}'
    
    syn_lem_inst = Synonyms_and_lemmas_saver(paths.get("class_path"), paths.get("nov_trad_path"))

    assert "perfecciollouni" == Synonyms_and_lemmas_saver._reduce_repeated_letters("perrfecciiollouuníi")
    assert "elefantes balanceaban cuerdas asi varios seguían haciéndolo" == Synonyms_and_lemmas_saver.text_cleaner("de los 1927 elefantes que se balanceaban sobre 34 cuerdas> o as'i! varios seguían haciéndolo")

    syn_lem_inst.raw_prox = None
    syn_lem_inst._get_proxies(url)
    assert syn_lem_inst.r_obj and syn_lem_inst.raw_prox
    syn_lem_inst.notfound_voc = []
    syn_lem_inst.synonyms_dict = {}
    syn_lem_inst.nombres_propios = []
    syn_lem_inst.legit_lemmas = {}
    syn_lem_inst.irr_verbs_dict = {}
    syn_lem_inst.sleeper = 0
    syn_lem_inst = syn_lem_inst._synonyms_finder(word)
    lemma = syn_lem_inst.legit_lemmas.get(word) if syn_lem_inst.legit_lemmas.get(word) else syn_lem_inst.legit_lemmas.get(Synonyms_and_lemmas_saver.custom_decoder(word))
    assert lemma
    assert syn_lem_inst.synonyms_dict.get(lemma[0]) or syn_lem_inst.synonyms_dict.get(Synonyms_and_lemmas_saver.custom_decoder(lemma[0]))
    
    syn_lem_inst = syn_lem_inst.main(iterations_for_unfound_syns=iterations_for_unfound_syns, 
                                     increase_syn_dict=increase_syn_dict,
                                     save_increase_step=save_increase_step,
                                     save_class=save_class) 
    
    supposed_class_atrributes = "class_path nov_trad_path synonyms_dict legit_lemmas sleeper irr_verbs_dict".split()
    class_atrributes = dir(syn_lem_inst)
    for ca in supposed_class_atrributes:
        assert ca in class_atrributes
        
    print(len(syn_lem_inst.synonyms_dict))
    inversed_lemmas = syn_lem_inst.lemma_dict_inverter(syn_lem_inst.final_lemmas)
    stop_words = ["también", "tiempo"]
    
    row = "jóvenes deslizamos ocurre acabe frutos buceo vocablo tiempo"
    relev_trad = """De jóvenes no estamos a rememorar, porque el presente y el futuro lo son todo, 
    pero, con el paso de los años, nos vamos haciendo más a bucear en recuerdos, no porque el presente 
    y el futuro hayan perdido su prestancia, si es que eso puede llegar a ocurrir, sino porque 
    <mark1>son nuestras experiencias pasadas las que nos equipan para navegarlos<mark1/>, vivencias felices 
    y amargas, que van ganando en luminosidad lo que pierden en nitidez con el transcurso del tiempo. 
    Consiguientemente, habré de armarme de paciencia y tesón <mark2>para conseguir desenmarañar y 
    poner en renglones de palabras inteligibles mi recolección de los hechos<mark2/>."""
    
    alter_trad = """En la juventud, nos avergonzamos de sumergirnos en los recuerdos porque el presente 
    y el futuro son más importantes y prominentes. Sin embargo, a medida que avanzan los años, la vergüenza 
    disminuye en nosotros y nos deslizamos hacia los recuerdos, no porque el presente y el futuro pierdan 
    importancia y prominencia, aunque eso también podría ser posible, sino <mark1>porque no soportamos mucho 
    de ellos excepto buscando ayuda de nuestras antiguas experiencias<mark1/>. Esas experiencias son paseos 
    alegres y dolorosos que se intensifican en la mente con ternura y se intensifican al mismo tiempo en 
    la memoria. Ven, paciencia, ven, perseverancia, y ven, palabras. <mark2>Para explicarlas con un poco 
    de claridad, para entrelazarlas en líneas comprensibles<mark2/>."""
    
    row_quality = real_time_processor(row, relev_trad, alter_trad, syn_lem_inst)
    print(syn_lem_inst.synonyms_dict.get(syn_lem_inst.final_lemmas.get("fruto")[0]))
    print(row_quality.get("frutos"))
    print({row: response_grader(row_quality)}) 
    
    