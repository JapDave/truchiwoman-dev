# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:26:26 2024

@author: Windows10
"""

import os
from synonyms_extractor import Synonyms_and_lemmas_saver

# Variables
increase_syn_dict = 10000
verbose = True
iterations_for_unfound_syns = 15

# Paths
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
resources_path = os.path.join(data_path, "linguistic_resources")

nov_trad_path = os.path.join(resources_path, "novela_traducida.txt")
class_path = os.path.join(resources_path, "synonyms_and_lemmas_class.pickle")

if not os.path.exists(resources_path):
    os.makedirs(resources_path)

syn_lem_inst = Synonyms_and_lemmas_saver(class_path, nov_trad_path)
syn_lem_inst = syn_lem_inst.main(iterations_for_unfound_syns=iterations_for_unfound_syns, 
                                 increase_syn_dict=increase_syn_dict)    

print(len(syn_lem_inst.synonyms_dict))
print(len(syn_lem_inst.final_lemmas))