#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:23:31 2024

@author: mukelembe
"""

import regex
import pandas as pd
import numpy as np 

import os

from collections import Counter

from synonyms_extractor import Synonyms_and_lemmas_saver
from DatabaseConnector import _read_file, _write_file, _list_dir

def real_time_processor(row, relev_trad, alter_trad, syn_lem_inst, inversed_lemmas, stop_words):
        
    response_quality = {}

    word_salad = syn_lem_inst.text_cleaner(row).lower().split()
    response_quality["n_tokens"] = len(row.split())
    response_quality["n_words"] = len(word_salad)
    
    in_albis_exprs = "no (sé|entiendo|me he enterado), a saber, cualquiera (sabe|entiende), más quisiera yo (saber|entender), no hay qui[eé]n se (entere|cosque|pispe|sepa)"
    in_albis_re = regex.compile("^("+regex.sub(r"\,\s", ")|(", in_albis_exprs)+")(?=(\W|$))", regex.I)
    
    answers = " ".join(regex.findall(r"(?<=\<\w+\d\>)[^\<\>]+(?=\<\w+\d[\\\/]\>)", relev_trad))
    if regex.search(in_albis_re, row):
        response_quality["phrase"] = "wondering_expression"

    for x in word_salad:        
        x_poss_lemmas0 = syn_lem_inst.final_lemmas.get(x) if syn_lem_inst.final_lemmas.get(x) else syn_lem_inst.final_lemmas.get(syn_lem_inst.custom_decoder(x))
        x_poss_lemmas = [x]+sorted(x_poss_lemmas0, key=lambda x: regex.search(r"o$", x).group(0) if regex.search(r"o$", x) else '&')[::-1] if x_poss_lemmas0 else x
        syn0, proper_w1 = None, None
        for proper_w0 in x_poss_lemmas:
            if syn_lem_inst.synonyms_dict.get(proper_w0):
                proper_w1, syn0 = (proper_w0, syn_lem_inst.synonyms_dict.get(proper_w0)) 
                
            if syn0:
                break
          
        proper_w1, syn0 = (proper_w1, syn0) if syn0 else (regex.sub(r"(?<=\w{2,})e?s$", "", x), syn_lem_inst.synonyms_dict.get(regex.sub(r"(?<=\w{2,})e?s$", "", x))) 
        proper_w, syn = (proper_w1, syn0) if syn0 else (proper_w1, syn_lem_inst.synonyms_dict.get(syn_lem_inst.custom_decoder(regex.sub(r"(?<=\w{2,})e?s$", "", x_poss_lemmas[0]))))
        
        response_quality[x] = {}
        response_quality[x]["lemma"] = proper_w
        response_quality[x]["senses"] = syn.get("senses") if syn else 1
        response_quality[x]["n_synonyms"] = len(syn.get("synonyms")) if syn else 1
        response_quality[x]["eval_labels"] = []

        this_word_poss = [proper_w]+inversed_lemmas.get(proper_w, [])
        this_word_re = '(?<=(^|\W))('+"|".join(set(this_word_poss))+')(?=($|\W))'
        
        wrd_in_relev_bit_cond = regex.search(regex.compile(this_word_re, regex.I), answers)
        wrd_in_relev_trad_cond = regex.search(regex.compile(this_word_re, regex.I), relev_trad)
        wrd_in_alter_trad_cond = regex.search(regex.compile(this_word_re, regex.I), alter_trad) 
        
        if not syn:
            response_quality[x]["eval_labels"].append("no_syn_word")
        else:
            response_quality[x]["eval_labels"].append("syn_word")


        if x in syn_lem_inst.nombres_propios:
            response_quality[x]["eval_labels"].append("proper_noun")
        
        elif proper_w in stop_words or x in stop_words:
            response_quality[x]["eval_labels"].append("stop_word")
        
        elif proper_w in syn_lem_inst.unresolved_words:
            response_quality[x]["eval_labels"].append("known_no_syn_word")
        
        elif regex.search(r"[^aoeiu]{4,}|^\w$", x):

            response_quality[x]["eval_labels"].append("not_spanish_word")
        
        else:
            response_quality[x]["eval_labels"].append("not_identified_word")
        
        
        if wrd_in_relev_bit_cond:
            wall_finds00 = len([e for e in regex.findall(regex.compile(this_word_re, regex.I), answers) if e])
            response_quality[x]["eval_labels"].append(f"{wall_finds00}_words_found_in_relev_bit")
       
        if wrd_in_relev_trad_cond:
            wall_finds0 = len([e for e in regex.findall(regex.compile(this_word_re, regex.I), relev_trad) if e])
            response_quality[x]["eval_labels"].append(f"{wall_finds0}_words_found_in_relev")
        
        if wrd_in_alter_trad_cond:
            wall_finds1 = len([e for e in regex.findall(regex.compile(this_word_re, regex.I), alter_trad) if e])
            response_quality[x]["eval_labels"].append(f"{wall_finds1}_words_found_in_alter")

        if syn and syn.get("synonyms"):
            for synom in syn.get("synonyms"):
                syn_syn = syn_lem_inst.synonyms_dict.get(synom) if syn_lem_inst.synonyms_dict.get(synom) else syn_lem_inst.synonyms_dict.get(syn_lem_inst.custom_decoder(synom))
                if syn_syn:
                    synom_senses = syn_syn.get("senses")
                else:
                    synom_senses = 1
                
                if inversed_lemmas.get(synom):
                    synonyms_re = '(?<=(^|\W))('+"|".join(set([x]+inversed_lemmas.get(synom)))+')(?=($|\W))'
                    syn_in_relev_trad_cond = regex.findall(regex.compile(synonyms_re, regex.I), relev_trad + " " + alter_trad)
                    
                    if syn_in_relev_trad_cond:
                        all_finds = len([e for e in syn_in_relev_trad_cond if e])
                        response_quality[x]["eval_labels"].append(f"{all_finds}_finds_{synom_senses}_senses_synonyms_found_in_text")

    return response_quality



def response_grader(response_quality):
    
    relev_grade, abstr_grade = 0, 0
    for w, info in response_quality.items():
        if isinstance(info, dict):
            all_elabel = " ".join(info.get("eval_labels"))
            if not ("stop_word" in info.get("eval_labels") or "not_spanish_word" in info.get("eval_labels")):
                relev_grade += 1
                if "syn_word" in info.get("eval_labels"):
                    abstr_grade += 1
                elif "known_no_syn_word" in info.get("eval_labels") or "proper_noun" in info.get("eval_labels"):
                    relev_grade += 1
                    
                if regex.search(r"found_in_relev_bit", all_elabel):
                    n_finds = [int(e[0]) for e in regex.findall(r"(?<=\s)\d(?=_words_found_in_relev_bit)", all_elabel) if e and regex.search(r"\d", e[0])]
                    relev_grade += 500/(info.get("senses")*2+info.get("n_synonyms")+sum(n_finds))
                elif regex.search(r"found_in_relev_trad", all_elabel):
                    n_finds = [int(e[0]) for e in regex.findall(r"(?<=\s)\d(?=_words_found_in_relev_trad)", all_elabel) if e and regex.search(r"\d", e[0])]
                    relev_grade += 200/(info.get("senses")*2+info.get("n_synonyms")+sum(n_finds))
                elif regex.search(r"found_in_(alter|text)", all_elabel):
                    n_finds = [int(e[0]) for e in regex.findall(r"(?<=\s)(\d+)(?=_finds_\d+_senses_(synonyms|words)_found_in_(alter|text))", all_elabel) if e and regex.search(r"\d", e[0])]
                    syn_sen = [info.get("senses")]+[int(e[0]) for e in regex.findall(r"(?<=_finds_)(\d+)(?=_senses_(synonyms|words)_found_in_(alter|text))", all_elabel) if e and regex.search(r"\d", e[0])]
                    relev_grade += 100/(np.mean(syn_sen)*2+info.get("n_synonyms")+sum(n_finds))
                    abstr_grade += 250/(np.mean(syn_sen)*2+info.get("n_synonyms")+sum(n_finds))
            
            elif "not_spanish_word" in info.get("eval_labels"):
                relev_grade -= 10
                abstr_grade -= 5
            
    if info.get("phrase"):
        relevancy = (relev_grade+response_quality.get("n_tokens"))*.5/response_quality.get("n_words")
    else:
        relevancy = (relev_grade+response_quality.get("n_tokens"))/response_quality.get("n_words")
    
    abstraction = (abstr_grade+response_quality.get("n_tokens"))/response_quality.get("n_words")
    
    return {"relevancy": relevancy, "abstraction": abstraction}



def meaning_extractor(resp_folder_path, syn_lem_inst):
    
    inversed_lemmas = syn_lem_inst.lemma_dict_inverter(syn_lem_inst.final_lemmas)
    all_nvl_wrds = syn_lem_inst.clean_text.lower().split()
    
    confused_stops = ["nada", "casa", "así", "muy", "sólo"]
    confused_by_stops = ["nadar", "casar", "asir", "poner"]
    all_nvl_lemmas = list(map(lambda x: syn_lem_inst.final_lemmas.get(x)[0] if syn_lem_inst.final_lemmas.get(x) else x, all_nvl_wrds))
    stop_words = set(confused_stops+[k for k, v in Counter(all_nvl_lemmas).items() if v/len(all_nvl_wrds) > .0021 and len(k)>1])-set(confused_by_stops)
    
    enriched_dfs = []
    files = _list_dir(resp_folder_path)
    for file in files:
        if regex.search(r"\.csv$", file):
            onto_df = _read_file(os.path.join(resp_folder_path, file))
            if not (onto_df.response1.isna().all() or onto_df.response2.isna().all()):
                response_df = onto_df[(onto_df.response1 + onto_df.response1).str.len() > 2]
                if response_df.shape[0]>0:
                    row_qualities, row_evals = [], []
                    for row_n in range(response_df.shape[0]):
                        whole_row = response_df.iloc[row_n]
                        row = whole_row["response1"] + " " + whole_row["response2"]
                        
                        alt_trad = list(set(["human", "chatgpt"]).difference(set([whole_row.agent])))[0]
                        relev_trad = whole_row[whole_row.agent]
                        alter_trad = whole_row[alt_trad]
                    
                        row_quality = real_time_processor(row, relev_trad, alter_trad, syn_lem_inst, inversed_lemmas, stop_words)
                        row_qualities.append(row_quality)
                        row_evals.append(response_grader(row_quality) | {"unique_id": int(whole_row["unique_id"])})
                        
                    eval_df = pd.DataFrame(row_evals).set_index("unique_id")
                    useful_cols = [c for c in response_df.columns if not regex.search(r"\s|unique_id", c)]
                    whole_df = pd.concat([response_df.set_index("unique_id")[useful_cols], eval_df], axis=1)
                    enriched_dfs.append(whole_df)
    
    if enriched_dfs:
        final_df = pd.concat(enriched_dfs, axis=0)
        scaled_cols = {f"scaled_{col}": (final_df[col]-final_df[col].min())/(final_df[col].max()-final_df[col].min()) for col in ["relevancy", "abstraction"]}
        final_df = final_df.assign(**scaled_cols)
        return final_df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    
    # Variables
    increase_syn_dict = 500
    iterations_for_unfound_syns = 0
    save_increase_step = 500
    save_class = False
    verbose = False

    # Paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    # responses_folder = os.path.join(data_path, "responses")
    configs_folder = os.path.join(data_path, "configs")
    resources_path = os.path.join(data_path, "linguistic_resources")
    
    nov_trad_path = os.path.join(resources_path, "novela_traducida.txt")
    class_path = os.path.join(resources_path, "synonyms_and_lemmas_class.joblib")
    
    meaningful_df_path = os.path.join(data_path, "meaningful_df.csv")

        
    paths = {
        "configs_folder": configs_folder,
        "nov_trad_path": nov_trad_path,
        "class_path": class_path
             }
    
    syn_lem_inst = Synonyms_and_lemmas_saver(paths.get("class_path"), paths.get("nov_trad_path"))
    syn_lem_inst = syn_lem_inst.main(iterations_for_unfound_syns=iterations_for_unfound_syns, 
                                     increase_syn_dict=increase_syn_dict,
                                     save_increase_step=save_increase_step, 
                                     verbose=verbose,
                                     save_class=save_class) 
    
    if not os.path.exists(meaningful_df_path):
        final_df = meaning_extractor(paths.get("configs_folder"), syn_lem_inst)
        final_df.to_csv(meaningful_df_path)
    else:
        final_df = pd.read_csv(meaningful_df_path)


    final_df = final_df.assign(text=final_df["answer1"]+" "+final_df["answer2"])

    print("From:")
    print(final_df.text.tolist())
    print("\nMax relevancy:")
    print(final_df[final_df.relevancy==max(final_df.relevancy)].text.values) 
    print("\nMin relevancy:")
    print(final_df[final_df.relevancy==min(final_df.relevancy)].text.values)    
    
    print("\n\nMax abstraction:")
    print(final_df[final_df.abstraction==max(final_df.abstraction)].text.values) 
    print("\nMin abstraction:")
    print(final_df[final_df.abstraction==min(final_df.abstraction)].text.values)    
            
    print(final_df.groupby("agent")[["time_elapsed", "scaled_relevancy", "scaled_abstraction"]].mean())
    
    