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


def correct_lemma_identifier(x, syn_lem_inst):
    if x in syn_lem_inst.synonyms_dict and not len(x)<=2:
        return x
    else:
        return "&&"

def text_lemmatiser(text, syn_lem_inst):
    lemmatised_text = ""
    for e in regex.findall(r"(?<=^|(\<mark\/?\d\/?\>))([^\<\>]+)(?=$|(\<mark\/?\d\/?\>))", text):
        clean_te = syn_lem_inst.text_cleaner(e[1])
        for x in clean_te.split():   
            x_poss_lemmas0 = syn_lem_inst.final_lemmas.get(x) if syn_lem_inst.final_lemmas.get(x) else syn_lem_inst.final_lemmas.get(syn_lem_inst.custom_decoder(x))
            lemmatised_text += (sorted(x_poss_lemmas0, key=lambda x: correct_lemma_identifier(x, syn_lem_inst))[0] if x_poss_lemmas0 else x) + " "
        lemmatised_text += e[-1]
    return lemmatised_text.strip()



def real_time_processor(row, relev_trad, alter_trad, syn_lem_inst, stop_words):
        
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
        x_poss_lemmas = sorted(x_poss_lemmas0, key=lambda x: correct_lemma_identifier(x, syn_lem_inst))[0] if x_poss_lemmas0 else x
        syn0 = syn_lem_inst.synonyms_dict.get(x_poss_lemmas)
        proper_w, syn = (x_poss_lemmas, syn0) if syn0 else (syn_lem_inst.custom_decoder(x_poss_lemmas), syn_lem_inst.synonyms_dict.get(syn_lem_inst.custom_decoder(x_poss_lemmas))) 
        
        response_quality[x] = {}
        response_quality[x]["lemma"] = proper_w
        response_quality[x]["senses"] = syn.get("senses") if syn else 1
        response_quality[x]["n_synonyms"] = len(syn.get("synonyms")) if syn else 1
        response_quality[x]["eval_labels"] = []

        this_word_re = '(?<=(^|\W))('+proper_w+')(?=($|\W))'
        
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

        if syn and syn.get("synonyms") and not "stop_word" in response_quality[x]["eval_labels"]:
            synonyms_re = '(?<=(^|\W))('+"|".join(set([proper_w]+syn.get("synonyms")))+')(?=($|\W))'
            syn_in_trads_cond = regex.findall(regex.compile(synonyms_re, regex.I), relev_trad + " " + alter_trad)
            
            if syn_in_trads_cond:
                all_finds = len([e for e in syn_in_trads_cond if e and e[1]])
                response_quality[x]["eval_labels"].append(f"{all_finds}_finds_{syn.get('senses')}_senses_synonyms_found_in_text")

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
                
        else:
            info = {}
            
    if info.get("phrase"):
        relevancy = (relev_grade+response_quality.get("n_tokens"))*.5/(response_quality.get("n_words")+1)
    else:
        relevancy = (relev_grade+response_quality.get("n_tokens"))/(response_quality.get("n_words")+1)
    
    abstraction = (abstr_grade+response_quality.get("n_tokens"))/(response_quality.get("n_words")+1)
    
    return {"relevancy": relevancy, "abstraction": abstraction}



def meaning_extractor(resp_folder_path, syn_lem_inst, lemmatised_extr_df):

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
                response_df0 = onto_df[(onto_df.response1 + onto_df.response2).str.len() > 2]
                if response_df0.shape[0]>0:
                    response_df = response_df0[response_df0.relevancy==""]
                    if response_df.shape[0]>0:
                        row_evals = []
                        for row_n in range(response_df.shape[0]):
                            whole_row = response_df.iloc[row_n]
                            row = whole_row["response1"] + " " + whole_row["response2"]
                            
                            alt_trad = list(set(["human", "chatgpt"]).difference(set([whole_row.agent])))[0]
                            lemm_trads = lemmatised_extr_df[lemmatised_extr_df.unique_id==whole_row["unique_id"]]
                            relev_trad = lemm_trads["clean_"+whole_row.agent].tolist()[0]
                            alter_trad = lemm_trads["clean_"+alt_trad].tolist()[0]
                            row_quality = real_time_processor(row, relev_trad, alter_trad, syn_lem_inst, stop_words)
                            row_evals.append(response_grader(row_quality) | {"unique_id": int(whole_row["unique_id"])})
                            
                        eval_df = pd.DataFrame(row_evals).set_index("unique_id")
                        whole_df0 = onto_df.set_index("unique_id").merge(eval_df, how="left", left_index=True, right_index=True)
                        new_cols = {k: np.where(whole_df0[k+"_x"]=="", whole_df0[k+"_y"], whole_df0[k+"_x"]) for k in ["relevancy", "abstraction"]}
                        whole_df = whole_df0.assign(**new_cols)
                        onto_df = whole_df.loc[:, ~whole_df.columns.str.contains('_[xy]$', regex=True)].reset_index()
                        _write_file(onto_df, os.path.join(resp_folder_path, file))
                    enriched_dfs.append(onto_df[onto_df.relevancy!=""])
    
    if enriched_dfs:
        final_df = pd.concat(enriched_dfs, axis=0)
        relev_cols = {col: pd.to_numeric(final_df[col], errors="coerce") for col in ["relevancy", "abstraction"]}
        scaled_new_cols = {f"scaled_{k}": (v-v.min())/(v.max()-v.min()) for k, v in relev_cols.items()}
        final_df = final_df.assign(**(relev_cols | scaled_new_cols))
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
    onto_path = os.path.join(data_path, 'truchiontologia_translations.csv')
    lemmatised_extr_path = os.path.join(data_path, "lemmatised_extracts.csv")

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
        
            
    if not os.path.exists(lemmatised_extr_path):
        onto = _read_file(onto_path)
        lemmatised_extr_df = onto.assign(**dict(onto[["human", "chatgpt"]].map(lambda x: text_lemmatiser(x, syn_lem_inst)).rename(columns=lambda x: "clean_"+x)))
        _write_file(lemmatised_extr_df, lemmatised_extr_path)
    else:
        lemmatised_extr_df = _read_file(lemmatised_extr_path).reset_index()
    
        
    final_df = meaning_extractor(paths.get("configs_folder"), syn_lem_inst, lemmatised_extr_df)
    final_df = final_df.assign(text=final_df["response1"]+" "+final_df["response2"])

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
    
    