#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 09:26:05 2019

@author: rita

Flask app

"""

import regex, os
from random import sample, shuffle, choice
import pandas as pd
import numpy as np 
import pickle
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from unidecode import unidecode

import seaborn as sns
from flask import Flask, render_template, request, session, jsonify, redirect, url_for, flash

import uuid
import time
from datetime import datetime
import sys

from insight_retriever import meaning_extractor
from synonyms_extractor import Synonyms_and_lemmas_saver



app = Flask(__name__)
app.secret_key = b'_asfqwr54q3rfvcEQ@$'

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
sessions_folder = os.path.join(data_path, 'sessions')
configs_folder = os.path.join(data_path, 'configs')
responses_folder = os.path.join(data_path, 'responses')
feedback_folder = os.path.join(data_path, 'feedback')

onto_path = os.path.join(data_path, 'truchiontologia_translations.csv')
explain_path = os.path.join(data_path, 'truchiontologia_explanations.csv')

resources_path = os.path.join(data_path, "linguistic_resources")

nov_trad_path = os.path.join(resources_path, "novela_traducida.txt")
class_path = os.path.join(resources_path, "synonyms_and_lemmas_class.joblib")

meaningful_df_path = os.path.join(data_path, "meaningful_df.csv")
success_rates_path = os.path.join(data_path, "success_rates_df.csv")

paths = {
    "configs_folder": configs_folder,
    "nov_trad_path": nov_trad_path,
    "class_path": class_path,
    "meaningful_df_path": meaningful_df_path
         }

increase_syn_dict = 500
iterations_for_unfound_syns = 0
save_increase_step = 500
save_class = False
verbose = False
color_codes_wanted = {"human":"#9828bd", "chatgpt":"#ffcc33"}


time_notion0 = """
"<strong>Tiempo</strong>" es la división de
    la cantidad de <strong>segundos</strong> transcurridos desde que se mostró la primera traducción de cada azulejo hasta que se hubieron contestado las preguntas relativas a la misma
entre
    la cantidad de <strong>caracteres del texto original</strong> en árabe.
"""
relev_notion0 = """
"<strong>Relevancia</strong>" es la división normalizada de
    la cantidad de <strong>palabras de las respuestas</strong> enviadas cuyos lemas coinciden con los de las que aparecen (mostrados en el orden de la relevancia que se asigna, de más a menos)
        a) en el extracto de la primera traducción mostrada <strong>donde se ubican las respuestas</strong>
        b) y, si no, el resto de <strong>la primera traducción mostrada</strong>
    y, si no, más la cantidad de <strong>sinónimos de las palabras de las respuestas</strong> enviadas cuyos lemas coinciden con los de las palabras que aparecen
        c) en cualquiera de <strong>ambas traducciones</strong>
entre
    la <strong>cantidad de acepciones</strong> de cada término en las respuestas enviadas (o su sinónimo) que coincide con otro que otorga "relevancia" en cualquiera de las opciones listadas con anterioridad
    más la <strong>cantidad de veces que se encuentra</strong>
todo ello dividido, a su vez, por la cantidad total de palabras que contienen las respuestas.
"""
abstr_notion0 = """
"<strong>Abstracción</strong>" es la división normalizada de
    la cantidad de <strong>palabras de las respuestas</strong> enviadas y sus sinónimos cuyos lemas coinciden con los de las que aparecen
        a) en cualquiera de <strong>ambas traducciones</strong>
entre
    la <strong>cantidad de acepciones</strong> de cada término en las respuestas enviadas (o su sinónimo) que coincide con otro que otorga "abstracción" en cualquiera de ambas traducciones
    más la <strong>cantidad de veces que se encuentra</strong>
todo ello dividido, a su vez, por la cantidad total de palabras que contienen las respuestas.
"""

concepts_dict = {a: regex.sub("\'", '"', 
                              regex.sub("(?<=\d+px\W+\>([\w\s\(\)\.\"\,]*(\<\/?strong\>)*[\w\s\(\)\.\"\,]*)+)(?=$|(\<br\>){2,})", "</span>", 
                                      regex.sub(r"\n+", "<br><br>", 
                                                regex.sub(r"\n\s{3,}", "<br><br><span style='margin-left:20px'>", b.strip()))))
                 for a, b in [("time", time_notion0), ("relev", relev_notion0), ("abstr", abstr_notion0)]}


syn_lem_inst = Synonyms_and_lemmas_saver(paths.get("class_path"), paths.get("nov_trad_path"))
syn_lem_inst = syn_lem_inst.main(iterations_for_unfound_syns=iterations_for_unfound_syns, 
                                 increase_syn_dict=increase_syn_dict,
                                 save_increase_step=save_increase_step, 
                                 verbose=verbose,
                                 save_class=save_class) 
    
onto_df = pd.read_csv(onto_path)
explain_df = pd.read_csv(explain_path)

my_guide = {v: k for k, v in [e.values() for e in explain_df[["field_name", "field"]].to_dict("records")]}

letters = "ABCD"
numbers = set(range(1, 8)).difference(set([4]))
opt_list = []
for letter in letters:
    for i in numbers:
        opt_list.append(letter+str(i))



def legibility_checker(x):
    if ((regex.search(r'([aeiou]\w|\w[aeiou]){2,}', x) and len(x)>3) or 
        (len(x)<=3 and regex.search(r'(si|no)', unidecode(x)))):
        return True
    else:
        return False


def get_plot(user, boxplot_df, color_codes_wanted):
    
    sns.set_theme()
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10, 5))
    
    left   =  0.1  # the left side of the subplots of the figure
    right  =  0.9    # the right side of the subplots of the figure
    bottom =  0.1   # the bottom of the subplots of the figure
    top    =  0.9    # the top of the subplots of the figure
    wspace =  0.5     # the amount of width reserved for blank space between subplots
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

    sns.boxplot(ax=axes[0], data=boxplot_df, x="agent", y="weighted_times", hue="agent", palette=color_codes_wanted, legend=False, gap=.1, boxprops=dict(alpha=.75))
    sns.boxplot(ax=axes[1], data=boxplot_df, x="agent", y="scaled_relevancy", hue="agent", palette=color_codes_wanted, legend=False, gap=.1, boxprops=dict(alpha=.75))
    sns.boxplot(ax=axes[2], data=boxplot_df, x="agent", y="scaled_abstraction", hue="agent", palette=color_codes_wanted, legend=False, gap=.1, boxprops=dict(alpha=.75))

    sns.stripplot(ax=axes[0], data=boxplot_df[boxplot_df.user==user], x="agent", y="weighted_times", hue="agent", palette=color_codes_wanted, dodge=True, jitter=False, linewidth=1, s=8)
    sns.stripplot(ax=axes[1], data=boxplot_df[boxplot_df.user==user], x="agent", y="scaled_relevancy", hue="agent", palette=color_codes_wanted, dodge=True, jitter=False, linewidth=1, s=8)
    sns.stripplot(ax=axes[2], data=boxplot_df[boxplot_df.user==user], x="agent", y="scaled_abstraction", hue="agent", palette=color_codes_wanted, dodge=True, jitter=False, linewidth=1, s=8)
    
    axes[0].set_title("Tiempo", x=.5, y=1.025) 
    axes[1].set_title("Relevancia", x=.5, y=1.025)                       
    axes[2].set_title("Abstracción", x=.5, y=1.025) 
    
    
    for j in range(3):
        axes[j].set_xticks(["human", "chatgpt"])
        axes[j].set_xticklabels(["persona", "máquina"])
        axes[j].set(xlabel=None)
        axes[j].set(ylabel=None)
    
    plt.savefig(os.path.join('static', 'plots', 'boxplots.png'), transparent=True) 

    return None
    


def flahsing(prev_explored_df, n_tiles=3):

    if not prev_explored_df.empty:
        cond_resp0 = prev_explored_df[["response1", "response2"]].map(legibility_checker).apply(sum, axis=1)
        n_corr_resp = cond_resp0[cond_resp0==2].shape[0]
        if n_corr_resp < n_tiles:
            rest = n_tiles-n_corr_resp
            m_rest = f"n {rest}" if rest > 1 else f" {rest}"
            m = f"Tienes que contestar satisfactoriamente a las preguntas de al menos {n_tiles} de los azulejos de los laterales para poder acceder a este contenido, de los cuales te falta{m_rest} por explorar."
            flash(m, "danger")
    else:
        flash("Tienes que contestar satisfactoriamente a las preguntas de al menos {n_tiles} de los azulejos de los laterales para poder acceder a este contenido.", "danger")



def onto_df_reader(session_user, configs_folder, max_or_min="max"):
        
    init_time = [float(regex.sub(r"\.csv", "", e.split("__")[-1])) for e in os.listdir(configs_folder) 
                 if len(os.listdir(configs_folder)) > 0 and regex.match(regex.compile(session_user), e.split("__")[0])]
    
    if init_time: 
        if max_or_min == "max":
            outpath = f"{session_user}__{max(init_time)}.csv"
        else:
            outpath = f"{session_user}__{min(init_time)}.csv"
        onto_df = pd.read_csv(os.path.join(configs_folder, outpath), index_col=0, keep_default_na=False)
        return onto_df, outpath
    else:
        return pd.DataFrame(), None

            

            
@app.route("/mosca", methods=['GET'])
def landing():
    
    if request.method == 'GET':   
        if not "user" in session or not "passed_fly" in session:
            session["passed_fly"] = 1
            return render_template('mosca.html')
        else:
            return redirect(url_for('index'))


@app.route("/leyenda", methods=['GET'])
def recounting():
    
    if request.method == 'GET':   
        if not "user" in session or not "passed_guide" in session:
            session["passed_guide"] = 1
            return render_template('leyenda.html')
        else:
            return redirect(url_for('index'))


@app.route("/small_screen", methods=['GET'])
def small_screen():
    
    if request.method == 'GET':   
        return render_template('small_screen.html')


@app.route("/", methods=['GET', 'POST'])
def passing():
    
    if request.method == 'GET': 
        if not "user" in session:
            return render_template('portal.html')
        else:
            return redirect(url_for('landing'))
        
    if request.method == 'POST':
        contra = request.form.get('passable')

        if contra == "masmu7":
            return redirect(url_for('landing'))
        else:
            return render_template('portal.html')

    
@app.route("/index", methods=['GET'])
def index():

    if request.method == 'GET':   
        if ("passed_fly" in session and "passed_guide" in session) and not "user" in session:
            session['user'] = str(uuid.uuid4())
            session['start_time'] = time.time()
            
        elif not "passed_fly" in session:
            return redirect(url_for('passing'))
        
        elif not "passed_guide" in session:
            return redirect(url_for('recounting'))
            
            
        outpath = f"{session['user']}__{time.time()}.csv"
        init_time = [float(regex.sub(r"\.csv", "", e.split("__")[-1])) for e in os.listdir(configs_folder) 
                     if regex.match(regex.compile(session['user']), e.split("__")[0])]
        
        human_bot_shuffle = sample(["chatgpt", "human"]*15, 24)
        
        shuffled_onto = pd.concat([onto_df.sample(len(opt_list)).reset_index(), 
                                   pd.DataFrame({"field": opt_list, "agent": human_bot_shuffle, 
                                                 "response1":[""]*24, "response2":[""]*24})], 
                                  axis=1)
        
        if init_time:
            elapsed_hours = (datetime.fromtimestamp(time.time())-datetime.fromtimestamp(max(init_time))).total_seconds()/3600
            if elapsed_hours >= 24:
                shuffled_onto.to_csv(os.path.join(configs_folder, outpath))
                  
        else:
            shuffled_onto.to_csv(os.path.join(configs_folder, outpath))
                
        return render_template('index.html')

    
    
@app.route("/translations", methods=['GET', 'POST'])
def translator():
    
    onto, outpath = onto_df_reader(session["user"], configs_folder)
    field = request.args.get("location")

    if request.method == 'GET':  
        translations = {}
        for col in ["chatgpt", "human"]:
            trans0 = onto[onto.field==field][col].tolist()[0]
            trans1 = regex.sub(r"(?<=\<)\/mark\d+(?=\>)|(?<=\<)mark\d+\/(?=\>)", "/span", 
                                       regex.sub(r"(?<=\<)mark(?=\d+\>)", "span class=highlit_span id=", trans0))
            
            translations[col] = trans1

        original0 = onto[onto.field==field]["original"].tolist()[0]    
        original1 = regex.sub(r"(?<=\<)\/mark\d+(?=\>)|(?<=\<)mark\d+\/(?=\>)", "/span", 
                              regex.sub(r"(?<=\<)mark(?=\d+\>)", "span class=trigger id=", original0))
        
        agent = onto[onto.field==field]["agent"].tolist()[0]   
        other_agent = list(set(["chatgpt", "human"]).difference([agent]))[0]

        session["trans1"] = regex.sub(r"\n", "<br>", translations[agent].strip())
        session["trans2"] = regex.sub(r"\n", "<br>", translations[other_agent].strip())
        session["original"] = regex.sub(r"\n", "<br>", original1.strip())
        session["pregunta1"] = onto[onto.field==field]["question1"].tolist()[0]
        session["pregunta2"] = onto[onto.field==field]["question2"].tolist()[0]
        
        session["respuesta1"] = onto[onto.field==field]["response1"].tolist()[0]
        session["respuesta2"] = onto[onto.field==field]["response2"].tolist()[0]

        session["unique_id"] = onto[onto.field==field]["unique_id"].tolist()[0]
        session["len_orig"] = len(regex.sub(r'\<mark\d+\>', "", original0))
        session["agent"] = agent
        session["start_translation"] = time.time()       
             
        return render_template('translations.html')
    

    if request.method == 'POST':
        _, outpath0 = onto_df_reader(session["user"], configs_folder, max_or_min="min")

        onto.loc[onto.field==field, "response1"] = request.form.get('respuesta1')
        onto.loc[onto.field==field, "response2"] = request.form.get('respuesta2')
        onto.loc[onto.field==field, "time_elapsed"] = time.time()-session["start_translation"]
        onto.loc[onto.field==field, "time_elapsed_since_beginning"] = time.time()-float(regex.search(r"(?<=__)\d+\.\d+(?=\.csv)", outpath0).group())
        onto.loc[onto.field==field, "len_orig"] = session["len_orig"]
        onto.loc[onto.field==field, "user"] = session["user"]

        onto.to_csv(os.path.join(configs_folder, outpath))
        
        return "", 204
        
    
@app.route("/explanations", methods=['GET'])
def explainer():
    
    if request.method == 'GET':  
        field = request.args.get("location")
        explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
        explan_parts = regex.split(r"(?<=\.)\s*\n+\s*(?=[A-Z]\w)", explanation)
        session["title"] = "El futuro de la traducción"
        session["text1"] = "<span style='margin-left:2em'>" + explan_parts[0] + "</span>"
        session["text2"] = "<span style='margin-left:2em'>" + "</span><br><span style='margin-left:2em'>".join(explan_parts[1:]) + "</span>"

        return render_template('explanations.html')


@app.route("/wheres_wally", methods=['GET', 'POST'])
def wally_searcher():
    
    onto, outpath = onto_df_reader(session["user"], configs_folder)

    session["votes"] = {}
    session["success"] = {}
    session["hide"] = {}
    for l in "ABCD":
        for i in list(range(1, 4)) + list(range(5, 8)):
            session["hide"][l+str(i)] = int(bool(len(onto[onto.field==l+str(i)]["response1"].values[0])>0))
    
    if request.method == 'GET':  
        session["success_rate"] = ""
        flahsing(onto, 3)

        field = request.args.get("location")
        explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
        explanation1 = regex.sub(r"(persona|human[oa])", 
                                 f"<span style='background-color:{color_codes_wanted.get('human')}75'>{regex.search(r'(persona|human[oa])', explanation, regex.I).group()}</span>", 
                                 explanation)
        
        explanation2 = regex.sub(regex.compile('(máquina|automátic[oa]|chatgpt)', regex.I), 
                                 f"<span style='background-color:{color_codes_wanted.get('chatgpt')}75'>{regex.search(r'(máquina|automátic[oa]|chatgpt)', explanation1, regex.I).group()}</span>", 
                                 explanation1)
        
        session["text"] = regex.sub(r"\n", "<br>", explanation2)

        return render_template('wheres_wally.html')


    if request.method == 'POST':
        guesses = {}
        guesses["total_guessed"] = 0
        guesses["correct_guesses"] = 0
                
        for l in "abcd":
            for i in list(range(1, 4)) + list(range(5, 8)):
                elem = request.form.get(l+str(i))
                # elem = choice(["human", "chatgpt", "none"])
                this_tile = onto[onto.field==l.upper()+str(i)].agent.tolist()[0]
                if elem and not regex.search(r"none", elem):
                    session["votes"][l+str(i)] = elem
                    guesses["total_guessed"] += 1
                    guesses["correct_guesses"] += int(bool(regex.match(elem, this_tile)))
                    session["success"][l+str(i)] = int(bool(regex.match(elem, this_tile)))

        if guesses["total_guessed"] > 2:
            relev_session = {k: v for k, v in session.items() if k in ["user", "votes", "success"]}
            success_df0 = pd.DataFrame(relev_session)
            success_df0 = success_df0.reset_index().rename(columns={"index": "field_lower"})
            success_df1 = success_df0.assign(field=success_df0.field_lower.str.upper())
            success_df = onto.merge(success_df1, on=["user", "field"], how="inner")
            success_df1 = pd.concat([success_df[success_df.success==1].agent.value_counts()/success_df.agent.value_counts()*100, pd.Series({"success_rate": success_df.success.mean()*100})], axis=0)
            if os.path.exists(success_rates_path):
                success_df0 = pd.read_csv(success_rates_path, index_col=0)
                success_df2 = pd.concat([success_df0, 
                                         pd.DataFrame(success_df1).T], axis=0).fillna(0)
            else:
                success_df2 = pd.DataFrame(success_df1).T.fillna(0)
                
            success_df2.to_csv(success_rates_path)
            human_mean = round(success_df2.human.mean())
            chat_mmean = round(success_df2.chatgpt.mean())
            your_success = round(success_df1.success_rate) if not pd.isna(success_df1.success_rate.round()) else 0
            your_success_per_trans = success_df.groupby("agent")["success"].sum()
            
            session["success_rate"] = regex.sub(r"\n+", "<br><br>", f"""
            De {success_df.shape[0]} azulejos que habías completado y la naturaleza de cuyas traducciones has jugado a adivinar, has acertado el {your_success} % de las veces, <span style='color:white; background-color:{color_codes_wanted.get('chatgpt')}97'>&ensp;{your_success_per_trans.get("chatgpt", 0)}&ensp;</span> cuando se trataba de una traducción automática y <span style='color:white; background-color:{color_codes_wanted.get('human')}90'>&ensp;{your_success_per_trans.get("human", 0)}&ensp;</span> cuando no.
            De media, la gente acierta el {round(success_df2.success_rate.mean())} % de las veces, el <span style='color:white; background-color:{color_codes_wanted.get('human')}97'>&ensp;{human_mean}%&ensp;</span> si la primera traducción mostrada ha sido realizada por una persona y el <span style='color:white; background-color:{color_codes_wanted.get('chatgpt')}90'>&ensp;{chat_mmean}%&ensp;</span> si es maquinal.
            """.strip())
        else:
            session["success_rate"] = "Tienes que explorar un mínimo de 3 azulejos para poder jugar."
            session["success"] = {}
        
        
        return render_template('wheres_wally.html')



@app.route("/stats", methods=['GET'])
def stats_grabber(): 
    if request.method == 'GET':  
    
        field = request.args.get("location")
        explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
        session["text"] = regex.sub(r"\n+", "<br><br>", explanation)
        
        onto, outpath = onto_df_reader(session["user"], configs_folder)
        flahsing(onto, n_tiles=4)
        
        final_df = meaning_extractor(paths.get("configs_folder"), syn_lem_inst)
        if not final_df.empty and final_df.shape[0] >= 4:
            cond_time = final_df.time_elapsed<7
            cond_resp = final_df[["response1", "response2"]].map(legibility_checker).apply(sum, axis=1)
            cond_first_obs = final_df.time_elapsed_since_beginning==final_df.time_elapsed_since_beginning.min()
            users_df = final_df.assign(weighted_times = final_df.time_elapsed/final_df.len_orig,
                                       too_quick = cond_time,
                                       too_sloppy_resp = np.where(cond_resp == 2, False, True),
                                       first_obs = np.where(cond_first_obs, True, False))   
        
            boxplot_df = users_df[(users_df.too_quick==False) & (users_df.too_sloppy_resp==False) & (users_df.shape[0]>=3) & (users_df.first_obs==False)]
            if not boxplot_df.empty and boxplot_df.shape[0] >= 4:
                _ = get_plot(session['user'], boxplot_df, color_codes_wanted)
                
                session["time_notion"] = concepts_dict.get("time")
                session["relev_notion"] = concepts_dict.get("relev")
                session["abstr_notion"] = concepts_dict.get("abstr")
                session["amount_users"] = boxplot_df.user.nunique()
                session["amount_solved_tiles"] = boxplot_df["agent"].shape[0]
                amount_solved_tiles_human = boxplot_df.agent.value_counts().get('human', 0)
                amount_solved_tiles_chatgpt = boxplot_df.agent.value_counts().get('chatgpt', 0)
                session["amount_solved_tiles_human"] = f"<span style='color:white; background-color:{color_codes_wanted.get('human')}'>&ensp;{int(amount_solved_tiles_human)}&ensp;</span>"
                session["amount_solved_tiles_chatgpt"] = f"<span style='color:white; background-color:{color_codes_wanted.get('chatgpt')}'>&ensp;{int(amount_solved_tiles_chatgpt)}&ensp;</span>"
                
        return render_template('stats.html') 
            
    
    
@app.route("/feedback", methods=['GET', 'POST'])
def feedb_requester():
    field = request.args.get("location")
    explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
    session["text"] = regex.sub(r"\n+", "<br><br>", explanation)
    
    if request.method == 'GET':  
        onto, outpath = onto_df_reader(session["user"], configs_folder)
        flahsing(onto, 3)
        return render_template('feedback.html')    
    
    if request.method == 'POST':
        if 'user_feedback' in request.form:
            feedback_block = request.form.get('user_feedback')
            
            with open(os.path.join(feedback_folder, session["user"]+".txt"), 'a') as fh:
                fh.write(feedback_block+"\n\n")
                
        return "", 204


        
    

if __name__ == '__main__':
    app.run(debug = True, threaded = False)#, host="0.0.0.0")
    