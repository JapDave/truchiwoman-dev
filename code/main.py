#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 09:26:05 2019

@author: rita

Flask app

"""

import regex, os, io
from random import sample
import pandas as pd
import numpy as np 

from unidecode import unidecode

from flask import Flask, render_template, request, session, redirect, url_for, flash

import uuid
import time
from datetime import datetime

from synonyms_extractor import Synonyms_and_lemmas_saver
from insight_retriever import text_lemmatiser, meaning_extractor

from google.cloud import storage

from DatabaseConnector import _write_file, _read_file, _list_dir, _path_exists

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
class_path = os.path.join(resources_path, "synonyms_and_lemmas_class.pickle")
    
lemmatised_extr_path = os.path.join(data_path, "lemmatised_extracts.csv")
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
    
onto_df = _read_file(onto_path)
explain_df = _read_file(explain_path)

my_guide = {v: k for k, v in [e.values() for e in explain_df[["field_name", "field"]].to_dict("records")]}

letters = "ABCD"
numbers = set(range(1, 8)).difference(set([4]))
opt_list = []
for letter in letters:
    for i in numbers:
        opt_list.append(letter+str(i))



def legibility_checker(x):
    if ((regex.search(r'([aeiouáéíóú]\w|\w[aeiouáéíóú]){2,}', x) and len(x)>3) or 
        (len(x)<=3 and regex.search(r'(si|no)', unidecode(x)))):
        return True
    else:
        return False


def get_plot(user, boxplot_df, color_codes_wanted):
    from matplotlib import pyplot as plt
    import seaborn as sns
    
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
    
    if os.environ.get('SERVER_TYPE', '') == 'GCP': 
        client = storage.Client(project='truchiwoman')
        bucket = client.bucket('data_truchiwoman')
        blob = bucket.blob('plots/'+user+'.png')
        # temporarily save image to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        
        # upload buffer contents to gcs
        blob.upload_from_string(
            buf.getvalue(),
            content_type='image/png')
        
        buf.close()
        
        img_path = blob.public_url
    else:
        plt.savefig(os.path.join('static', 'plots', user+'.png'), transparent=True) 
        img_path = url_for('static', filename='plots/'+user+'.png')

    return img_path
    


def flashing(prev_explored_df, n_tiles=3):
    if '_flashes' in session:
        session['_flashes'].clear()    

    if not prev_explored_df.empty:
        cond_resp0 = prev_explored_df[["response1", "response2"]].map(legibility_checker).apply(sum, axis=1)
        n_corr_resp = cond_resp0[cond_resp0==2].shape[0]
        if not session["passed_fly"] > 1 and n_corr_resp < n_tiles:
            rest = n_tiles-n_corr_resp
            m_rest = f"n {rest}" if rest > 1 else f" {rest}"
            m = f"Tienes que contestar satisfactoriamente a las preguntas de al menos {n_tiles} de los azulejos de los laterales para poder acceder a este contenido, de los cuales te falta{m_rest} por explorar."
            flash(m, "danger")
    else:
        flash(f"Tienes que contestar satisfactoriamente a las preguntas de al menos {n_tiles} de los azulejos de los laterales para poder acceder a este contenido.", "danger")



def onto_df_reader(session_user, configs_folder):
    
    all_configs = _list_dir(configs_folder)
    init_time = ['.'.join(e.split("__")[-1].split(".")[:-1]) for e in all_configs if len(all_configs) > 0 and session_user==e.split("__")[0]]
    
    if init_time: 
        outpath = f"{session_user}__{max(init_time)}.csv"
        onto_df = _read_file(os.path.join(configs_folder, outpath))
        return onto_df, outpath
    else:
        return pd.DataFrame(), None

            
            
@app.route("/mosca", methods=['GET'])
def landing():
    if not "user" in session:
        session['user'] = str(uuid.uuid4())
        session['start_time'] = time.time()
    
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
        elif contra == "taller_unis":
            session["passed_fly"] = 1
            return redirect(url_for('index'))
        elif contra == "superadmina":
            session["passed_fly"] = 2
            session["passed_guide"] = 1
            return redirect(url_for('index'))
        else:
            return render_template('portal.html')


    
@app.route("/index", methods=['GET'])
def index():
    if request.method == 'GET':   
        if not "user" in session:
            return redirect(url_for('landing'))
            
        elif not "passed_fly" in session:
            return redirect(url_for('passing'))
        
        elif not "passed_guide" in session:
            return redirect(url_for('recounting'))
        
        outpath = f"{session['user']}__{time.time()}.csv"
        all_configs = _list_dir(configs_folder)
        init_time = ['.'.join(e.split("__")[-1].split(".")[:-1]) for e in all_configs if len(all_configs) > 0 and session["user"]==e.split("__")[0]]
        
        human_bot_shuffle = sample(["chatgpt", "human"]*15, 24)
        
        shuffled_onto = pd.concat([onto_df.sample(len(opt_list)).reset_index(), 
                                   pd.DataFrame({"field": opt_list, "agent": human_bot_shuffle, 
                                                 "response1":[""]*24, "response2":[""]*24,
                                                 "relevancy":[""]*24, "abstraction":[""]*24})], 
                                  axis=1)    
        if init_time:
            elapsed_hours = (datetime.fromtimestamp(time.time())-datetime.fromtimestamp(float(max(init_time)))).total_seconds()/3600
            if elapsed_hours >= 24:
                _write_file(shuffled_onto, os.path.join(configs_folder, outpath))
                  
        else:
            _write_file(shuffled_onto, os.path.join(configs_folder, outpath))
        
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

        content = {}

        content["trans1"] = regex.sub(r"\n+", "<br><br>", translations[agent].strip())
        content["trans2"] = regex.sub(r"\n+", "<br><br>", translations[other_agent].strip())
        content["original"] = regex.sub(r"\n+", "<br><br>", original1.strip())
        content["pregunta1"] = onto[onto.field==field]["question1"].tolist()[0]
        content["pregunta2"] = onto[onto.field==field]["question2"].tolist()[0]
        
        content["respuesta1"] = onto[onto.field==field]["response1"].tolist()[0]
        content["respuesta2"] = onto[onto.field==field]["response2"].tolist()[0]

        session["len_orig"] = len(regex.sub(r'\<mark\d+\>', "", original0))
        session["start_translation"] = time.time()       
             
        return render_template('translations.html', content = content)
    

    if request.method == 'POST':
        onto.loc[onto.field==field, "response1"] = request.form.get('respuesta1')
        onto.loc[onto.field==field, "response2"] = request.form.get('respuesta2')
        onto.loc[onto.field==field, "time_elapsed"] = time.time()-session["start_translation"]
        onto.loc[onto.field==field, "len_orig"] = session["len_orig"]
        onto.loc[onto.field==field, "user"] = session["user"]

        _write_file(onto, os.path.join(configs_folder, outpath))
        
        return "", 204
        
    
    
@app.route("/explanations", methods=['GET'])
def explainer():
    
    if request.method == 'GET':  
        field = request.args.get("location")
        explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
        explan_parts = regex.split(r"(?<=\.)\s*\n+\s*(?=[A-Z]\w)", explanation)
        content = {}
        content["title"] = "El futuro de la traducción"
        content["text1"] = "<span style='margin-left:25px'>" + explan_parts[0] + "</span>"
        content["text2"] = "<span style='margin-left:25px'>" + "</span><br><span style='margin-left:25px'>".join(explan_parts[1:]) + "</span>"

        return render_template('explanations.html', content = content)



@app.route("/wheres_wally", methods=['GET', 'POST'])
def wally_searcher():
    onto, outpath = onto_df_reader(session["user"], configs_folder)

    content = {}
    session["votes"] = {}
    session["success"] = {}
    content["hide"] = {}
    field = request.args.get("location")
    explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
    explanation1 = regex.sub(r"(persona|human[oa])", 
                                     f"<span style='background-color:{color_codes_wanted.get('human')}75'>{regex.search(r'(persona|human[oa])', explanation, regex.I).group()}</span>", 
                                     explanation)
            
    explanation2 = regex.sub(regex.compile('(máquina|automátic[oa]|chatgpt)', regex.I), 
                                     f"<span style='background-color:{color_codes_wanted.get('chatgpt')}75'>{regex.search(r'(máquina|automátic[oa]|chatgpt)', explanation1, regex.I).group()}</span>", 
                                     explanation1)
    content["text"] = regex.sub(r"\n", "<br>", explanation2)
    
    for l in "ABCD":
        for i in list(range(1, 4)) + list(range(5, 8)):
            content["hide"][l+str(i)] = int(bool(len(onto[onto.field==l+str(i)]["response1"].values[0])>0))
    
    if request.method == 'GET':  
        content["success_rate"] = ""
        flashing(onto, 3)
        return render_template('wheres_wally.html', content = content)
    
    
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
            
            success_df1 = {}
            success_df1["success_chatgpt"] = len(success_df[(success_df.success==1) & (success_df.agent=="chatgpt")])
            success_df1["fail_chatgpt"] = len(success_df[(success_df.success==0) & (success_df.agent=="chatgpt")])
            success_df1["success_human"] = len(success_df[(success_df.success==1) & (success_df.agent=="human")])
            success_df1["fail_human"] = len(success_df[(success_df.success==0) & (success_df.agent=="human")])
            success_df1 = pd.DataFrame(success_df1, index=[0])

            if  _path_exists(success_rates_path):
                success_df0 = _read_file(success_rates_path)
                success_df2 = pd.concat([success_df0, success_df1], axis=0)
            else:
                success_df2 = success_df1

            _write_file(success_df2, success_rates_path)
            
            global_mean = round((success_df2.success_human.sum()+success_df2.success_chatgpt.sum())/success_df2.sum().sum()*100)
            try:
                human_mean = round(success_df2.success_human.sum()/(success_df2.success_human.sum()+success_df2.fail_human.sum())*100)
            except:
                human_mean = "N/A"
                
            try:
                chat_mmean = round(success_df2.success_chatgpt.sum()/(success_df2.success_chatgpt.sum()+success_df2.fail_chatgpt.sum())*100)
            except:
                chat_mmean = "N/A"
                
            your_success = round((success_df1.success_human.sum()+success_df1.success_chatgpt.sum())/success_df1.sum().sum()*100)
            your_success_per_trans = success_df.groupby("agent")["success"].sum()
            
            content["success_rate"] = regex.sub(r"\n+", "<br><br>", f"""
            De {success_df.shape[0]} azulejos que has completado y la naturaleza de cuyas traducciones has jugado a adivinar, has acertado el {your_success} % de las veces, <span style='color:white; letter-spacing: .2rem; background-color:{color_codes_wanted.get('chatgpt')}97'>{your_success_per_trans.get("chatgpt", 0)}</span> cuando se trataba de una traducción automática y <span style='color:white; letter-spacing: .2rem; background-color:{color_codes_wanted.get('human')}90'>{your_success_per_trans.get("human", 0)}</span> cuando no.
            De media, la gente acierta el {global_mean} % de las veces, el <span style='color:white; letter-spacing: .2rem; background-color:{color_codes_wanted.get('human')}97'>{human_mean} %</span> si la primera traducción mostrada ha sido realizada por una persona y el <span style='color:white; letter-spacing: .2rem; background-color:{color_codes_wanted.get('chatgpt')}90'>{chat_mmean} %</span> si es maquinal.
            """.strip())
        else:
            content["success_rate"] = "Tienes que explorar un mínimo de 3 azulejos para poder jugar."
            content["success"] = {}
        
        
        return render_template('wheres_wally.html', content = content)



@app.route("/stats", methods=['GET'])
def stats_grabber(): 
    if request.method == 'GET':  
        content = {}
        field = request.args.get("location")
        explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
        content["text"] = regex.sub(r"\n+", "<br><br>", explanation)
        
        onto, outpath = onto_df_reader(session["user"], configs_folder)
        flashing(onto, n_tiles=4)
        
        if not _path_exists(lemmatised_extr_path):
            onto = _read_file(onto_path)
            lemmatised_extr_df = onto.assign(**dict(onto[["human", "chatgpt"]].map(lambda x: text_lemmatiser(x, syn_lem_inst)).rename(columns=lambda x: "clean_"+x)))
            _write_file(lemmatised_extr_df, lemmatised_extr_path)
        else:
            lemmatised_extr_df = _read_file(lemmatised_extr_path).reset_index()
            
        if not '_flashes' in session:      
            final_df = meaning_extractor(paths.get("configs_folder"), syn_lem_inst, lemmatised_extr_df)
        else:
            final_df = pd.DataFrame()
            
        if not final_df.empty and final_df.shape[0] >= 4:
            cond_time = pd.to_numeric(final_df.time_elapsed)<7
            cond_resp = final_df[["response1", "response2"]].map(legibility_checker).apply(sum, axis=1)
            cond_first_obs = pd.to_numeric(final_df.time_elapsed)==pd.to_numeric(final_df.time_elapsed).min()
            users_df = final_df.assign(weighted_times = pd.to_numeric(final_df.time_elapsed)/pd.to_numeric(final_df.len_orig),
                                       too_quick = cond_time,
                                       too_sloppy_resp = np.where(cond_resp == 2, False, True),
                                       first_obs = np.where(cond_first_obs, True, False))   
        
            boxplot_df = users_df[(users_df.too_quick==False) & (users_df.too_sloppy_resp==False) & (users_df.shape[0]>=3) & (users_df.first_obs==False)]
            flashing(boxplot_df, n_tiles=3)

            if not boxplot_df.empty and boxplot_df.shape[0] >= 3 and not '_flashes' in session:
                try:
                    content["img_path"] = get_plot(session['user'], boxplot_df, color_codes_wanted)
                except:
                    _write_file(str(session["user"])+"\n"+str(final_df.shape[0])+"\n"+str(boxplot_df.shape[0]), "/data/log_img_err.txt")
                    _write_file(final_df, "/data/final_df.csv")
                    _write_file(boxplot_df, "/data/boxplot_df.csv")
                content["time_notion"] = concepts_dict.get("time")
                content["relev_notion"] = concepts_dict.get("relev")
                content["abstr_notion"] = concepts_dict.get("abstr")
                content["amount_users"] = boxplot_df.user.nunique()
                content["amount_solved_tiles"] = boxplot_df["agent"].shape[0]
                amount_solved_tiles_human = boxplot_df.agent.value_counts().get('human', 0)
                amount_solved_tiles_chatgpt = boxplot_df.agent.value_counts().get('chatgpt', 0)
                content["amount_solved_tiles_human"] = f"<span style='color:white; background-color:{color_codes_wanted.get('human')}'>&ensp;{int(amount_solved_tiles_human)}&ensp;</span>"
                content["amount_solved_tiles_chatgpt"] = f"<span style='color:white; background-color:{color_codes_wanted.get('chatgpt')}'>&ensp;{int(amount_solved_tiles_chatgpt)}&ensp;</span>"   
            else:
                _write_file(str(session["user"])+"\n"+str(final_df.shape[0])+"\n"+str(boxplot_df.shape[0]), "/data/log_img_err.txt")
                _write_file(final_df, "/data/final_df_arg.csv")
                _write_file(boxplot_df, "/data/boxplot_df_arg.csv")
        return render_template('stats.html', content = content) 
            
    
    
@app.route("/feedback", methods=['GET', 'POST'])
def feedb_requester():
    content = {}
    field = request.args.get("location")
    explanation = explain_df[explain_df.field==field].text.tolist()[0].strip()
    content["text"] = regex.sub(r"\n+", "<br><br>", explanation)
    
    if request.method == 'GET':  
        onto, outpath = onto_df_reader(session["user"], configs_folder)
        flashing(onto, 3)
        return render_template('feedback.html', content = content)    
    
    if request.method == 'POST':
        if 'user_feedback' in request.form:
            feedback_block = request.form.get('user_feedback')  
            _write_file(feedback_block+"\n\n", os.path.join(feedback_folder, session["user"]+"__"+str(time.time())+".txt"))
                
        return "", 204



@app.route("/small_screen", methods=['GET'])
def small_screen():
    if request.method == 'GET':                   
        return render_template('small_screen.html')
      
    

if __name__ == '__main__':
    app.run(debug = True, threaded = False, host="0.0.0.0", port=8080)#, host="0.0.0.0")
    