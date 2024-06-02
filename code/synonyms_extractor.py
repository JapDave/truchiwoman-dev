#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:44:28 2024

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
from collections import Counter

from DatabaseConnector import _read_file, _write_file, _path_exists
    
class Synonyms_and_lemmas_saver():
    
    def __init__(self, class_path, nov_trad_path, verbose=True):
        self.class_path = class_path
        self.nov_trad_path = nov_trad_path
        self.verbose = verbose
        
        
    def _get_proxies(self, url):
        """
        Function meant to provide different proxies 
        to scrape the website of the url provided as input.
        """
        if not self.raw_prox:
            fix_url = 'https://free-proxy-list.net/'
            response = requests.get(fix_url)
            core_soup = bs(response.text, "html.parser")
            ips = [regex.sub(r"[\<td\>\/]", "", str(e)) for e in core_soup.find_all("td") if regex.match(r"\<td\>\d+(\.\d+){3}\<", str(e))]
            ports = [regex.sub(r"[\<td\>\/]", "", str(e)) for e in core_soup.find_all("td") if regex.match(r"\<td\>\d+\<", str(e))]
            self.raw_prox = [":".join(e) for e in list(zip(ips, ports))]
        
        proxy = random.choice(self.raw_prox)
        proxies = {"http"  : "http://" + proxy, 
                   "https" : "http://" + proxy, 
                   "ftp"   : "http://" + proxy} 
        
        self.r_obj = ""
        i = 0
        while i <= 100 and not self.r_obj:
            i += 1
            try:
                self.r_obj = requests.get(url, proxies).text
            except:
                proxy = random.choice(self.raw_prox)
                proxies = {"http"  : "http://" + proxy, 
                           "https" : "http://" + proxy, 
                           "ftp"   : "http://" + proxy}
        return self
    
    
    @staticmethod
    def _reduce_repeated_letters(x):
        finds = regex.findall(r"([^aeiou])\1(?=[^aeiou])|([aeiou])\2(?=[aeiou])|([iu])\3", x)
        finds = [e for f in finds for e in f if e]
        for m in finds:
            x = regex.sub(regex.compile(f"{m}"+"{2,}"), m, x)
        
        m1 = regex.search(r"([ií]){2,}", x)
        if m1:
            x = regex.sub(r"([ií]){2,}", m1.group()[-1],  x)
            
        return x
    
    
    @staticmethod
    def custom_decoder(word_root):
        wmatch = regex.search("ñ", word_root)
        uwr = unidecode(word_root)
        if wmatch:
            uwr = uwr[:wmatch.span()[0]] + "ñ" + uwr[wmatch.span()[-1]:]
    
        return uwr
    
    
    @staticmethod
    def text_cleaner(text):
        
        prep_list = "a, al, ante, bajo, cabe, con, contra, de, del, desde, durante, en, entre, hacia, hasta, mediante, para, por, según, sin, so, sobre, tras, través, versus, vía"
        desastre = """
        qu[eé] c[oó]mo cu[aá]n([td]o)? cu[aá]l d[oó]nde qui[eé]n(es)? porque pues dr pero a[uú]n aunque además 
        también tampoco quizás? acaso ta[ln] ya y e o u no ni s[íi] algo antes siempre nunca ahora enseguida ay ojalá
        """.strip()
        demonstr = "yo|t[uú]s?|ti|m[íi]o?s?|[eé]l|(nosotr|vosotr|ell|es|otr)[oa]s?|est?[aeo]s?|aquel(l[oa]s?)?|(alg|ning)?[uú]n|((alg|ning)?un|tod|nuestr|[tsc]uy|vuestr?)[oa]s?|cada|cual(es)?quiera?|sus?|con[tms]igo|últim(amente|o)|primer"
        reflex_list = "me te se nos [^m]os los? las? les?"
    
        prep_list0 = regex.sub(r"\,\s", "|", prep_list)
        preps = regex.compile("(?<=^|\W)"+f"({prep_list0})"+"(?=$|\W)", regex.I)
        
        desastre0 = "|".join(desastre.split())
        desastres = regex.compile("(?<=^|\W)"+f"({desastre0})"+"(?=$|\W)", regex.I)
    
        demonsters = regex.compile("(?<=^|\W)"+f"({demonstr})"+"(?=$|\W)", regex.I)
        
        reflex0 = "|".join(reflex_list.split())
        reflexes = regex.compile("(?<=^|\W)"+f"({reflex0})"+"(?=$|\W)", regex.I)
       
        clean_text = regex.sub("[\s\n]+", " ", 
                               regex.sub(r"([^\w\s\n]|\d)+", "",
                                         regex.sub(reflexes, "", 
                                                   regex.sub(preps, "", 
                                                             regex.sub(desastres, "", 
                                                                       regex.sub(demonsters, "", text)))))).strip()
    
        return clean_text
    
    
    @staticmethod
    def lemma_dict_inverter(legit_lemmas):
        lemmas_dict = {}
        for k, v in legit_lemmas.items():
            lemmas_dict.setdefault(k, []).append(k)
            for va in v:
                if not k == va:
                    lemmas_dict.setdefault(va, []).append(k)
        return lemmas_dict  
    
    
    
    def _synonyms_extractor(self, word):
        
        """
        Scrapes Word Reference to look for the word's root and synonyms 
        and scrapes deleahora.com and thefreedictionary.com for the word's
        amount of senses.
        """

        word_root = ""
        self._get_proxies(f'https://www.wordreference.com/sinonimos/{word}')
        core_soup = bs(self.r_obj, "html.parser")
        content = core_soup.find_all("div", class_="trans esp clickable")
        if content:
            synoms0 = []
            for entry in content:
                word_root = entry.find('h3').text
                if regex.search(r"\w+[aei]rse", word_root):
                    word_root = regex.sub(r"(?<=\w+)\,\s+.+", "", regex.sub(r"(?<=\w[aei]r)se", "", word_root))
                    
                for ul in entry.find_all('ul'):
                    for li in ul.find_all("li"):
                        if regex.search(r"\w+\,\s", ul.find("li").text):
                            synoms0 += list(map(lambda x: regex.sub(r'Ant\w+\:', "", x).strip(), li.text.split(", ")))
        
            if word_root:
                inflexes = []
                inflex_cont0 = core_soup.find_all("div", class_="inflectionsSection")
                inflex_cont = [inf for inf in inflex_cont0 if inf.find("strong") and regex.match(r"inflex", inf.find("strong").get_text(), regex.I)]
                if inflex_cont:
                    for inf in inflex_cont:
                        m = regex.search(r"((?<=.+\:\s*)(\w+)(?=[\s\n]?$)|(?<=(^|.+\:\s*))(\w+)(?=\s?\n([A-Z]\w+\s)*verbo.+))", inf.get_text())
                        if m:
                            inflexes.append(m.group().strip())
                    inflexes = [word] + inflexes
                else:
                    inflexes = [word] 
                
                self._get_proxies(f'https://deleahora.com/diccionario/{word_root}')
                core_soup1 = bs(self.r_obj, "html.parser")
                art_soup1 = core_soup1.find_all("ol")
                if art_soup1: 
                    lis = [li for e in art_soup1 for li in e.find_all("li") if not regex.match(r"\<li\>\<em\>", str(li))]
                    li_re = "(?<=(span(\<a+[^\>]+\>)?|li)\>\s?|[^\s\w\>]|\,\s(\<a+[^\>]+\>)?)(\w+)(?=[\,\.]|\<\/a)"
                    prel_synoms = sum([list(i) for e in lis for i in regex.findall(li_re, str(e)) if regex.search(li_re, str(e)) and i], [])
                    synoms1 = [e.lower() for e in prel_synoms if e and not regex.match(r"\<|\d|(span|li)$", e)]
                    senses = len(lis)         
        
                else:
                    synoms1 = None
                    self._get_proxies(f'https://es.thefreedictionary.com/{word_root}')
                    core_soup2 = bs(self.r_obj, "html.parser")
                    art_soup2 = core_soup2.find_all("section", {"data-src": "Larousse_GDLE"})
                    if art_soup2: 
                        senses = max(list(map(int, regex.findall(r"(?<=\<b\>)\d+(?=\<\/b\>)", str(art_soup2[0])))))
                    else:
                        senses = 1 
                
                if synoms0 or synoms1:
                    synoms = list(set(synoms0)) if synoms0 else list(set(synoms1))

                    self.synonyms_dict[word_root] = {}
                    self.synonyms_dict[word_root]["synonyms"] = synoms
                    self.synonyms_dict[word_root]["senses"] = senses
                    for wd in inflexes:
                        self.legit_lemmas.setdefault(wd, []).append(word_root)
                                            
        return self
    
    
    
    def _irreg_verbs_conjugator(self, verb):
        
        url0 = f'https://www.conjugacion.es/del/verbo/{verb}.php'
        url1 = f"https://www.elconjugador.com/conjugacion/verbo/{verb}.html"
        
        pronoms0 = "|".join("yo tú él nosotros vosotros ellos".split())
        reflex0 = "|".join("me te se nos os".split())
        pronoms = f"({pronoms0})\s+(({reflex0})\s+)?"
    
        if not self.haber_conjs:
            if self.irr_verbs_dict:
                self.haber_conjs = [k for k, e in self.irr_verbs_dict.items() if "haber" in e]
                
            if not self.haber_conjs:
                re2scrap = regex.compile(f"{pronoms}")
    
                self._get_proxies("https://www.conjugacion.es/del/verbo/haber.php")
                core_soup = bs(self.r_obj, "html.parser")
    
                content = core_soup.find_all("div", class_="tempscorps")
                if not content:
                    time.sleep(1) 
                    content = core_soup.find_all("div", class_="tempscorps")
                
                if content:
                    irr_verbs = []
                    for entry in content:
                        if regex.match(regex.compile(pronoms), entry.text):
                            irr_verbs += regex.sub(r"\/", " ", regex.sub(re2scrap, " ", entry.text)).split()
        
                    self.haber_conjs = list(set(irr_verbs+["haber", "habiendo"]))
                    
                    for conj in self.haber_conjs:
                        self.irr_verbs_dict[conj] = ["haber"]
                  
        re2scrap = regex.compile(f"{pronoms}(({'|'.join(self.haber_conjs)})\s+)?")
        
        self._get_proxies(url0)
        core_soup = bs(self.r_obj, "html.parser")
        content0 = core_soup.find_all("div", class_="tempscorps")
        
        irr_verbs = []
        if content0:
            for entry in content0:
                if regex.match(regex.compile(pronoms), entry.text):
                    irr_verbs += regex.sub(r"\/", " ", regex.sub(re2scrap, " ", entry.text)).split()
            
        pronoms0 = "\)|\(".join("yo tú él ns vs ellos".split())
        pronoms = f"(que\s+)?(\({pronoms0}\))\s+(({reflex0})\s+)?"
        rest0 = "|".join("simple compuesto perfecto anterior condicional pretérito presente futuro imperativo participio gerundio no\W negativo" .split())
        rest = regex.compile(f"\w*({rest0})\s*", regex.I)
    
        re2scrap = regex.compile(f"{pronoms}(({'|'.join(self.haber_conjs)})\s+)?")
    
        self._get_proxies(url1)
        core_soup = bs(self.r_obj, "html.parser")
        content1 = core_soup.find_all("div", class_="conjugBloc")
        if content1:
            for entry in content1:
                if not (len(entry.text) > 20 and not regex.search(r"\s", entry.text)):
                    irr_verbs += regex.sub(r"\(\d+\)|\-", " ", regex.sub(rest, " ", regex.sub(re2scrap, " ", entry.text))).split()
    
        irr_verbs = set(irr_verbs+[verb]).difference(set(self.haber_conjs))
            
        for conj in irr_verbs:
            self.irr_verbs_dict.setdefault(conj, []).append(verb)
                    
        return self
            
        
                
    def _irreg_verbs_extractor(self):
        
        self._get_proxies("https://www.esfacil.eu/es/verbos/categorias/11-irregular.html")
        core_soup = bs(self.r_obj, "html.parser")
    
        content = core_soup.find_all("div", class_="verb-category")
        irr_verbs = []
        for entry in content[0].find_all("a"):
            irr_verbs.append(entry.text)
            
        irr_verbs0 = [regex.sub(r"(?<=\w)se$", "", e) for e in irr_verbs if regex.search(r"(?<=\w)se$", e)]
        irr_verbs1 = [e for e in irr_verbs if not regex.search(r"(?<=\w)se$", e)]
        self.irreg_verbs = set(irr_verbs0+irr_verbs1)
        
        return self
          
    
    
    def _synonyms_finder(self, verb):
        
        reflex_list = "me te se nos [^m]os los? las? les?"
        reflex0 = "|".join(reflex_list.split())
        reflex = regex.compile("(?<=\w{2,})"+f"({reflex0})"+"{1,3}$")
        re_verb_conjs_pat_ar = regex.compile("([oó]|[ée](s|mos|is|n)?|(ab)?[aáe](s|mos|is|n)?|a(r[ae]|se)(s|mos|is|n)?|ar[áa][ns]?|ar[eé](is)?|aría(s|mos|is|n)?|[óé]|aste(is)?|ad|aron|an?d[oa]?s?)$")
        re_verb_conjs_pat_rest = regex.compile("([óoí[aeéií]+(s|mos|n|o|eron|ste(is)?)?|ie(r[ae]|se)(s|mos|is|n)?|[ei]r([áée]|ía)(s|n|mos|e?is)?|[ei]d|[yi](en)?d[ao]?s?)$")
    
        n0 = 0 if not self.sleeper else self.sleeper
    
        uword = Synonyms_and_lemmas_saver.custom_decoder(verb)
        ulemma = self.legit_lemmas.get(verb) if self.legit_lemmas.get(verb) else self.legit_lemmas.get(uword)
        syn = self.synonyms_dict.get(ulemma[0]) if ulemma else None
        if not syn:
            ulemma, syn = ([verb], self.synonyms_dict.get(verb)) if self.synonyms_dict.get(verb) else ([uword], self.synonyms_dict.get(uword))
    
        lemmas = {}
        if not (syn or verb in self.notfound_voc): 
            # Checking if it's a noun
            lemmas["noun0"] = regex.sub(r"(?<=\w{2,}[aeiou])s$|(?<=\w{2,}[^aeiou])es$", "", verb)
            lemmas["noun1"] = regex.sub(r"(?<=\w+[aeiouíéúóá])ces", "z", verb)
            lemmas["noun2"] = regex.sub(r"(?<=\w{3,})s$", "", verb)
            lemmas["noun3"] = regex.sub(r"(?<=\w{4,})as?$", "", verb)
            lemmas["noun4"] = regex.sub(r"(?<=\w{3,})as?$", "o", verb)
            lemmas["noun_may0"] = regex.sub(r"(?<=\w{3,})az[oa]s?$", "o", verb)
            lemmas["noun_may1"] = regex.sub(r"(?<=\w{3,})az[oa]s?$", "e", verb)
            lemmas["noun_may2"] = regex.sub(r"(?<=\w{3,})az[oa]s?$", "a", verb)
            lemmas["noun_may3"] = regex.sub(r"(?<=\w{3,})az[oa]s?$", "", verb)
            lemmas["noun_dim"] = regex.sub(r"(?<=\w{2,})[oa]s?$", "o", 
                                           regex.sub(r"(?<=\w{4,})s$", "", 
                                                   regex.sub(r"(?<=\w{3,5})[lc]$", "", 
                                                             regex.sub(r"(?<=\w+)gu(?=[ao]s?$)", "g", 
                                                                       regex.sub(r"(?<=\w+)qu(?=[ao]s?$)", "c", 
                                                                                 regex.sub(r"(?<=([^aeiou]|[gq]u))i(ll|t|c)(?=[oa]s?$)", "", verb))))))
                        
            # Checking if it's a adverb
            if regex.search(r"\w+mente$", verb):
                lemmas["adverb0"] = regex.sub(r"(?<=\w+)amente$", "o", verb)
                lemmas["adverb1"] = regex.sub(r"(?<=\w+)mente$", "", verb)
        
            # Checking if it's a verb
            if regex.search(reflex, verb):
                lemmas["reflex_verb"] = Synonyms_and_lemmas_saver.custom_decoder(regex.sub(reflex, "", verb))
                verb1 = lemmas["reflex_verb"]
            else:
                verb1 = Synonyms_and_lemmas_saver.custom_decoder(verb)
                
            results = []
            for i, vb in enumerate(set([verb, verb1])):
                if regex.search(re_verb_conjs_pat_ar, vb):
                    results.append(regex.sub(r"gu(?=ar$)", "g", 
                                             regex.sub(r"qu(?=ar$)", "c", 
                                                       regex.sub(re_verb_conjs_pat_ar, "ar", vb))))
                    results.append(regex.sub(r"j(?=ir$)", "g", regex.sub(re_verb_conjs_pat_ar, "ir", vb)))
                    results.append(regex.sub(r"j(?=er$)", "g", regex.sub(re_verb_conjs_pat_ar, "er", vb)))
            
                if regex.search(re_verb_conjs_pat_rest, vb) and not regex.search(r"[ao]$", regex.sub(re_verb_conjs_pat_rest, "", vb)):
                    results.append(regex.sub(r"j(?=er$)", "g", regex.sub(re_verb_conjs_pat_rest, "er", vb)))
                    results.append(Synonyms_and_lemmas_saver._reduce_repeated_letters(regex.sub(r"j(?=ir$)", "g", regex.sub(re_verb_conjs_pat_rest, "ir", vb))))
                    results.append(regex.sub(r"qu(?=ar$)", "c", regex.sub(re_verb_conjs_pat_rest, "ar", vb)))
            
                if results:
                    lemmas[f"reg_verb_possibles{i}"] = sorted(results, key=len)
        
            verbal_forms = [v for k, v in lemmas.items() if regex.search(r"verb", k)]
            verbal_forms = [verb]+sorted(list(set(sum([[e] if isinstance(e, str) else e for e in verbal_forms], []))), key=len)
            
            for v in verbal_forms:
                rv0 = self.irr_verbs_dict.get(v) if self.irr_verbs_dict.get(v) else self.irr_verbs_dict.get(Synonyms_and_lemmas_saver.custom_decoder(v))
                if rv0:
                    lemmas["irreg_verb"] = rv0
                    break
                else:
                    pre_match = regex.search(r"^(des|re)(?=\w+)", v)
                    if pre_match:
                        rv0 = self.irr_verbs_dict.get(regex.sub(r"^(des|re)(?=\w+)", "", v))
                        if rv0:
                            lemmas["irreg_verb"] = list(map(lambda x: pre_match.group() + x, rv0))
                            break
        
            possibilities = sorted(list(set(sum([[e] if isinstance(e, str) else e for e in lemmas.values()], []))), key=len)
            for word in possibilities:
                ulemma = self.legit_lemmas.get(word) if self.legit_lemmas.get(word) else [word]
                syn = self.synonyms_dict.get(ulemma[0]) if self.synonyms_dict.get(ulemma[0]) else self.synonyms_dict.get(Synonyms_and_lemmas_saver.custom_decoder(ulemma[0]))
                if syn:
                    break
                else:
                    syn = self.synonyms_dict.get(ulemma[0]+"se") if self.synonyms_dict.get(ulemma[0]+"se") else self.synonyms_dict.get(Synonyms_and_lemmas_saver.custom_decoder(ulemma[0]+"se"))
                    if syn:
                        break
                    else:
                        if self.sleeper % 50 == 0 and self.sleeper > n0+20:
                            time.sleep(1)
                            if self.verbose:
                                print("""
                                      Slept one second after looking up synonyms for the 50th word 
                                      that wasn't found.
                                      """)
                                      
                        self.sleeper =+ 1

                        if regex.search(r"\w+[aei]r$", word):
                            self._synonyms_extractor(word+"se")
                            ulemma = self.legit_lemmas.get(word+"se") if self.legit_lemmas.get(word+"se") else [word+"se"]
                            syn = self.synonyms_dict.get(ulemma[0]) if self.synonyms_dict.get(ulemma[0]) else self.synonyms_dict.get(Synonyms_and_lemmas_saver.custom_decoder(ulemma[0]))
                            if syn:
                                break
                            else:
                                uword = Synonyms_and_lemmas_saver.custom_decoder(word)+"se"
                                self._synonyms_extractor(uword)
                                ulemma = self.legit_lemmas.get(uword) if self.legit_lemmas.get(uword) else [uword]
                                syn = self.synonyms_dict.get(ulemma[0]) if self.synonyms_dict.get(ulemma[0]) else self.synonyms_dict.get(Synonyms_and_lemmas_saver.custom_decoder(ulemma[0]))
                                if syn:
                                    break
                        else:
                            self._synonyms_extractor(word)
                            ulemma = self.legit_lemmas.get(word) if self.legit_lemmas.get(word) else [word]
                            syn = self.synonyms_dict.get(ulemma[0]) if self.synonyms_dict.get(ulemma[0]) else self.synonyms_dict.get(Synonyms_and_lemmas_saver.custom_decoder(ulemma[0]))
                            if syn:
                                break
                            else:
                                decoded_w = Synonyms_and_lemmas_saver.custom_decoder(word)
                                if not decoded_w == word:
                                    self._synonyms_extractor(decoded_w)
                                    ulemma = self.legit_lemmas.get(decoded_w) if self.legit_lemmas.get(decoded_w) else [word]
                                    syn = self.synonyms_dict.get(ulemma[0]) if self.synonyms_dict.get(ulemma[0]) else self.synonyms_dict.get(decoded_w)
                                    if syn:
                                        break
        else:
            possibilities = []
         
        if syn:
            self.legit_lemmas.setdefault(verb, []).append(ulemma[0])
            self.legit_lemmas[verb] = sorted(list(set(self.legit_lemmas[verb])), key=len)
        else:
            self.notfound_voc.append(verb)
            
        return self
    
    
        
    def lemmas_enricher(self):
        
        lemmas_dict = Synonyms_and_lemmas_saver.lemma_dict_inverter(self.legit_lemmas)
        
        ar_verbs = [e for e in lemmas_dict.keys() if regex.search(r"\w+ar$", e) and len(e) > 3]
        er_verbs = [e for e in lemmas_dict.keys() if regex.search(r"\w+er$", e) and len(e) > 3]
        ir_verbs = [e for e in lemmas_dict.keys() if regex.search(r"\w+ir$", e) and len(e) > 3]
        
        nouns = [e for e in lemmas_dict.keys() if len(e) > 3 and not regex.search(r"\w+[aei]r$", e)]
        
        enriched_lemmas_dict = {}
        for l, verb_type in [("verb", ar_verbs), ("verb", er_verbs), ("verb", ir_verbs), ("noun", nouns)]:
            declinations = []
            for root in verb_type:
                if l == "verb":
                    lemma = regex.sub(r"[eai]r$", "", root)
                else:
                    lemma = regex.sub(r"o$", "", root)
                    
                declinations.append(list(map(lambda x: regex.sub(f"{lemma}", "", x), lemmas_dict.get(root))))
            
            if l == "verb": 
                declinations = [a for a, n in Counter(sum([e for e in declinations if isinstance(e, list)], [])).items() if n>2]
            else:
                declinations = [a for a, n in Counter(sum([e for e in declinations if isinstance(e, list)], [])).items() if n>6]

            for root in verb_type:
                if l == "verb":
                    lemma = regex.sub(r"[eai]r$", "", root)
                else:
                    lemma = regex.sub(r"[oa]$", "", root)
                    
                enriched_lemmas_dict[root] = set(list(map(lambda x: Synonyms_and_lemmas_saver._reduce_repeated_letters(lemma+x), declinations)))
                enriched_lemmas_dict[root] = [e for e in enriched_lemmas_dict[root] if not regex.search(r"([^aeioun][jg]\w{1,4}|[^aeiou]yendo)$|[^aeiou]{4,}", e)]

        final_lemmas0 = Synonyms_and_lemmas_saver.lemma_dict_inverter(enriched_lemmas_dict)
                        
        self.final_lemmas = {}
        all_keys = set(list(final_lemmas0.keys())+list(self.legit_lemmas.keys())+list(self.irr_verbs_dict.keys()))

        for k in all_keys:
            self.final_lemmas[k] = []
            if k in self.irr_verbs_dict:
                self.final_lemmas[k] += self.irr_verbs_dict.get(k)
            if k in self.legit_lemmas:
                self.final_lemmas[k] += self.legit_lemmas.get(k)
            if k in final_lemmas0:
                self.final_lemmas[k] += final_lemmas0.get(k)
            
            self.final_lemmas[k] = list(set(self.final_lemmas.get(k)))
            
        return self


        
    def main(self, iterations_for_unfound_syns=3, increase_syn_dict=5000, save_increase_step=50, save_class=True, verbose=True):
        
        if _path_exists(self.class_path):
            self = _read_file(self.class_path)

        else:
            self.clean_text = None
            
        self.verbose = verbose
        
        if not self.clean_text:
            if self.verbose:
                print("The synonyms' dictionary has yet to be built.")
            
            self.notfound_voc = []
            self.nombres_propios = []
            self.legit_lemmas = {}
            self.sleeper = 0
            self.final_lemmas = None
            self.iterations_made = 0
            self.haber_conjs = []
            self.raw_prox = None

            if not os.path.exists(self.nov_trad_path):
                raise Exception("Tienes que tener un corpus compendioso a partir del que poder crear el diccionario de sinónimos.")
                
            with open(self.nov_trad_path, "r") as fh:
                nov_trad = fh.read() 
        
            self.clean_text = Synonyms_and_lemmas_saver.text_cleaner(nov_trad)
            
            self._irreg_verbs_extractor()
            self.irr_verbs_dict = {}
            for irv in self.irreg_verbs:
                if not irv in self.irr_verbs_dict:
                    self._irreg_verbs_conjugator(irv)
                     
            if self.verbose:
                print("Looking up synonyms of irregular verbs ...")
                
            self.synonyms_dict = {}
            for word in tqdm(self.irreg_verbs):
                if not word in self.synonyms_dict:
                    self._synonyms_extractor(word)
                if not word in self.synonyms_dict:
                    self._synonyms_extractor(word+"se")
            
            if self.verbose:
                notfound = self.irreg_verbs.difference(set(self.synonyms_dict.keys()))
                self.not_in_wref = list(map(lambda x: regex.sub(r"se$", "", x), set(map(lambda x: x+"se", notfound)).difference(set(self.synonyms_dict.keys()))))
                print(f"\nIrregular verbs without traceable synonyms:\n\n{', '.join(self.not_in_wref)}\n")
                
            if save_class:
                _write_file(self, self.class_path)                
                                  
        # Enriching the synonyms' dictionary
        if self.iterations_made < iterations_for_unfound_syns:
            if self.verbose:
                print(f"""
                      The percentage of iterations completed rises to {self.iterations_made}/{iterations_for_unfound_syns}.
                      """)
                      
            for iteration in range(iterations_for_unfound_syns):
                if len(self.synonyms_dict) < increase_syn_dict or self.iterations_made < iterations_for_unfound_syns:
                    init_syn_len = len(self.synonyms_dict.keys())
                    text = self.clean_text.lower().split()
                    
                    if self.nombres_propios and not self.notfound_voc:
                        self.notfound_voc = list(self.nombres_propios)
                    else:
                        self.notfound_voc = list(self.notfound_voc)
                
                    for verb in text:
                        self._synonyms_finder(verb)
                
                        if len(self.synonyms_dict.keys()) > init_syn_len and len(self.synonyms_dict.keys()) % save_increase_step == 0:
                            init_syn_len = len(self.synonyms_dict.keys())
                            _write_file(self, self.class_path) 
                                 
                            self.nombres_propios = []
                            if len(self.notfound_voc) > 750:
                                nombres_propios_count = Counter(regex.findall(r'(?<=\b)[A-Z]\w+(?=\b)', self.clean_text))
                                nombres_propios0 = [name.lower() for name, count in nombres_propios_count.items() if count > 5 and not name in self.legit_lemmas]
                                self.nombres_propios = set(nombres_propios0).intersection(set(self.notfound_voc))

                            if self.verbose:
                                print(f"""
                                      Saving class after new batch of {save_increase_step} synonyms was added. 
                                      The amount of synonyms collected so far reaches {len(self.synonyms_dict)}.
                                      The amount of words without synonyms reaches {len(self.notfound_voc)}.
                                      """)
                                      
                if save_class:
                    _write_file(self, self.class_path)  
                    
                    self.iterations_made += 1
                    self.notfound_voc = []
                    
                if self.verbose:
                    print(f"""
                          Iteration nr. {self.iterations_made} complete.
                          """)
    
            
        if not self.final_lemmas:
            self.lemmas_enricher()    
            self.unresolved_words = set(self.notfound_voc).difference(self.nombres_propios)

            if save_class:
                _write_file(self, self.class_path)   

                if self.verbose:
                    print("\nThe saved class has all its attributes.\n")
        
        if self.verbose:
            print("\nLoaded class.\n")
        
        return self


    