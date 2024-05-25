#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:14:06 2023

@author: mukelembe
"""



trad1 = """
Mi traducción podría ser la siguiente: 
   Por cada renglón, un año, un mes, o, por lo menos, un día. ¿Cómo han de contener lo inteligible? Como si los vocablos a enhebrar nos llegaran tras haber logrado atravesar sin rozarse siquiera el páramo en que necesariamente se ha de constituir el yo a tal fin, en el que se vive acampado, por temor a dejar huella. 
   """


trad2 = """
La traducción de ChatGPT podría ser la siguiente:
    ––En eso estoy. ¿Sabes lo que dijo Picasso tras retratar a Gertrud Stein? 
Por lo visto, alguien le fue a leer la cartilla a propósito de lo irreconocible 
que le había quedado el retrato y él contestó que tiempo al tiempo, que ya se 
le acabaría asemejando la obra a su modelo. ¿Y te puedes creer que Gertrud Stein 
acabó convencida de que, con los años, el semblante le había ido cambiando 
efectivamente hasta terminar que ni calcado al retratado por Picasso?
"""


questions = """
Qué es lo que dijo Gertrud Stein?
Quién es Picasso?
"""

original = """
في بداية الإسلام، استخدم العرب المسلمون عملة بيزنطية ذهبية ونحاسية،
 وأبقوا على حروفها اللاتينية، وعلى صور ملوك بيزنطة المصورة، والرموز المسيحية. كما استخدموا عملة فضية ساسانية،
 عليها صور الملوك الساسانيين، ورمز النار (المقدسة عندهم)،
 وعبارات بلغتهم الفهلوية. ثم بدأت تظهر عبارات إسلامية بحروف عربية على العملة المستخدمة لتعطيها طابعاً يتماشى مع الخلافة الجديدة.
 هنا مثال لعملة ساسانية، طبع في صدرها صورة الملك الساساني يزدجرد الثالث حفيد كسرى الثاني، وعبارة "بسم الله".
"""


with open("/Users/mukelembe/Documents/truchiwoman/data/1/mi_trad.txt", "w") as fh:
    fh.write(trad1)
    
with open("/Users/mukelembe/Documents/truchiwoman/data/1/bot_trad.txt", "w") as fh:
    fh.write(trad2)
    
with open("/Users/mukelembe/Documents/truchiwoman/data/1/preguntas.txt", "w") as fh:
    fh.write(questions)
        
with open("/Users/mukelembe/Documents/truchiwoman/data/1/original.txt", "w") as fh:
    fh.write(original)
        