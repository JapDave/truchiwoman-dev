#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:22:56 2020

@author: jose
"""

import os
import pickle
import joblib
import pandas as pd
import gcsfs
from google.auth import default
from google.cloud import storage



def _read_file (path):
    """
    Carga un archivo csv, txt o pickle y devuelve el contenido correspondiente.
    Funciona tanto para GCP como para entorno local.

    @params:
        path(str): path completo desde donde cargar el archivo
        
    @returns:
        obj: Devuelve lo que habia almacenado en el archivo. Si el archivo
            era .csv, obj es un dataframe.
    """
    
    _, file_extension = os.path.splitext(path)
    
    if os.environ.get('SERVER_TYPE', '') == 'GCP':
        path = "gs://data_truchiwoman/" + path
        credentials, project = default()
        if file_extension == '.csv':
            if path.find("configs")>0:
                obj = pd.read_csv(path, keep_default_na=False)
            elif path.find("truchiontologia")>0:
                obj = pd.read_csv(path)
            else:
                obj = pd.read_csv(path)
        elif file_extension == '.pickle' or file_extension==".pkl":
            fs = gcsfs.GCSFileSystem(credentials=credentials)
            with fs.open(path, 'rb') as f:
                obj = pickle.load(f)
        elif file_extension == '.joblib':
                fs = gcsfs.GCSFileSystem(credentials=credentials)
                with fs.open(path, 'rb') as f:
                    obj = joblib.load(f)
        else:
            fs = gcsfs.GCSFileSystem(credentials=credentials)
            with fs.open(path, 'r') as fh:
                obj = fh.read()

    else:        
        if file_extension == '.csv':
            if path.find("configs")>0:
                obj = pd.read_csv(path, keep_default_na=False)
            elif path.find("truchiontologia")>0:
                obj = pd.read_csv(path)
            else:
                obj = pd.read_csv(path, index_col=0)
        elif file_extension == '.pickle':
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        elif file_extension == '.joblib':
            with open(path, 'rb') as f:
                obj = joblib.load(f)
        else:
            with open(path, 'r', encoding="utf-8") as fh:
                obj = fh.read()
    return obj


def _write_file (obj, path):
    """
    Guarda un archivo csv o pickle
    Funciona tanto para GCP como para entorno local
    
    @params:
        obj: objeto a guardar. Debe ser dataframe si path acaba en .csv,
            o cualquier clase de objeto si path acaba en .pickle
        path(str): path completo donde guardar el archivo
    """
    _, file_extension = os.path.splitext(path)
    if os.environ.get('SERVER_TYPE', '') == 'GCP':
        credentials, project = default()
        path = "gs://data_truchiwoman/" + path
        fs = gcsfs.GCSFileSystem(credentials=credentials)
        if file_extension == '.csv':
            obj.to_csv(path, index=False)

        elif file_extension == '.txt':
            with fs.open(path, 'w') as fh:
                fh.write(obj)

        else:
            with fs.open(path, 'wb') as f:
                pickle.dump(obj, f)
        
    else:
        if file_extension == '.csv':
            obj.to_csv(path, index=False)
            
        elif file_extension == '.txt':
            with open(path, 'a+') as fh:
                fh.write(obj)
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
                

def _list_dir(path):
    if os.environ.get('SERVER_TYPE', '') == 'GCP':
        credentials, project = default() 
        storage_client = storage.Client(credentials=credentials)
        blobs = storage_client.list_blobs('data_truchiwoman', prefix=path)
        
        cand_files = [b.name for b in blobs]
        if len(cand_files)>0:
            cand_files = [file.split("/")[-1] for file in cand_files if len(file)>0]      
        else:
            cand_files = []


    else:
        cand_files = os.listdir(path)
        
    return cand_files
        

def _path_exists(path):
    if os.environ.get('SERVER_TYPE', '') == 'GCP':
        path = "gs://data_truchiwoman/" + path
        credentials, project = default() 
        fs = gcsfs.GCSFileSystem(credentials=credentials)
        return fs.exists(path)
    else:
        return os.path.exists(path)

