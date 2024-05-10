import argparse
import base64
import csv
import datetime
import difflib
import io
import json

import logging
import math
import operator
import os
import pickle
from collections import defaultdict
from os.path import expanduser
from pathlib import Path
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd


import nltk
import dash
import sacrebleu
import dash_bootstrap_components as dbc
import diff_match_patch
import editdistance
import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
import spacy
from indicnlp.tokenize import sentence_tokenize
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize


import timeit
import json
from collections import Counter
import dask
from dask.distributed import Client
import gc  # for garbage collection


def parse_args():
    parser = argparse.ArgumentParser(description='Text Data Explorer')
    parser.add_argument(
        '--manifest', help='path to JSON manifest file',
    )
    parser.add_argument('--vocab', help='optional vocabulary to highlight OOV words')
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument(
        '--text_base_path',
        default=None,
        type=str,
        help='A base path for the relative paths in manifest. It defaults to manifest path.',
    )
    parser.add_argument(
        '--names_compared',
        '-nc',
        nargs = '+',
        type=str,
        help='names of the two fields that will be compared, example: pred_text_contextnet pred_text_conformer. "pred_text_" prefix IS IMPORTANT!',
    )
    parser.add_argument(
        '--prediction_mode',
        '-pm',
        type = bool,
        help = 'prediction mode is on when the data is pairwise two languages with one as OG language and another as true translated language',
    )
    parser.add_argument(
        '--comparison_mode',
        '-cm',
        default = False,
        type = bool,
        choices = [True,False]
    )
    parser.add_argument(
        '--total_tokens',
        '-tt',
        default = -1,
        type = int
    )
    parser.add_argument(
        '--reference',
        '-r',
        default = False,
        type = bool,
        choices = [True,False]
    )

    args = parser.parse_args()

    # assume text_filepath is relative to the directory where the manifest is stored
    if args.text_base_path is None:
        args.text_base_path = os.path.dirname(args.manifest)

    return args

def main(args):
    # start = datetime.datetime.now()
    # app.run_server(port = 8061,debug=True)


# %%
    PAGE_SIZE = 10

    # %%
    ''' Defining Parser '''
    

    # %%
    if(args == None):
        args = parse_args()

    # %%
    ''' Function load_data takes argument as the datafilename, text_base_path for the path of the datasets, comparison_mode for enabling comparison mode,
    names_compared for the names of the individual datasets and total which is the total no of tokens for computation'''
    def load_data(
        data_filename,
        text_base_path = None,
        comparison_mode = False,
        names_compared = None,
        total = 10000):
        def append_data(
            data_filename,
            text_base_path,
            f_name,total):
            data = []
            word_count = 0
            sentence_count = 0
            char_distribution = {}
            word_distribution = {}
            sent_tokens = 0

            with open(data_filename, 'r', encoding='utf8') as f:
                json_content = f.read()
            parsed_data = json.loads(json_content)
            if total == -1:
                with open(parsed_data['path' + f_name], 'r') as file:
                    data = file.read().replace('\n', '')
            else:
                with open(parsed_data['path' + f_name], 'r') as file:
                    data = file.read(total).replace('\n', '')
                
            count = 0
            for x in data:
                if(x in char_distribution.keys()):
                    char_distribution[x] += 1
                else:
                    char_distribution[x] = 1
            sub_list = ["\\"]
            for sub in sub_list:
                data = data.replace(sub, ' ')
            tokenizer = TreebankWordTokenizer()
            tokens = tokenizer.tokenize(data)
            for x in tokens:
                word_count += 1
                if(x in word_distribution.keys()):
                    word_distribution[x] += 1
                else:
                    word_distribution[x] = 1
            
            sent_tokens = sentence_tokenize.sentence_split(data,lang="en")
            # sent_tokens = sent_tokenize(data)
            sentence_count = len(sent_tokens)
            return data,word_count,sentence_count,sent_tokens,char_distribution,word_distribution
        data = {}
        word_count = {}
        sentence_count = {}
        char_distribution = {}
        word_distribution = {}
        sent_tokens = {}
        for x in range(1,len(names_compared)+1):
            name = str(x)
            f_name = names_compared[x-1]
            data[f_name],word_count[f_name],sentence_count[f_name],sent_tokens[f_name],char_distribution[f_name],word_distribution[f_name] = append_data(data_filename,text_base_path,name,total)
                        
        return data,word_count,sentence_count,sent_tokens,char_distribution,word_distribution

            



    # %%
    # manifest = args.manifest
    # text_base_path = args.text_base_path
    # comparison_mode = args.comparison_mode

    # %%
    # data,word_count,sentence_count,sent_tokens,char_distribution,word_distribution = load_data(manifest,text_base_path)

    # %%
    # data,word_count,sentence_count,sent_tokens,char_distribution,word_distribution = load_data("manifest.json",text_base_path="/Users/crysiswar999/Documents/MTP/English.rtf")

    # %%
    # comparison_mode = True
    # manifest = "compare.json"
    # text_base_path = "/Users/crysiswar999/Documents/MTP"
    # names_compared = ['English1','Odia1','Odia2','English2','Hindi1','Hindi2']
    # names_compared = ['English1','Odia1','English2','Hindi1']
    # prediction_mode = True
    # reference = True
    # total_tokens = 1000

    # %%
    comparison_mode = args.comparison_mode
    manifest = args.manifest
    manifest = os.path.abspath(os.path.join("translation", manifest))
    names_compared = args.names_compared
    prediction_mode = args.prediction_mode
    reference = args.reference
    total_tokens = args.total_tokens
    text_base_path = args.text_base_path

    # %%
    # data,word_count,sentence_count,sent_tokens,char_distribution,word_distribution = load_data(manifest,text_base_path,comparison_mode,names_compared,total_tokens)
    gc.collect()
    def process_chunk(chunk):
        # Word distribution and count
        words = chunk.split()
        word_count = len(words)
        word_distribution = Counter(words)

        # Character distribution and count
        char_distribution = Counter(chunk)
        char_count = len(chunk)

        return word_distribution, word_count, char_distribution, char_count

    # Dask client setup
    client = Client(memory_limit='3GB')

    import atexit

    def cleanup():
        client.close()

    atexit.register(cleanup)


    # Load the JSON file to get the file paths
    with open(manifest, 'r') as json_file:
        file_info = json.load(json_file)


    chunk_size = 100 * 1024 * 1024  # 100MB

    # Placeholders for results
    word_distribution = {}
    char_distribution = {}
    word_count = {}
    char_count = {}
    sent_tokens  = {}
    sentence_count = {}
    # Process each dataset 
    for key, (file_path, dataset_name) in enumerate(zip(file_info.values(), names_compared)):
        futures = []
        
        if total_tokens == -1:
            read_whole_file = True
        else:
            read_whole_file = False
            current_tokens = 0
        if total_tokens < chunk_size:
            chunk_size = total_tokens
        with open(file_path, 'rb') as file:
            while read_whole_file or (current_tokens < total_tokens):
                chunk = file.read(chunk_size).decode('utf-8','ignore')
                sub_list = ["\\"]
                for sub in sub_list:
                    chunk = chunk.replace(sub, ' ')
                
                # Adjust chunk based on last space or newline
                if len(chunk) == chunk_size:
                    last_space_index = chunk.rfind(' ')
                    last_newline_index = chunk.rfind('\n')
                    break_index = max(last_space_index, last_newline_index)
                    
                    file.seek(file.tell() - (chunk_size - break_index))
                    chunk = chunk[:break_index]

                if not chunk:
                    break

                # Scatter the chunk to the cluster
                chunk_future = client.scatter(chunk)
                result_future = client.submit(process_chunk, chunk_future)
                futures.append(result_future)
                
                # Update token count only if we're not reading the whole file
                if not read_whole_file:
                    current_tokens += chunk_size
                # print(current_tokens)

        # Gather results

        results = client.gather(futures)
        
        # Split the results into word and character distributions and counts
        word_distributions, word_counts_chunk, char_distributions, char_counts_chunk = zip(*results)

        # Aggregate results for the current dataset
        total_word_distribution = sum(word_distributions, Counter())
        total_char_distribution = sum(char_distributions, Counter())

        word_distribution[dataset_name] = dict(total_word_distribution)
        char_distribution[dataset_name] = dict(total_char_distribution)

        word_count[dataset_name] = sum(word_counts_chunk)
        char_count[dataset_name] = sum(char_counts_chunk)
        
        with open(file_path, 'r') as file:
            data = file.read(total_tokens).replace('\n', '')
        sub_list = ["\\"]
        for sub in sub_list:
            data = data.replace(sub, ' ')
        sen_tok = sentence_tokenize.sentence_split(data, lang="en")
        # sen_tok = sent_tokenize(data)
        sen_count = len(sen_tok)  
        sent_tokens[dataset_name] = sen_tok
        sentence_count[dataset_name] = sen_count
        
        # Memory cleanup
        del results
        del futures
        gc.collect()  # Invoke garbage collector
        print(file_path)

    client.close()


    # %%
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css'],
        title="Text Data Explorer",
    )

    # %%
    import plotly.express as px

    ''' Function plot_histogram takes the data,xlabel and ylabel and the count to make histograms for visualization'''
    def plot_histogram(data,xlabel,ylabel,count):
        fig = px.histogram(
            x=list(data.keys())[:count],  # Use the characters as x values
            y=list(data.values())[:count],  # Use the frequencies as y values
            nbins=50,
            log_y=True,
            labels={'x': xlabel, 'y': ylabel},  # Labels for axes
            opacity=0.5,
            color_discrete_sequence=['cornflowerblue'],
            height=200,
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
        return fig
    figures_hist = {}
    for x in range(1,len(names_compared)+1):
        f_name = names_compared[x-1]
        char_distribution[f_name] = dict(sorted(char_distribution[f_name].items(),key = lambda x : x[1],reverse=True))
        word_distribution[f_name] = dict(sorted(word_distribution[f_name].items(),key = lambda x : x[1],reverse=True))


        figures_hist['char_distribution' + str(x)] = ['Character Distribution' + str(x), plot_histogram(char_distribution[f_name],'characters','Frequency',20)] 
        figures_hist['word_distribution' + str(x)] = ['Word Distribution' + str(x),plot_histogram(word_distribution[f_name],'words','Frequency',20)]



    # %%


    # %%
    tabs_styles = {
        'height': '44px'
    }
    tab_style = {
        'borderBottom': '1px solid #d6d6d6',
        'padding': '6px',
        'fontWeight': 'bold'
    }

    tab_selected_style = {
        'borderTop': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': '#119DFF',
        'color': 'white',
        'padding': '6px'
    }

    # %%

    ''' Defining the tab labels for different operating modes like predicition mode,reference mode etc '''
    if(prediction_mode == True):
        if(reference == True):
            tab_labels = []
            for x in range(0,len(names_compared)-2,3):
                f_name1 = names_compared[x];
                f_name2 = names_compared[x+1];
                f_name3 = names_compared[x+2];
                tab_labels += [
                    dbc.Tab(label = f_name1[0:len(f_name1) - 1] + "->" + f_name2[0:len(f_name2)-1],tab_id  = f_name1[0:len(f_name1) - 1] + "->" + f_name2[0:len(f_name2)-1])
                ]
        else:
            tab_labels = []
            for x in range(0,len(names_compared)-1,2):
                f_name1 = names_compared[x];
                f_name2 = names_compared[x+1]
                tab_labels += [
                    dbc.Tab(label = f_name1[0:len(f_name1) - 1] + "->" + f_name2[0:len(f_name2)-1],tab_id  = f_name1[0:len(f_name1) - 1] + "->" + f_name2[0:len(f_name2)-1])
            ]

    # %%
    ''' Defining tab contents for the statistics page for different operating modes '''
    if(prediction_mode == True):
        if(reference == True):
            tab_content = {}
            for y in range(0,len(names_compared)-2,3):
                f_name1 = names_compared[y]
                f_name2 = names_compared[y + 1]
                f_name3 = names_compared[y + 2]
                x = y+1
                df1 = pd.DataFrame(list(word_distribution[f_name1].items()), columns=['Word', 'Count'])
                df2 = pd.DataFrame(list(word_distribution[f_name2].items()), columns=['Word', 'Count'])
                df3 = pd.DataFrame(list(word_distribution[f_name3].items()), columns=['Word', 'Count'])


                sub_tab_labels = []
                sub_tab_labels += [dbc.Tab(label = f_name1,tab_id = f_name1)]
                sub_tab_labels += [dbc.Tab(label = f_name2,tab_id = f_name2)]
                sub_tab_labels += [dbc.Tab(label = "Translated " + f_name3,tab_id = "Translated " + f_name3)]
                sub_tab_content = {}
                sub_tab_content[f_name1] = dbc.Card(
                    dbc.CardBody(
                        [
                    dbc.Row(dbc.Col(html.H6(children = f_name1[0:len(f_name1)-1]),className = 'text-secondary'),className = 'mt-3'),
                    dbc.Row(
                        [
                            dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Sentence Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Word Distribution', className='text-secondary'), width=3),
                        ],
                        className='bg-light mt-2 rounded-top border-top border-start border-end',
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    word_count[f_name1],
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(sentence_count[f_name1], className='text-center p-1', style={'color': 'cornflowerblue', 'opacity': 0.7}),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} characters'.format(len(char_distribution[f_name1])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} words'.format(len(word_distribution[f_name1])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                            ),
                        ],
                        className='bg-light rounded-bottom border-bottom border-start border-end',
                    ),
                    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(
                    dbc.Col(html.Div('{}'.format(sorted(char_distribution[f_name1].keys()))),), class_name='mt-2 bg-light font-monospace rounded border'
                ),
                    dbc.Row(dbc.Col(html.Div([
                        html.H5("Word Count Dashboard"),
                        dcc.Input(id='word-filter-input ' +str(x) , type='text', placeholder='Filter by word...'),
                        dcc.Input(id='count-filter-input ' + str(x), type='number', placeholder='Filter by count...'),
                        dash_table.DataTable(
                            id='word-table ' + str(x),
                            columns=[
                                {'name': 'Word', 'id': 'Word'},
                                {'name': 'Count', 'id': 'Count'},
                            ],
                            data=df1.to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            filter_action='custom',
                            page_size=10,  # Number of rows per page
                        )
                    ]))),
                    dbc.Row(dbc.Col(html.H5(figures_hist['char_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist['char_distribution' + str(x)][1])),),
                    dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='char_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                    dbc.Row(dbc.Col(html.H5(figures_hist['word_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='word_dist_graph' + str(x), figure=figures_hist['word_distribution' + str(x)][1]),),),
                    dbc.Row(dbc.Col(html.H6('Top K words Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='word_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                ]
                    )
                )

                x += 1


                sub_tab_content[f_name2] = dbc.Card(
                    dbc.CardBody(
                        [
                    dbc.Row(dbc.Col(html.H6(children = f_name2[0:len(f_name2)-1]),className = 'text-secondary'),className = 'mt-3'),
                    dbc.Row(
                        [
                            dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Sentence Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Word Distribution', className='text-secondary'), width=3),
                        ],
                        className='bg-light mt-2 rounded-top border-top border-start border-end',
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    word_count[f_name2],
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(sentence_count[f_name2], className='text-center p-1', style={'color': 'cornflowerblue', 'opacity': 0.7}),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} characters'.format(len(char_distribution[f_name2])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} words'.format(len(word_distribution[f_name2])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                            ),
                        ],
                        className='bg-light rounded-bottom border-bottom border-start border-end',
                    ),
                    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(
                    dbc.Col(html.Div('{}'.format(sorted(char_distribution[f_name2].keys()))),), class_name='mt-2 bg-light font-monospace rounded border'
                ),
                    dbc.Row(dbc.Col(html.Div([
                        html.H5("Word Count Dashboard"),
                        dcc.Input(id='word-filter-input ' +str(x) , type='text', placeholder='Filter by word...'),
                        dcc.Input(id='count-filter-input ' + str(x), type='number', placeholder='Filter by count...'),
                        dash_table.DataTable(
                            id='word-table ' + str(x),
                            columns=[
                                {'name': 'Word', 'id': 'Word'},
                                {'name': 'Count', 'id': 'Count'},
                            ],
                            data=df2.to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            filter_action='custom',
                            page_size=10,  # Number of rows per page
                        )
                    ]))),
                    dbc.Row(dbc.Col(html.H5(figures_hist['char_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist['char_distribution' + str(x)][1])),),
                    dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='char_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                    dbc.Row(dbc.Col(html.H5(figures_hist['word_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='word_dist_graph' + str(x), figure=figures_hist['word_distribution' + str(x)][1]),),),
                    dbc.Row(dbc.Col(html.H6('Top K words Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='word_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                ]
                    )
                )
                x += 1
                sub_tab_content[f_name3] = dbc.Card(
                    dbc.CardBody(
                        [
                    dbc.Row(dbc.Col(html.H6(children = "Translated " + f_name3[0:len(f_name3)-1]),className = 'text-secondary'),className = 'mt-3'),
                    dbc.Row(
                        [
                            dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Sentence Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Word Distribution', className='text-secondary'), width=3),
                        ],
                        className='bg-light mt-2 rounded-top border-top border-start border-end',
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    word_count[f_name3],
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(sentence_count[f_name3], className='text-center p-1', style={'color': 'cornflowerblue', 'opacity': 0.7}),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} characters'.format(len(char_distribution[f_name3])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} words'.format(len(word_distribution[f_name3])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                            ),
                        ],
                        className='bg-light rounded-bottom border-bottom border-start border-end',
                    ),
                    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(
                    dbc.Col(html.Div('{}'.format(sorted(char_distribution[f_name3].keys()))),), class_name='mt-2 bg-light font-monospace rounded border'
                ),
                    dbc.Row(dbc.Col(html.Div([
                        html.H5("Word Count Dashboard"),
                        dcc.Input(id='word-filter-input ' +str(x) , type='text', placeholder='Filter by word...'),
                        dcc.Input(id='count-filter-input ' + str(x), type='number', placeholder='Filter by count...'),
                        dash_table.DataTable(
                            id='word-table ' + str(x),
                            columns=[
                                {'name': 'Word', 'id': 'Word'},
                                {'name': 'Count', 'id': 'Count'},
                            ],
                            data=df1.to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            filter_action='custom',
                            page_size=10,  # Number of rows per page
                        )
                    ]))),
                    dbc.Row(dbc.Col(html.H5(figures_hist['char_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist['char_distribution' + str(x)][1])),),
                    dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='char_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                    dbc.Row(dbc.Col(html.H5(figures_hist['word_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='word_dist_graph' + str(x), figure=figures_hist['word_distribution' + str(x)][1]),),),
                    dbc.Row(dbc.Col(html.H6('Top K words Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='word_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                ]
                    )
                )


                tab_content[f_name1[0:len(f_name1)-1] + "->" + f_name2[0:len(f_name2)-1]] = dbc.Card(
                dbc.CardBody(
                    [
                    dbc.Tabs(
                        [dbc.Tab(sub_tab_content[f_name1],label = f_name1[0:len(f_name1)-1],tab_id = f_name1[0:len(f_name1)-1]),
                        dbc.Tab(sub_tab_content[f_name2],label = f_name2[0:len(f_name2)-1],tab_id =  f_name2[0:len(f_name2)-1] ),
                        dbc.Tab(sub_tab_content[f_name3],label = "Translated " + f_name3[0:len(f_name3)-1],tab_id = "Translated " + f_name3[0:len(f_name3)-1])],
                        active_tab = f_name1[0:len(f_name1)-1],
                    )
                ]
                ),
                className="mt-3",
                )
            active = names_compared[0][0:len(names_compared[0])-1] + '->' +names_compared[1][0:len(names_compared[1])-1]
        else:    
            tab_content = {}
            for y in range(0,len(names_compared)-1,2):
                f_name1 = names_compared[y]
                f_name2 = names_compared[y + 1]
                x = y+1
                df1 = pd.DataFrame(list(word_distribution[f_name1].items()), columns=['Word', 'Count'])
                df2 = pd.DataFrame(list(word_distribution[f_name2].items()), columns=['Word', 'Count'])


                sub_tab_labels = []
                sub_tab_labels += [dbc.Tab(label = f_name1[0:len(f_name1) - 1],tab_id = f_name1)]
                sub_tab_labels += [dbc.Tab(label = f_name2[0:len(f_name2) - 1],tab_id = f_name2)]
                sub_tab_content = {}
                sub_tab_content[f_name1] = dbc.Card(
                    dbc.CardBody(
                        [
                    dbc.Row(dbc.Col(html.H6(children = f_name1[0:len(f_name1)-1]),className = 'text-secondary'),className = 'mt-3'),
                    dbc.Row(
                        [
                            dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Sentence Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Word Distribution', className='text-secondary'), width=3),
                        ],
                        className='bg-light mt-2 rounded-top border-top border-start border-end',
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    word_count[f_name1],
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(sentence_count[f_name1], className='text-center p-1', style={'color': 'cornflowerblue', 'opacity': 0.7}),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} characters'.format(len(char_distribution[f_name1])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} words'.format(len(word_distribution[f_name1])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                            ),
                        ],
                        className='bg-light rounded-bottom border-bottom border-start border-end',
                    ),
                    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(
                    dbc.Col(html.Div('{}'.format(sorted(char_distribution[f_name1].keys()))),), class_name='mt-2 bg-light font-monospace rounded border'
                ),
                    dbc.Row(dbc.Col(html.Div([
                        html.H5("Word Count Dashboard"),
                        dcc.Input(id='word-filter-input ' +str(x) , type='text', placeholder='Filter by word...'),
                        dcc.Input(id='count-filter-input ' + str(x), type='number', placeholder='Filter by count...'),
                        dash_table.DataTable(
                            id='word-table ' + str(x),
                            columns=[
                                {'name': 'Word', 'id': 'Word'},
                                {'name': 'Count', 'id': 'Count'},
                            ],
                            data=df1.to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            filter_action='custom',
                            page_size=10,  # Number of rows per page
                        )
                    ]))),
                    dbc.Row(dbc.Col(html.H5(figures_hist['char_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist['char_distribution' + str(x)][1])),),
                    dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='char_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                    dbc.Row(dbc.Col(html.H5(figures_hist['word_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='word_dist_graph' + str(x), figure=figures_hist['word_distribution' + str(x)][1]),),),
                    dbc.Row(dbc.Col(html.H6('Top K words Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='word_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                ]
                    )
                )

                x += 1


                sub_tab_content[f_name2] = dbc.Card(
                    dbc.CardBody(
                        [
                    dbc.Row(dbc.Col(html.H6(children = f_name2[0:len(f_name2) - 1]),className = 'text-secondary'),className = 'mt-3'),
                    dbc.Row(
                        [
                            dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Sentence Count', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                            dbc.Col(html.Div('Word Distribution', className='text-secondary'), width=3),
                        ],
                        className='bg-light mt-2 rounded-top border-top border-start border-end',
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    word_count[f_name2],
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(sentence_count[f_name2], className='text-center p-1', style={'color': 'cornflowerblue', 'opacity': 0.7}),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} characters'.format(len(char_distribution[f_name2])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                                className='border-end',
                            ),
                            dbc.Col(
                                html.H5(
                                    '{} words'.format(len(word_distribution[f_name2])),
                                    className='text-center p-1',
                                    style={'color': 'cornflowerblue', 'opacity': 0.7},
                                ),
                                width=3,
                            ),
                        ],
                        className='bg-light rounded-bottom border-bottom border-start border-end',
                    ),
                    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(
                    dbc.Col(html.Div('{}'.format(sorted(char_distribution[f_name2].keys()))),), class_name='mt-2 bg-light font-monospace rounded border'
                ),
                    dbc.Row(dbc.Col(html.Div([
                        html.H5("Word Count Dashboard"),
                        dcc.Input(id='word-filter-input ' +str(x) , type='text', placeholder='Filter by word...'),
                        dcc.Input(id='count-filter-input ' + str(x), type='number', placeholder='Filter by count...'),
                        dash_table.DataTable(
                            id='word-table ' + str(x),
                            columns=[
                                {'name': 'Word', 'id': 'Word'},
                                {'name': 'Count', 'id': 'Count'},
                            ],
                            data=df2.to_dict('records'),
                            style_table={'height': '300px', 'overflowY': 'auto'},
                            filter_action='custom',
                            page_size=10,  # Number of rows per page
                        )
                    ]))),
                    dbc.Row(dbc.Col(html.H5(figures_hist['char_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist['char_distribution' + str(x)][1])),),
                    dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='char_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                    dbc.Row(dbc.Col(html.H5(figures_hist['word_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                    dbc.Row(dbc.Col(dcc.Graph(id='word_dist_graph' + str(x), figure=figures_hist['word_distribution' + str(x)][1]),),),
                    dbc.Row(dbc.Col(html.H6('Top K words Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                    dbc.Row(html.Div([dcc.Slider(
                        1,
                        20,
                        step=None,
                        value=10,
                        marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                        id='word_slider' + str(x),
                        className = 'slider'
                    )],style = {}
                        )),
                ]
                    )
                )


                tab_content[f_name1[0:len(f_name1) - 1] + "->" + f_name2[0:len(f_name2)-1]] = dbc.Card(
                dbc.CardBody(
                    [
                    dbc.Tabs(
                        [dbc.Tab(sub_tab_content[f_name1],label = f_name1[0:len(f_name1)-1],tab_id = f_name1),
                        dbc.Tab(sub_tab_content[f_name2],label = f_name2[0:len(f_name2)-1],tab_id = f_name2),],
                        active_tab = f_name1,
                    )
                ]
                ),
                className="mt-3",
            )
            active = names_compared[0][0:len(names_compared[0]) - 1] + '->' +names_compared[1][0:len(names_compared[1])-1]
            

    # %%
    ''' defining tab labels for prediction mode = False '''
    if prediction_mode == False:
        tab_labels = []
        for x in range(1,len(names_compared) + 1):
            f_name = names_compared[x-1]
            tab_labels += [
                dbc.Tab(label = f_name,tab_id = f_name)
            ]

    # %%
    ''' Defining Tab content for prediction_mode = False '''
    if prediction_mode == False:
        tab_content = {}
        for x in range(1,len(names_compared) + 1):
            f_name = names_compared[x - 1]
            df = pd.DataFrame(list(word_distribution[f_name].items()), columns=['Word', 'Count'])
            tab_content[f_name] = dbc.Card(
            dbc.CardBody(
                [
                dbc.Row(dbc.Col(html.H6(children = f_name),className = 'text-secondary'),className = 'mt-3'),
                dbc.Row(
                    [
                        dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                        dbc.Col(html.Div('Sentence Count', className='text-secondary'), width=3, className='border-end'),
                        dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                        dbc.Col(html.Div('Word Distribution', className='text-secondary'), width=3),
                    ],
                    className='bg-light mt-2 rounded-top border-top border-start border-end',
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5(
                                word_count[f_name],
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                        dbc.Col(
                            html.H5(sentence_count[f_name], className='text-center p-1', style={'color': 'cornflowerblue', 'opacity': 0.7}),
                            width=3,
                            className='border-end',
                        ),
                        dbc.Col(
                            html.H5(
                                '{} characters'.format(len(char_distribution[f_name])),
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                        dbc.Col(
                            html.H5(
                                '{} words'.format(len(word_distribution[f_name])),
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                        ),
                    ],
                    className='bg-light rounded-bottom border-bottom border-start border-end',
                ),
                dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
            dbc.Row(
                dbc.Col(html.Div('{}'.format(sorted(char_distribution[f_name].keys()))),), class_name='mt-2 bg-light font-monospace rounded border'
            ),
                dbc.Row(dbc.Col(html.Div([
                    html.H5("Word Count Dashboard"),
                    dcc.Input(id='word-filter-input ' +str(x) , type='text', placeholder='Filter by word...'),
                    dcc.Input(id='count-filter-input ' + str(x), type='number', placeholder='Filter by count...'),
                    dash_table.DataTable(
                        id='word-table ' + str(x),
                        columns=[
                            {'name': 'Word', 'id': 'Word'},
                            {'name': 'Count', 'id': 'Count'},
                        ],
                        data=df.to_dict('records'),
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        filter_action='custom',
                        page_size=10,  # Number of rows per page
                    )
                ]))),
                dbc.Row(dbc.Col(html.H5(figures_hist['char_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist['char_distribution' + str(x)][1])),),
                dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                dbc.Row(html.Div([dcc.Slider(
                    1,
                    20,
                    step=None,
                    value=10,
                    marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                    id='char_slider' + str(x),
                    className = 'slider'
                )],style = {}
                    )),
                dbc.Row(dbc.Col(html.H5(figures_hist['word_distribution' + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(dbc.Col(dcc.Graph(id='word_dist_graph' + str(x), figure=figures_hist['word_distribution' + str(x)][1]),),),
                dbc.Row(dbc.Col(html.H6('Top K words Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                dbc.Row(html.Div([dcc.Slider(
                    1,
                    20,
                    step=None,
                    value=10,
                    marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                    id='word_slider' + str(x),
                    className = 'slider'
                )],style = {}
                    )),
            ]
            ),
            className="mt-3",
        )
        active = names_compared[0]
    stats_layout = []
    stats_layout += [dbc.Row(dbc.Col(html.H5(children='Dataset Level Statistics'), className='text-secondary'), className='mt-3')]
    stats_layout += [html.Div(
        [
            dbc.Tabs(
                tab_labels,
                id="tabs",
                active_tab=active,
            ),
            html.Div(id="tab-content"),
        ]
    )]
    stats_layout = html.Div(stats_layout)


    # %%
    ''' Defining all the callback functions for different tabs for different modes in statistics page'''
    for x in range(1,len(names_compared) + 1):
        f_name = names_compared[x-1]
        df = pd.DataFrame(list(word_distribution[f_name].items()), columns=['Word', 'Count'])
        @app.callback(
            Output('char_dist_graph' + str(x),'figure'),
            Input('char_slider' + str(x),'value')
        )
        def word_plot_histogram(count,data = char_distribution[f_name],xlabel = 'Characters',ylabel = 'Frequency'):
            fig = px.histogram(
                x=list(data.keys())[:count],  # Use the characters as x values
                y=list(data.values())[:count],  # Use the frequencies as y values
                nbins=50,
                log_y=True,
                labels={'x': xlabel, 'y': ylabel},  # Labels for axes
                opacity=0.5,
                color_discrete_sequence=['cornflowerblue'],
                height=200,
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
            return fig
        @app.callback(
        Output('word_dist_graph' + str(x),'figure'),
        Input('word_slider' + str(x),'value')
        )
        def word_plot_histogram(count,data = word_distribution[f_name],xlabel = 'words',ylabel = 'Frequency'):
            fig = px.histogram(
                x=list(data.keys())[:count],  # Use the characters as x values
                y=list(data.values())[:count],  # Use the frequencies as y values
                nbins=50,
                log_y=True,
                labels={'x': xlabel, 'y': ylabel},  # Labels for axes
                opacity=0.5,
                color_discrete_sequence=['cornflowerblue'],
                height=200,
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
            return fig
        @app.callback(
        Output('word-table '+str(x), 'data'),
        Input('word-filter-input '+str(x), 'value'),
        Input('count-filter-input '+str(x), 'value')
        )
        def update_table(word_filter, count_filter,data = word_distribution[f_name]):
            df = pd.DataFrame(list(data.items()), columns=['Word', 'Count'])
            filtered_df = df
            if word_filter:
                filtered_df = filtered_df[filtered_df['Word'].str.contains(word_filter, case=True)]
            
            if count_filter is not None:
                filtered_df = filtered_df[filtered_df['Count'] >= int(count_filter)]

            return filtered_df.to_dict('records')
    @app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
    def switch_tab(at):
        return tab_content[at]

    # %% [markdown]
    ''' BackEnd computation for Samples page'''
    # 

    # %%
    model = SentenceTransformer('sentence-transformers/LaBSE')

    # %%
    def cosine_similarity(embedding1, embedding2):
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)

    # %%
    # from datasets import load_metric
    # bleurt_metric = load_metric('bleurt')

    ''' Computation for different operating modes for sample page '''
    # if prediction_mode == True and reference == True:
    #     df ={}
    #     for x in range(0,len(names_compared) - 2,3):
    #         data1 = {}
    #         f_name1 = names_compared[x]
    #         f_name2 = names_compared[x+1]
    #         f_name3 = names_compared[x+2]
    #         total_length = min(len(sent_tokens[f_name1]),len(sent_tokens[f_name2]),len(sent_tokens[f_name3]))
    #         data1[f_name1] = sent_tokens[f_name1][0:total_length]
    #         data1[f_name2] = sent_tokens[f_name2][0:total_length]
    #         data1[f_name3] = sent_tokens[f_name3][0:total_length]
    #         df["translation " + str(x)] = pd.DataFrame(data1)
    #         bleu = []
    #         labse = []
    #         chrf = []
    #         chrfpp = []
    #         bleurt = []
    #         for index, row in df["translation " + str(x)].iterrows():
    #             embeddings = model.encode(row)
    #             labse += [cosine_similarity(embeddings[1],embeddings[2])]
    #             source = [row[0]]
    #             refer = [row[1]]
    #             ref = [nltk.word_tokenize(r) for r in refer]
    #             prediction = row[2] 
    #             candidate = nltk.word_tokenize(prediction)
    #             prediction = [prediction]
    #             bleu += [sentence_bleu(ref,candidate)]
    #             chrf += [sacrebleu.corpus_chrf(candidate,ref).score]
    #             chrfpp += [sacrebleu.corpus_chrf(candidate,ref,word_order=2).score]
    #             result = bleurt_metric.compute(predictions=prediction, references=refer)
    #             bleurt += [result["scores"][0]]

    #         data1["Bleu Score"] = bleu
    #         data1["LabSe Score"] = labse
    #         data1["Chrf Score"] = chrf
    #         data1["Chrf++ Score"] = chrfpp
    #         data1["BleuRT Score"] = bleurt
    #         df["translation " + str(x)] = pd.DataFrame(data1)

    if prediction_mode == True and reference == True:
        # Function to calculate metrics for a chunk
        from dask import delayed, compute
        def process_metrics(chunk):
            bleu = []
            labse = []
            chrf = []
            chrfpp = []

            for index, row in chunk.iterrows():
                embeddings = model.encode(row)
                labse += [cosine_similarity(embeddings[1], embeddings[2])]
                source = [row[0]]
                refer = [row[1]]
                a = [x for x in source]
                b = [x for x in refer]
                ref = [nltk.word_tokenize(r) for r in refer]
                prediction = row[2]
                candidate = nltk.word_tokenize(prediction)
                prediction = [prediction]
                bleu +=[1]
                # bleu += [sacrebleu.corpus_bleu(a,[b])]
                # bleu += [sentence_bleu(ref, candidate)]
                chrf += [sacrebleu.corpus_chrf(candidate, ref).score]
                chrfpp += [sacrebleu.corpus_chrf(candidate, ref, word_order=2).score]
                # result = bleurt_metric.compute(predictions=prediction, references=refer)
                # bleurt += [result["scores"][0]]

            chunk["Bleu Score"] = bleu
            chunk["LabSe Score"] = labse
            chunk["Chrf Score"] = chrf
            chunk["Chrf++ Score"] = chrfpp

            return chunk

        # Function to process chunks in parallel
        def process_chunks_in_parallel(df):

            delayed_chunks = [delayed(process_metrics)(chunk) for _, chunk in df.items()]
            processed_chunks = compute(*delayed_chunks, scheduler="threads")

            result = {f"translation {i}": processed_chunk for i, processed_chunk in enumerate(processed_chunks)}

            return result
        def process_dataframe(df):
            chunk_size = 500  # Define your chunk size here
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            print(len(chunks))

            delayed_chunks = [delayed(process_metrics)(chunk) for chunk in chunks]
            processed_chunks, = compute(*delayed_chunks, scheduler="threads")
            return processed_chunks

            return pd.concat(processed_chunks)

        # Create a dictionary of DataFrames
        df = {}

        # Loop through data and create DataFrames
        for x in range(0, len(names_compared) - 2, 3):
            data1 = {}
            f_name1 = names_compared[x]
            f_name2 = names_compared[x + 1]
            f_name3 = names_compared[x + 2]
            total_length = min(len(sent_tokens[f_name1]), len(sent_tokens[f_name2]), len(sent_tokens[f_name3]))
            data1[f_name1] = sent_tokens[f_name1][0:total_length]
            data1[f_name2] = sent_tokens[f_name2][0:total_length]
            data1[f_name3] = sent_tokens[f_name3][0:total_length]
            df["translation " + str(x)] = pd.DataFrame(data1)

        # Prepare chunks and process them in parallel
        # chunks_to_process = process_chunks_in_parallel(df)
        processed_dfs = {key: process_dataframe(df[key]) for key in df.keys()}

        # Combine processed DataFrames into a final DataFrame
        # final_result = pd.concat(processed_dfs.values())

        df = processed_dfs
                    
                    

    # %%
    ''' Defining tab contents for Sample page for different operating mode '''
    if prediction_mode == True:
        if reference == False:
            sample_tab_content = {}
            for x in range(0,len(names_compared) - 1,2):
                data = {}
                f_name1 = names_compared[x]
                f_name2 = names_compared[x+1]
                min_len = min(len(sent_tokens[f_name1]),len(sent_tokens[f_name2]))
                data[f_name1] = sent_tokens[f_name1][0:min_len]
                data[f_name2] = sent_tokens[f_name2][0:min_len]
                df = pd.DataFrame(data)
                # original_data = df.copy(deep=True)
                col = []
                col += [{'name':f_name1[0:len(f_name1)-1],'id':f_name1}]
                col += [{'name':f_name2[0:len(f_name2)-1],'id':f_name2}]
                details = [html.H5('Samples Details')]
                details += [html.Div(id = 'original '+ f_name1)]
                details += [html.Div(id = 'original '+ f_name2)]
                details += [html.Div(id = 'details_datatable' + str(x))]
                hidden = []
                hidden += [html.Div(id = 'hidden ' + f_name1)]
                hidden += [html.Div(id = 'hidden ' + f_name2)]
                sample_tab_content[f_name1[0:len(f_name1)-1]+"->"+f_name2[0:len(f_name2)-1]] = dbc.Card(dbc.CardBody(html.Div([
                    html.Div(style = {'display':'none'},id = 'names' + str(x),children = [f_name1,f_name2]),
                    dash_table.DataTable(
                        id='table' + str(x),
                        columns=col,
                        data=df.to_dict('records'),
                        style_table={'width': '100%'},
                        row_selectable='single',  # Allow multiple row selection
                        selected_rows=[],
                        page_size = PAGE_SIZE,
                        page_current = 0,
                        # style_data_conditional=[
                        #     {
                        #         'if': {'row_index': 0},  # Apply the style to the first row
                        #         'display': 'none',       # Hide the first row
                        #     },
                        # ],
                        style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                                style_header={
                                    'color': 'text-primary',
                                    'text_align': 'center',
                                    'height': 'auto',
                                    'whiteSpace': 'normal',
                                },
                                css=[
                                    {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                                    {'selector': '.column-header--hide', 'rule': 'display: none'},
                                ],  # Initialize with no selected rows
                    ),
                    
                    # Hidden div to store original data
                    html.Div(id='hidden-data' + str(x), style={'display': 'none'}),
                    
                    # Page for displaying original text
                    html.Div(id='detail-page' + str(x), children = details, style={'display': 'none'})
                ])),className = 'mt-3')
            
                @app.callback(
                    [Output('hidden-data' + str(x), 'children'),
                    Output('detail-page' + str(x), 'style')],
                    Input('table' + str(x), 'selected_rows'),
                    State('table' + str(x), 'data'),
                    State('names' + str(x),'children')
                )
                def store_and_display_original_text(selected_rows,data,names):
                    if not selected_rows:
                        return [None, {'display': 'none'}]
                    original = []
                    for x in names:
                        original += [data[row][x] for row in selected_rows]
                    return [original, {'display': 'block'}]

                # Callback to display original text
                output = []
                output += [Output('original ' + f_name1,'children')]
                output += [Output('original ' + f_name2,'children')]
                @app.callback(
                    output,
                    Output('details_datatable' + str(x),'children'),
                    Input('hidden-data' + str(x), 'children'),
                    State('names' + str(x),'children')
                )
                def display_original_text(original_texts,names):
                    
                    if original_texts == None:
                        return ['' for x in range(len(names) + 1)]
                    
                    
                    metrics_data = []
                    for x in range(len(names)):
                        metrics_data += [{"Metric" : "character count" + "(" + names[x] + ')' , "Value" : len(original_texts[x])}]
                        metrics_data += [{"Metric" : "word count(" + names[x] +")" , "Value" : word_count(original_texts[x])}]

                    # Convert the data to a DataFrame
                    df_metrics = pd.DataFrame(metrics_data)

                    temp1 = html.Div(dash_table.DataTable(
                        id='metrics-table',
                        columns=[
                            {"name": "Metric", "id": "Metric"},
                            {"name": "Value", "id": "Value"}
                        ],
                        data=df_metrics.to_dict('records'),
                        style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                                style_header={
                                    'color': 'text-primary',
                                    'text_align': 'center',
                                    'height': 'auto',
                                    'whiteSpace': 'normal',
                                },
                                css=[
                                    {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                                    {'selector': '.column-header--hide', 'rule': 'display: none'},
                                ],
                    ))
                    temp2 = [html.Ul([html.Li(original_texts[x])]) for x in range(len(names))]
                    temp2.append(temp1)
                    return tuple(temp2)
                    return tuple([html.Ul([html.Li(original_texts[x])]) for x in range(len(names))])
                def word_count(text):
                    tokenizer = TreebankWordTokenizer()
                    tokens = tokenizer.tokenize(text)
                    return len(tokens)
                    
                def cosine_similarity(embedding1, embedding2):
                    dot_product = np.dot(embedding1, embedding2)
                    norm1 = np.linalg.norm(embedding1)
                    norm2 = np.linalg.norm(embedding2)
                    return dot_product / (norm1 * norm2)
            samples_layout = []
            samples_layout += [dbc.Row(dbc.Col(html.H5(children='Sample Statistics'), className='text-secondary'), className='mt-3')]
            samples_layout += [html.Div(
                [
                    dbc.Tabs(
                        tab_labels,
                        id="sample_tabs",
                        active_tab=names_compared[0][0:len(names_compared[0])-1] + '->' + names_compared[1][0:len(names_compared[1])-1],
                    ),
                    html.Div(id="samples-tab-content"),
                ]
            )]
            samples_layout = html.Div(samples_layout)
            @app.callback(Output("samples-tab-content", "children"), [Input("sample_tabs", "active_tab")])
            def switch_tab(at):
                return sample_tab_content[at]
        else:
            sample_tab_content = {}
            names_list = {}
            for x in range(0,len(names_compared) - 2,3):
                f_name1 = names_compared[x]
                f_name2 = names_compared[x+1]
                f_name3 = names_compared[x+2]
                names_list[x] = [f_name1,f_name2,f_name3]
                col = []
                col += [{'name':f_name1[0:len(f_name1)-1],'id':f_name1}]
                col += [{'name':f_name2[0:len(f_name2)-1],'id':f_name2}]
                col += [{'name':"Translated " + f_name3[0:len(f_name3)-1],'id':f_name3}]
                col += [{'name':'Bleu Score','id':'Bleu Score','type': 'numeric'}]
                col += [{'name':'Labse Score','id':'LabSe Score','type': 'numeric'}]
                col += [{'name':'Chrf Score','id':'Chrf Score','type': 'numeric'}]
                col += [{'name':'Chrf++ Score','id':'Chrf++ Score','type': 'numeric'}]
                # col += [{'name':'BleuRT Score','id':'BleuRT Score','type':'numeric'}]
                
                
                details = [html.H5('Samples Details')]
                details += [html.Div(id = 'original '+ f_name1)]
                details += [html.Div(id = 'original '+ f_name2)]
                details += [html.Div(id = 'original '+ f_name3)]
                details += [html.Div(id = 'details_datatable' + str(x))]
                hidden = []
                hidden += [html.Div(id = 'hidden ' + f_name1)]
                hidden += [html.Div(id = 'hidden ' + f_name2)]
                hidden += [html.Div(id = 'hidden ' + f_name3)]
                sample_tab_content[f_name1[0:len(f_name1)-1]+"->"+f_name2[0:len(f_name2)-1]] = dbc.Card(dbc.CardBody(html.Div([
                    html.Div(style = {'display':'none'},id = 'names' + str(x),children = [f_name1,f_name2,f_name3,x]),
                    dash_table.DataTable(
                        id='table' + str(x),
                        columns=col,
                        data=df['translation ' + str(x)].to_dict('records'),
                        style_table={'width': '100%'},
                        row_selectable='single',  # Allow multiple row selection
                        selected_rows=[],
                        page_size = PAGE_SIZE,
                        page_current = 0,
                        sort_action='native',

                        
                        style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                                style_header={
                                    'color': 'text-primary',
                                    'text_align': 'center',
                                    'height': 'auto',
                                    'whiteSpace': 'normal',
                                },
                                css=[
                                    {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                                    {'selector': '.column-header--hide', 'rule': 'display: none'},
                                ], 
                    ),
                    html.Div(id='names_div_' + str(x), children=[f_name1, f_name2, f_name3], style={'display': 'none'}),
                    
                    # Hidden div to store original data
                    html.Div(id='hidden-data' + str(x), style={'display': 'none'}),
                    
                    # Page for displaying original text
                    html.Div(id='detail-page' + str(x), children = details, style={'display': 'none'})
                ])),className = 'mt-3')

                # Callback to store selected rows and display detail page
                # @app.callback(
                #     [Output('hidden-data' + str(x), 'children'),
                #     Output('detail-page' + str(x), 'style')],
                #     Input('table' + str(x), 'selected_rows'),
                #     State('table' + str(x), 'data'),
                #     State('names' + str(x),'children')
                # )
                # def store_and_display_original_text(selected_rows, data,names):
                #     if not selected_rows:
                #         return [None, {'display': 'none'}]
                #     original = []
                #     for x in names:
                #         original += [data[row][x] for row in selected_rows]
                #     return [original, {'display': 'block'}]

                # Callback to display original text
                output = []
                output += [Output('original ' + f_name1,'children')]
                output += [Output('original ' + f_name2,'children')]
                output += [Output('original ' + f_name3,'children')]
                @app.callback(
                    output,
                    Output('details_datatable' + str(x),'children'),
                    Output('detail-page' + str(x), 'style'),
                    Input('table' + str(x), 'selected_rows'),
                    State('names' + str(x),'children')
                )
                def display_original_text(selected_row,names):
                    if len(selected_row) == 0:
                        temp = ['' for x in range(len(names))]
                        temp += [{'display': 'none'}]
                        return temp
                    
                    selected_row = selected_row[0]
                    key = names[-1]
                    
                    metrics_data = []
                    for x in range(len(names)-1):
                        if(x == 2):
                            metrics_data += [{"Metric" : "character count" + "(translated " + names[x][0:len(names[x])-1] + ')' , "Value" : len(df['translation ' + str(key)][names[x]][selected_row])}]
                            metrics_data += [{"Metric" : "word count(translated " + names[x][0:len(names[x])-1] +")" , "Value" : word_count(df['translation ' + str(key)][names[x]][selected_row])}]
                        else:
                            metrics_data += [{"Metric" : "character count" + "(" + names[x][0:len(names[x])-1] + ')' , "Value" : len(df['translation ' + str(key)][names[x]][selected_row])}]
                            metrics_data += [{"Metric" : "word count(" + names[x][0:len(names[x])-1] +")" , "Value" : word_count(df['translation ' + str(key)][names[x]][selected_row])}]
                    metrics_data += [{"Metric" : "Bleu Score(" + names[0][0:len(names[0])-1] +", "+ names[1][0:len(names[1]) - 1] + ")" , "Value" : df['translation ' + str(key)]['Bleu Score'][selected_row]}]
                    metrics_data += [{"Metric" : "Labse Score(" + names[0][0:len(names[0])-1] +", "+ names[1][0:len(names[1]) - 1] + ")" , "Value" : df['translation ' + str(key)]['LabSe Score'][selected_row]}]
                    metrics_data += [{"Metric" : "Chrf Score(" + names[0][0:len(names[0])-1] +", "+ names[1][0:len(names[1]) - 1] + ")" , "Value" : df['translation ' + str(key)]['Chrf Score'][selected_row]}]
                    metrics_data += [{"Metric" : "Chrf++ Score(" + names[0][0:len(names[0])-1] +", "+ names[1][0:len(names[1]) - 1] + ")" , "Value" : df['translation ' + str(key)]['Chrf++ Score'][selected_row]}] 
                    # metrics_data += [{"Metric" : "BleuRT Score(" + names[0][0:len(names[0])-1] +", "+ names[1][0:len(names[1]) - 1] + ")" , "Value" : df['translation ' + str(key)]['BleuRT Score'][selected_row]}]

                    # Convert the data to a DataFrame
                    df_metrics = pd.DataFrame(metrics_data)

                    temp1 = html.Div(dash_table.DataTable(
                        id='metrics-table',
                        columns=[
                            {"name": "Metric", "id": "Metric"},
                            {"name": "Value", "id": "Value"}
                        ],
                        data=df_metrics.to_dict('records'),
                        style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                                style_header={
                                    'color': 'text-primary',
                                    'text_align': 'center',
                                    'height': 'auto',
                                    'whiteSpace': 'normal',
                                },
                                css=[
                                    {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                                    {'selector': '.column-header--hide', 'rule': 'display: none'},
                                ],
                    ))
                    # temp2 = [html.Ul([html.Li(original_texts[x])]) for x in range(len(names_compared))]
                    temp2 = [html.Ul([html.Li(df["translation "+str(key)][names[x]][selected_row])]) for x in range(len(names)-1)]
                    temp2.append(temp1)
                    temp2.append({'display': 'block'})
                    return tuple(temp2)
                def word_count(text):
                    tokenizer = TreebankWordTokenizer()
                    tokens = tokenizer.tokenize(text)
                    return len(tokens)
                    
                def cosine_similarity(embedding1, embedding2):
                    dot_product = np.dot(embedding1, embedding2)
                    norm1 = np.linalg.norm(embedding1)
                    norm2 = np.linalg.norm(embedding2)
                    return dot_product / (norm1 * norm2)
            samples_layout = []
            samples_layout += [dbc.Row(dbc.Col(html.H5(children='Sample Statistics'), className='text-secondary'), className='mt-3')]
            samples_layout += [html.Div(
                [
                    dbc.Tabs(
                        tab_labels,
                        id="sample_tabs",
                        active_tab=names_compared[0][0:len(names_compared[0])-1] + '->' + names_compared[1][0:len(names_compared[1])-1],
                    ),
                    html.Div(id="samples-tab-content"),
                ]
            )]
            samples_layout = html.Div(samples_layout)
            @app.callback(Output("samples-tab-content", "children"), [Input("sample_tabs", "active_tab")])
            def switch_tab(at):
                return sample_tab_content[at]



    # %%
    if prediction_mode == False:
        data = {}
        min_len = len(sent_tokens[names_compared[0]])
        for x in range(1,len(names_compared) + 1):
            f_name = names_compared[x-1]
            min_len = min(min_len,len(sent_tokens[f_name]))
        for x in range(1,len(names_compared) + 1):
            f_name = names_compared[x-1]
            data[f_name] = sent_tokens[f_name][0:min_len]

        df = pd.DataFrame(data)

        # Store a copy of the original data
        original_data = df.copy(deep=True)

        # Define the app layout
        col = []
        for x in names_compared:
            col +=[{'name':x,'id':x}]
        details = [html.H5('Samples Details')]
        for x in names_compared:
            details += [html.Div(id = 'original ' + x)]
        details += [html.Div(id = 'details_datatable')]
        hidden = []
        for x in names_compared:
            hidden += [html.Div(id = 'hidden ' + x)]

        samples_layout = html.Div([
            dash_table.DataTable(
                id='table',
                columns=col,
                data=df.to_dict('records'),
                style_table={'width': '100%'},
                row_selectable='single',  # Allow multiple row selection
                selected_rows=[],
                page_size = PAGE_SIZE,
                page_current = 0,
                style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                        style_header={
                            'color': 'text-primary',
                            'text_align': 'center',
                            'height': 'auto',
                            'whiteSpace': 'normal',
                        },
                        css=[
                            {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                            {'selector': '.dash-filter--case', 'rule': 'display: none'},
                            {'selector': '.column-header--hide', 'rule': 'display: none'},
                        ],  # Initialize with no selected rows
            ),
            
            # Hidden div to store original data
            html.Div(id='hidden-data', style={'display': 'none'}),
            
            # Page for displaying original text
            html.Div(id='detail-page', children = details, style={'display': 'none'})
        ])


        # Callback to store selected rows and display detail page
        @app.callback(
            [Output('hidden-data', 'children'),
            Output('detail-page', 'style')],
            Input('table', 'selected_rows'),
            State('table', 'data')
        )
        def store_and_display_original_text(selected_rows, data):
            if not selected_rows:
                return [None, {'display': 'none'}]
            original = []
            for x in names_compared:
                original += [original_data.iloc[row][x] for row in selected_rows]
            return [original, {'display': 'block'}]

        # Callback to display original text
        output = []
        for x in names_compared:
            output +=[Output('original '+x,'children')]
        @app.callback(
            output,
            Output('details_datatable','children'),
            Input('hidden-data', 'children')
        )
        def display_original_text(original_texts):
            if original_texts == None:
                return ['' for x in range(len(names_compared) + 1)]
            # embeddings = model.encode(original_texts)
            # tokenizer = TreebankWordTokenizer()
            # similarity_score_labse = []
            # for x in range(1,len(original_texts)):
            #     similarity_score_labse += [cosine_similarity(embeddings[0],embeddings[x])]
            # similarity_score_bleu = []
            # for x in range(1,len(original_texts)):
            #     reference = [original_texts[0]]
            #     candidate = original_texts[x]
            #     reference = [nltk.word_tokenize(ref) for ref in reference]
            #     candidate = nltk.word_tokenize(candidate)
            #     similarity_score_bleu += [sentence_bleu(reference,candidate)]#,smoothing_function=SmoothingFunction().method1)]
            
            metrics_data = []
            for x in range(len(names_compared)):
                metrics_data += [{"Metric" : "character count" + "(" + names_compared[x] + ')' , "Value" : len(original_texts[x])}]
                metrics_data += [{"Metric" : "word count(" + names_compared[x] +")" , "Value" : word_count(original_texts[x])}]
            # for x in range(len(similarity_score_bleu)):
            #     metrics_data += [{"Metric" : "Bleu Score(" + names_compared[0] +", "+ names_compared[x + 1] + ")" , "Value" : similarity_score_bleu[x]}]
            # for x in range(len(similarity_score_labse)):
            #     metrics_data += [{"Metric" : "labse Score(" + names_compared[0] +", "+ names_compared[x + 1] + ")" , "Value" : similarity_score_labse[x]}]
            
                

            # Convert the data to a DataFrame
            df_metrics = pd.DataFrame(metrics_data)

            temp1 = html.Div(dash_table.DataTable(
                id='metrics-table',
                columns=[
                    {"name": "Metric", "id": "Metric"},
                    {"name": "Value", "id": "Value"}
                ],
                data=df_metrics.to_dict('records'),
                style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                        style_header={
                            'color': 'text-primary',
                            'text_align': 'center',
                            'height': 'auto',
                            'whiteSpace': 'normal',
                        },
                        css=[
                            {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                            {'selector': '.dash-filter--case', 'rule': 'display: none'},
                            {'selector': '.column-header--hide', 'rule': 'display: none'},
                        ],
            ))
            temp2 = [html.Ul([html.Li(original_texts[x])]) for x in range(len(names_compared))]
            temp2.append(temp1)
            return tuple(temp2)
            return tuple([html.Ul([html.Li(original_texts[x])]) for x in range(len(names_compared))])
        def word_count(text):
            tokenizer = TreebankWordTokenizer()
            tokens = tokenizer.tokenize(text)
            return len(tokens)
            
        def cosine_similarity(embedding1, embedding2):
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)


    # %%
    comparison_layout = [
        dbc.Row(dbc.Col(html.H5('comparison'), class_name='text-secondary'), class_name='mt-3')
    ]

    # %%
    if comparison_mode:
        app.layout = html.Div(
            [
                dcc.Location(id='url', refresh=False),
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                        dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                        dbc.NavItem(dbc.NavLink('Comparison tool', id='comp_tool', href='/comparison')),
                    ],
                    brand=html.Div(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src='/assets/images/logo.png', height='30px'), width='auto'),
                                dbc.Col(dbc.NavbarBrand('Text Data Explorer'), width='auto'),
                            ],
                            align='center',
                        )
                    ),
                    sticky='top',
                    color='orange',
                    dark=True,
                ),
                dbc.Container(id='page-content'),
            ]
        )
    else:
        app.layout = html.Div(
            [
                dcc.Location(id='url', refresh=False),
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                        dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                    ],
                    brand='Text Data Explorer',
                    sticky='top',
                    color='orange',
                    dark=True,
                ),
                dbc.Container(id='page-content'),
            ]
        )

    # %%
    if comparison_mode:

        @app.callback(
            [
                Output('page-content', 'children'),
                Output('stats_link', 'active'),
                Output('samples_link', 'active'),
                Output('comp_tool', 'active'),
            ],
            [Input('url', 'pathname')],
        )
        def nav_click(url):
            if url == '/samples':
                return [samples_layout, False, True, False]
            elif url == '/comparison':
                return [comparison_layout, False, False, True]
            else:
                return [stats_layout, True, False, False]


    else:

        @app.callback(
            [Output('page-content', 'children'), Output('stats_link', 'active'), Output('samples_link', 'active'),],
            [Input('url', 'pathname')],
        )
        def nav_click(url):
            if url == '/samples':
                return [samples_layout, False, True]
            else:
                return [stats_layout, True, False]
    
    print(tab_content.keys())
    app.run_server(port = 8063,debug=True)

    # end = datetime.datetime.now()

    # print('Elapsed time = ', end - start)

if __name__ == '__main__':
    main()



