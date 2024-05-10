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
# import dask.dataframe as dd
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
def main(args = None):

    starttime = datetime.datetime.now()
    print('transliteration_demo')
    start = datetime.datetime.now()
    def parse_args():
        parser = argparse.ArgumentParser(description='Text Data Explorer')
        parser.add_argument('--port', default='8050', help='serving port for establishing connection')
        parser.add_argument('--data','-d',help = 'Path to the data.json file that contains the NER dataset')
        parser.add_argument(
            '--comparison_mode',
            '-cm',
            default = True,
            type = bool,
            choices = [True,False]
        )
        parser.add_argument(
            '--reference',
            '-r',
            default = True,
            type = bool,
            choices = [True,False]
        )
        parser.add_argument(
            '--names_compared',
            '-nc',
            nargs = '+',
            type=str,
            help='names of the three fields that will be compared, example: pred_text_contextnet pred_text_conformer. "pred_text_" prefix IS IMPORTANT!',
        )
        parser.add_argument(
            '--total_tokens',
            '-tt',
            default = 10000,
            type = int
        )

        
        args = parser.parse_args()

        return args
    if(args == None):
        args = parse_args()

    json_data = args.data
    json_data = os.path.abspath(os.path.join("transliteration", json_data))
    names_compared = args.names_compared
    comparison_mode = args.comparison_mode
    reference_mode = args.reference
    total_tokens = args.total_tokens

    with open(json_data, 'r', encoding='utf8') as f:
        json_content = f.read()
    parsed_data = json.loads(json_content)

    def load_data(data_filename,comparison_mode,names_compared,total):
        word_count = total
        char_distribution = {}
        vocab = {}
        avg_length = {}
        for x in names_compared:
            char_distribution[x] = {}
        df = pd.read_csv(data_filename,names = names_compared,nrows = total)
        print(df.head())
        for x in names_compared:
            length = 0
            for y in range(len(df)):
                words = df.iloc(0)[y][x]
                length += len(words)
                for ch in words:
                    if(ch in char_distribution[x].keys()):
                        char_distribution[x][ch] += 1
                    else:
                        char_distribution[x][ch] = 1
            avg_length[x] = length/len(df[x])
        for x in names_compared:
            vocab[x] = sorted(char_distribution[x].keys())


        return df,char_distribution,word_count,vocab,avg_length



    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css'],
        title="Text Data Explorer",
        assets_folder = 'assets/'
    )


    count = 0
    data = {}
    for key in parsed_data:
        translation_pair = '-'.join(names_compared[count:count + 2])
        print(translation_pair)
        temp = load_data(parsed_data[key],True,names_compared[count:count+3],total_tokens)
        data[translation_pair] = temp
        count += 3
    
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
    count = 0
    for x in data:
        char_distribution = {}
        char_distribution[names_compared[count]] = dict(sorted(data[x][1][names_compared[count]].items(),key = lambda x : x[1],reverse = True))
        char_distribution[names_compared[count+1]] = dict(sorted(data[x][1][names_compared[count + 1]].items(),key = lambda x : x[1],reverse = True))
        char_distribution[names_compared[count+2]] = dict(sorted(data[x][1][names_compared[count + 2]].items(),key = lambda x : x[1],reverse = True))
        figures_hist[x + str(count)] = ['Character Distribution ' + names_compared[count], plot_histogram(char_distribution[names_compared[count]],'characters','Frequency',20)]
        figures_hist[x + str(count + 1)] = ['Character_Distribution ' + names_compared[count + 1],plot_histogram(char_distribution[names_compared[count + 1]],'characters','Frequency',20)] 
        figures_hist[x + str(count + 2)] = ['Character_Distribution ' + names_compared[count + 2],plot_histogram(char_distribution[names_compared[count + 2]],'characters','Frequency',20)] 

    
    if comparison_mode == True:
        tab_labels = []
        for x in data:
            tab_labels += [
                        dbc.Tab(label = x,tab_id  = x)
                    ]
    
    if comparison_mode == True:
        tab_content = {}
        for x in range(0,len(names_compared)-1,3):
            f_name1 = names_compared[x]
            f_name2 = names_compared[x+1]
            f_name3 = names_compared[x+2]
            translation_pair = '-'.join([f_name1,f_name2])
            sub_tab_labels = []
            sub_tab_labels += [dbc.Tab(label = f_name1,tab_id = f_name1)]
            sub_tab_labels += [dbc.Tab(label = f_name2,tab_id = f_name2)]
            sub_tab_labels += [dbc.Tab(label = f_name3,tab_id = f_name3)]
            sub_tab_content = {}
            sub_tab_content[f_name1] = dbc.Card(
                dbc.CardBody(
                    [
                dbc.Row(dbc.Col(html.H6(children = f_name1),className = 'text-secondary'),className = 'mt-3'),
                dbc.Row(
                    [
                        dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                        dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                    ],
                    className='bg-light mt-2 rounded-top border-top border-start border-end',
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5(
                                data[translation_pair][2],
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                        dbc.Col(
                            html.H5(
                                '{} characters'.format(len(data[translation_pair][1][f_name1])),
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                    ],
                    className='bg-light rounded-bottom border-bottom border-start border-end',
                ),
                dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
            dbc.Row(
                dbc.Col(html.Div('{}'.format(sorted(data[translation_pair][3][f_name1]))),), class_name='mt-2 bg-light font-monospace rounded border'
            ),
                dbc.Row(dbc.Col(html.H5(figures_hist[translation_pair + str(x)][0]), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x), figure=figures_hist[translation_pair + str(x)][1])),),
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
                
            ]
                )
            )
            sub_tab_content[f_name2] = dbc.Card(
                dbc.CardBody(
                    [
                dbc.Row(dbc.Col(html.H6(children = f_name2),className = 'text-secondary'),className = 'mt-3'),
                dbc.Row(
                    [
                        dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                        dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                    ],
                    className='bg-light mt-2 rounded-top border-top border-start border-end',
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5(
                                data[translation_pair][2],
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                        dbc.Col(
                            html.H5(
                                '{} characters'.format(len(data[translation_pair][1][f_name2])),
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                    ],
                    className='bg-light rounded-bottom border-bottom border-start border-end',
                ),
                dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
            dbc.Row(
                dbc.Col(html.Div('{}'.format(sorted(data[translation_pair][3][f_name2]))),), class_name='mt-2 bg-light font-monospace rounded border'
            ),
                dbc.Row(dbc.Col(html.H5(figures_hist[translation_pair + str(x+1)][0]), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x+1), figure=figures_hist[translation_pair + str(x+1)][1])),),
                dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                dbc.Row(html.Div([dcc.Slider(
                    1,
                    20,
                    step=None,
                    value=10,
                    marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                    id='char_slider' + str(x + 1),
                    className = 'slider'
                )],style = {}
                    )),
                
            ]
                )
            )
            sub_tab_content[f_name3] = dbc.Card(
                dbc.CardBody(
                    [
                dbc.Row(dbc.Col(html.H6(children = f_name3),className = 'text-secondary'),className = 'mt-3'),
                dbc.Row(
                    [
                        dbc.Col(html.Div('Word Count', className='text-secondary'), width=3, className='border-end'),
                        dbc.Col(html.Div('Vocabulary', className='text-secondary'), width=3, className='border-end'),
                    ],
                    className='bg-light mt-2 rounded-top border-top border-start border-end',
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5(
                                data[translation_pair][2],
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                        dbc.Col(
                            html.H5(
                                '{} characters'.format(len(data[translation_pair][1][f_name3])),
                                className='text-center p-1',
                                style={'color': 'cornflowerblue', 'opacity': 0.7},
                            ),
                            width=3,
                            className='border-end',
                        ),
                    ],
                    className='bg-light rounded-bottom border-bottom border-start border-end',
                ),
                dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
            dbc.Row(
                dbc.Col(html.Div('{}'.format(sorted(data[translation_pair][3][f_name3]))),), class_name='mt-2 bg-light font-monospace rounded border'
            ),
                dbc.Row(dbc.Col(html.H5(figures_hist[translation_pair + str(x+2)][0]), class_name='text-secondary'), class_name='mt-3'),
                dbc.Row(dbc.Col(dcc.Graph(id='char_dist_graph' + str(x + 2), figure=figures_hist[translation_pair + str(x + 2)][1])),),
                dbc.Row(dbc.Col(html.H6('Top K characters Slider'),class_name = 'text-secondary'),class_name = 'mt-3'),
                dbc.Row(html.Div([dcc.Slider(
                    1,
                    20,
                    step=None,
                    value=10,
                    marks={str(num): str(num) for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                    id='char_slider' + str(x + 2),
                    className = 'slider'
                )],style = {}
                    )),
                
            ]
                )
            )
            tab_content[translation_pair] = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(sub_tab_content[f_name1], label=f_name1, tab_id=f_name1),
                            dbc.Tab(sub_tab_content[f_name2], label=f_name2, tab_id=f_name2),
                            dbc.Tab(sub_tab_content[f_name3], label=f_name3, tab_id=f_name3),
                        ],
                        active_tab=f_name1,
                    )
                ]
            ),
            className="mt-3",
        )

            active = translation_pair

    for x in range(0,len(names_compared)):
        f_name = names_compared[x]
        # df = pd.DataFrame(list(word_distribution[f_name].items()), columns=['Word', 'Count'])
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
    
    def calculate_character_accuracy(reference, prediction):
        num_correct = sum(1 for c1, c2 in zip(reference, prediction) if c1 == c2)
        accuracy = num_correct / len(reference) * 100
        return accuracy

    from nltk.metrics import edit_distance

    from sacrebleu import BLEU

    import evaluate
    rouge = evaluate.load('rouge')

    def calculate_jaccard_score(reference,candidate):
        total = len(reference) + len(candidate)
        curr = 0
        ref_dict = {}
        can_dict = {}
        for x in reference:
            if(x in ref_dict.keys()):
                ref_dict[x] += 1
            else:
                ref_dict[x] = 1
        for x in candidate:
            if(x in can_dict.keys()):
                can_dict[x] += 1
            else:
                can_dict[x] = 1
        for x in ref_dict.keys():
            ref = ref_dict[x]
            can = 0
            if x in can_dict.keys():
                can = can_dict[x]
            curr += 2*min(ref,can)
        return curr/total
    
    # from datasets import load_metric

    # metric = evaluate.load("cer")


    # 
    def character_error_rate(reference, candidate):
        """
        Computes the Character Error Rate (CER) between a reference and a candidate string.
        
        Args:
        reference (str): The reference string.
        candidate (str): The candidate string.
        
        Returns:
        float: The Character Error Rate (CER) between the reference and candidate strings.
        """
        # Initialize variables to count errors and total characters
        num_errors = 0
        total_chars = max(len(reference), len(candidate))
        
        # Compare characters at each position
        for ref_char, cand_char in zip(reference, candidate):
            if ref_char != cand_char:
                num_errors += 1
        
        # Add the remaining characters if one string is longer than the other
        num_errors += abs(len(reference) - len(candidate))
        
        # Calculate CER
        cer = num_errors / total_chars if total_chars > 0 else 0.0
        
        return cer
    
    ## Dask Implementation

    from dask import delayed,compute

    def process_chunk(chunk):
        count = 0

        accuracy = []
        lev = []
        jaccard = []
        cer = []
        total_accuracy = 0
        total_jaccard = 0
        total_lev = 0
        total_cer = 0
        for x in chunk.iterrows():
            reference = x[1][f_name3]
            candidate = x[1][f_name2]
            a = calculate_character_accuracy(reference,candidate)
            b = edit_distance(reference,candidate)
            c = calculate_jaccard_score(reference,candidate)
            d = character_error_rate(reference,candidate)
            count += 1
            total_accuracy += a
            total_lev += b
            total_jaccard += c
            total_cer += d
            accuracy.append(a)
            lev.append(b)
            jaccard.append(c)
            cer.append(d)
        chunk['Accuracy'] = accuracy
        chunk['Levenshtein Distance'] = lev
        chunk['Jaccard Similarity'] = jaccard
        chunk['CER'] = cer

        return chunk,count,total_accuracy,total_lev,total_jaccard,total_cer

    def process_dataframe(df):
        chunk_size = 5000  # Define your chunk size here
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        print(len(chunks))

        delayed_chunks = [delayed(process_chunk)(chunk) for chunk in chunks]
        processed_chunks = compute(*delayed_chunks, scheduler="threads")
        final_chunks = []
        final_count = 0
        final_accuracy = 0
        final_lev = 0
        final_jacc = 0
        final_cer = 0
        for x in range(len(processed_chunks)):
            final_chunks += [processed_chunks[x][0]]
            final_count += processed_chunks[x][1]
            final_accuracy +=processed_chunks[x][2]
            final_lev += processed_chunks[x][3]
            final_jacc += processed_chunks[x][4]
            final_cer += processed_chunks[x][5]

            

        return pd.concat(final_chunks),final_count,final_accuracy,final_lev,final_jacc,final_cer


    samples_data = {}
    for x in range(0,len(names_compared)-1,3):
        f_name1 = names_compared[x]
        f_name2 = names_compared[x+1]
        f_name3 = names_compared[x+2]
        translation_pair = '-'.join([f_name1,f_name2])
        samples_data[translation_pair] = {}
        samples_data[translation_pair]['Accuracy'] = []
        samples_data[translation_pair]['Levenshtein Distance'] = []
        samples_data[translation_pair]['Jaccard Similarity'] = []
        samples_data[translation_pair]['CER'] = []
        df = data[translation_pair][0]
        # process_dataframe(df)

        processed_df,count,total_accuracy,total_lev,total_jaccard,total_cer = process_dataframe(df)

        samples_data[translation_pair] = processed_df
        # total_cer = 0
        # temp_cer = []
        # for x in samples_data[translation_pair].iterrows():
        #     reference = x[1][f_name3]
        #     candidate = x[1][f_name2]
        #     print(reference,candidate)
        #     d = metric.compute(references = reference,predictions = candidate)
        #     total_cer += d
        #     temp_cer.append(d)
        # samples_data[translation_pair]['CER'] = temp_cer

        # samples_data[translation_pair] = dict(samples_data[translation_pair])

            


    total_accuracy /= count
    total_lev /= count
    total_jaccard /= count
    total_cer /= count
    

    PAGE_SIZE = 10

    if comparison_mode == True:
        if reference_mode == True:
            sample_tab_content = {}
            for x in range(0,len(names_compared) - 1,3):
                f_name1 = names_compared[x]
                f_name2 = names_compared[x+1]
                f_name3 = names_compared[x+2]
                col = []
                col += [{'name':f_name1,'id':f_name1}]
                col += [{'name':f_name2,'id':f_name2}]
                col += [{'name':f_name3,'id':f_name3}]
                col += [{'name':'Accuracy','id':'Accuracy','type': 'numeric'}]
                col += [{'name':'Levenshtein Distance','id':'Levenshtein Distance','type':'numeric'}]
                col += [{'name':'Jaccard Similarity','id':'Jaccard Similarity','type':'numeric'}]
                col += [{'name':'CER','id':'CER','type':'numeric'}]

                translation_pair = '-'.join([f_name1,f_name2])
                # d = pd.DataFrame.from_dict(samples_data[translation_pair])
                # df = pd.merge(data[translation_pair][0],d,left_index=True,right_index=True)
                sample_tab_content[translation_pair] = dbc.Card(dbc.CardBody(html.Div([
                        html.Div(style = {'display':'none'},id = 'names' + str(x),children = [f_name1,f_name2]),
                        dash_table.DataTable(
                            id='table' + str(x),
                            columns=col,
                            data=samples_data[translation_pair].to_dict('records'),
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
                    
                    ])),className = 'mt-3')
    samples_layout = []
    samples_layout += [dbc.Row(dbc.Col(html.H5(children='Sample Statistics'), className='text-secondary'), className='mt-3')]
    samples_layout += [html.Div(
        [
            dbc.Tabs(
                tab_labels,
                id="sample_tabs",
                active_tab=names_compared[0] + '-' + names_compared[1],
            ),
            html.Div(id="samples-tab-content"),
        ]
    )]
    samples_layout = html.Div(samples_layout)
    @app.callback(Output("samples-tab-content", "children"), [Input("sample_tabs", "active_tab")])
    def switch_tab(at):
        return sample_tab_content[at]




    stats_layout = []
    stats_layout += [dbc.Row(dbc.Col(html.H5(children='Global Statistics'), className='text-secondary'), className='mt-3')]
    stats_layout += [dbc.Row(dbc.Col(html.H6(children = 'Dataset Level Statistics'),className = 'text-secondary'),className = 'mt-3')]
    stats_layout += [dbc.Row([dbc.Col(html.H6(children = 'Accuracy'),className='border-end'),
                            dbc.Col(html.H6(children = 'Levenshtein Distance'),className='border-end'),
                            dbc.Col(html.H6(children = 'Jaccard Similarity'),className='border-end'),
                            dbc.Col(html.H6(children = 'Character Error Rate'),className='border-end')
                                ],className='bg-light mt-2 rounded-top border-top border-start border-end')]
    stats_layout += [dbc.Row(
                            [
                                dbc.Col(
                                    html.H6(
                                        total_accuracy,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                                dbc.Col(
                                    html.H6(
                                        total_lev,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                                dbc.Col(
                                    html.H6(
                                        total_jaccard,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                                dbc.Col(
                                    html.H6(
                                        total_cer,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                            ],
                            className='bg-light mt-2 rounded-top border-top border-start border-end',

                        )]
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



    @app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
    def switch_tab(at):
        return tab_content[at]



    comparison_layout = []


    app.layout = html.Div(
            [
                dcc.Location(id='url', refresh=False),
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                        dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                        dbc.NavItem(dbc.NavLink('Comparison tool', id='comp_tool', href='/comparison')),
                    ],
                    brand=html.Div([
                        html.Img(src='/assets/images/logo.png', height='30px'),  # Replace 'logo.png' with your logo file path
                        html.Span("Text Data Explorer", style={'margin-left': '10px', 'vertical-align': 'middle'})
                    ]),
                    sticky='top',
                    color='orange',
                    dark=True,
                ),
                dbc.Container(id='page-content'),
            ]
        )
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
    endtime = datetime.datetime.now()

    elapsed_time = (endtime - starttime)
    print(elapsed_time.total_seconds())

    app.run(port = 8065,debug = True )

if __name__ == '__main__':
    main()
            



    




        



    
    

    

