
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

import dash
from dash import dcc, html, Input, Output
import dash_table
import spacy
from spacy.tokens import Doc, Span
from spacy import displacy

''' Defining Parser '''
def parse_args():
    parser = argparse.ArgumentParser(description='Text Data Explorer')
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument('--data','-d',help = 'Path to the data.json file that contains the NER dataset')
    

    args = parser.parse_args()

    return args

def main(args = None):
    start = datetime.datetime.now()
    # app.run_server(port = 8061,debug=True)

# %%
    PAGE_SIZE = 10

    # %%
    
    if(args == None):
        args = parse_args()
    data_path = os.path.abspath(os.path.join("NER", args.data))

    with open(data_path,'r',encoding = 'utf-8') as file:
        lines = file.readlines()
    dataset = []
    for line in lines:
        dataset.append(json.loads(line))
    predictions = [sentence['ner'] for sentence in dataset]
    true_labels = [sentence['true_ner'] for sentence in dataset]


    # Define a function to evaluate NER performance
    def evaluate_ner(predictions, true_labels):
        # Initialize counters
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        # Iterate over each sentence's predictions and true labels
        for pred_tags, true_tags in zip(predictions, true_labels):
            # Compare entire sequences of predicted and true labels
            for pred_tag, true_tag in zip(pred_tags, true_tags):
                if pred_tag == true_tag:
                    if pred_tag != "O":  # Exclude 'O' tags from true positives
                        tp += 1
                else:
                    if pred_tag != "O":
                        fp += 1
                    if true_tag != "O":
                        fn += 1

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate total number of named entities
        total_entities = sum(1 for tags in true_labels for tag in tags if tag != "O")

        # Calculate accuracy
        accuracy = tp / total_entities if total_entities > 0 else 0

        return round(precision,2), round(recall,2), round(f1_score,2), round(accuracy,2)

    precision, recall, f1_score, accuracy = evaluate_ner(predictions,true_labels)

    def calculate_entity_metrics(predictions, true_labels, entity_type):
        # Initialize counters
        true_positives = false_positives = false_negatives = true_negatives = 0

        # Iterate over each sentence's predictions and true labels
        for pred_tags, true_tags in zip(predictions, true_labels):
            # Compare entire sequences of predicted and true labels
            for pred_tag, true_tag in zip(pred_tags, true_tags):
                if true_tag == entity_type:
                    if pred_tag == entity_type:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if pred_tag == entity_type:
                        false_positives += 1
                    else:
                        true_negatives += 1

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate accuracy
        total_tokens = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total_tokens if total_tokens > 0 else 0

        return round(precision,2), round(recall,2), round(f1_score,2), round(accuracy,2)
    entity_data = []
    entity_type = []
    for sentence in dataset:
        for x in sentence['true_ner']:
            if x not in entity_type and x != 'O':
                entity_type.append(x)
    for type in entity_type:
        a,b,c,d = calculate_entity_metrics(predictions,true_labels,type)
        entity_data.append({'entity_type' : type , 'precision' : a , 'recall' : b, 'f1_score' : c, 'accuracy' : d })
    
    df = pd.DataFrame(entity_data)

    def calculate_sentence_metrics(predictions, true_labels):
        # Initialize lists to store metrics for each sentence
        sentence_precisions = []
        sentence_recalls = []
        sentence_f1_scores = []
        sentence_accuracies = []

        # Iterate over each sentence's predictions and true labels
        for i in range(len(predictions)):
            pred_tags = predictions[i]
            true_tags = true_labels[i]

            # Initialize counters for the current sentence
            true_positives = false_positives = false_negatives = true_negatives = 0

            # Compare entire sequences of predicted and true labels
            for j in range(len(pred_tags)):
                pred_tag = pred_tags[j]
                true_tag = true_tags[j]

                if true_tag != 'O':
                    if pred_tag == true_tag:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if pred_tag == true_tag:
                        true_negatives += 1
                    else:
                        false_positives += 1

            # Calculate precision, recall, and F1 score for the current sentence
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Calculate accuracy for the current sentence
            total_tokens = len(pred_tags)
            accuracy = (true_positives + true_negatives) / total_tokens if total_tokens > 0 else 0

            # Append metrics for the current sentence to the lists
            sentence_precisions.append(round(precision,2))
            sentence_recalls.append(round(recall,2))
            sentence_f1_scores.append(round(f1_score,2))
            sentence_accuracies.append(round(accuracy,2))

        return sentence_precisions, sentence_recalls, sentence_f1_scores, sentence_accuracies

    sentence_precisions, sentence_recalls, sentence_f1_scores, sentence_accuracies = calculate_sentence_metrics(predictions,true_labels)
    
    i = 0
    for data in dataset:
        data['precision'] = sentence_precisions[i]
        data['recall'] = sentence_recalls[i]
        data['f1_score'] = sentence_f1_scores[i]
        data['accuracy'] = sentence_accuracies[i]
        i += 1

    

    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css'],
        title="Text Data Explorer",
    )


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
    
    # Load a blank SpaCy model
    nlp = spacy.blank("xx")



    main_layout = []
    main_layout += [dbc.Row(dbc.Col(html.H5(children='Named Entity Recognizer Statistics'), className='text-secondary'), className='mt-3')]
    main_layout += [dbc.Row(dbc.Col(html.H6(children='Document Level statistics'),className='text-secondary'),className='mt-3')]
    main_layout += [dbc.Row(
                            [
                                dbc.Col(html.Div('Precision', className='text-secondary'), width=3, className='border-end'),
                                dbc.Col(html.Div('Recall', className='text-secondary'), width=3, className='border-end'),
                                dbc.Col(html.Div('F1-Score', className='text-secondary'), width=3, className='border-end'),
                                dbc.Col(html.Div('Accuracy', className='text-secondary'), width=3),
                            ],
                            className='bg-light mt-2 rounded-top border-top border-start border-end',
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5(
                                        precision,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                                dbc.Col(
                                    html.H5(
                                        recall,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                                dbc.Col(
                                    html.H5(
                                        f1_score,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                                dbc.Col(
                                    html.H5(
                                        accuracy * 100,
                                        className='text-center p-1',
                                        style={'color': 'cornflowerblue', 'opacity': 0.7},
                                    ),
                                    width=3,
                                    className='border-end',
                                ),
                            ],
                            className='bg-light mt-2 rounded-top border-top border-start border-end',

                        )]
    main_layout += [dbc.Row(dbc.Col(html.H6(children='Entity Level Statistics'),className='text-secondary'),className='mt-3')]

    # for x in entity_type:
    #     main_layout += [dbc.Row(dbc.Col(html.H6(children = x),className = 'text-secondary'),className='mt-3'),
    #                 dbc.Row(
    #                         [
    #                             dbc.Col(html.Div('Precision', className='text-secondary'), width=3, className='border-end'),
    #                             dbc.Col(html.Div('Recall', className='text-secondary'), width=3, className='border-end'),
    #                             dbc.Col(html.Div('F1-Score', className='text-secondary'), width=3, className='border-end'),
    #                             dbc.Col(html.Div('Accuracy', className='text-secondary'), width=3),
    #                         ],
    #                         className='bg-light mt-2 rounded-top border-top border-start border-end',
    #                     ),
    #                     dbc.Row(
    #                         [
    #                             dbc.Col(
    #                                 html.H5(
    #                                     ent_precision[x],
    #                                     className='text-center p-1',
    #                                     style={'color': 'cornflowerblue', 'opacity': 0.7},
    #                                 ),
    #                                 width=3,
    #                                 className='border-end',
    #                             ),
    #                             dbc.Col(
    #                                 html.H5(
    #                                     ent_recall[x],
    #                                     className='text-center p-1',
    #                                     style={'color': 'cornflowerblue', 'opacity': 0.7},
    #                                 ),
    #                                 width=3,
    #                                 className='border-end',
    #                             ),
    #                             dbc.Col(
    #                                 html.H5(
    #                                     ent_f1_score[x],
    #                                     className='text-center p-1',
    #                                     style={'color': 'cornflowerblue', 'opacity': 0.7},
    #                                 ),
    #                                 width=3,
    #                                 className='border-end',
    #                             ),
    #                             dbc.Col(
    #                                 html.H5(
    #                                     ent_accuracy[x] * 100,
    #                                     className='text-center p-1',
    #                                     style={'color': 'cornflowerblue', 'opacity': 0.7},
    #                                 ),
    #                                 width=3,
    #                                 className='border-end',
    #                             ),
    #                         ],
    #                         className='bg-light mt-2 rounded-top border-top border-start border-end',

    #                     ),

    #                     ]

    main_layout += [html.Div([
        dash_table.DataTable(
            id='table',
            columns=[
                {'name': 'Entity Type', 'id': 'entity_type'},
                {'name': 'Precision', 'id': 'precision'},
                {'name': 'Recall', 'id': 'recall'},
                {'name': 'F1 Score', 'id': 'f1_score'},
                {'name': 'Accuracy', 'id': 'accuracy'}
            ],
            data=df.to_dict('records')
        )
    ])]
    main_layout += [dbc.Row(dbc.Col(html.H6(children='Samples Table'),className='text-secondary'),className='mt-3')]
    count = 0
    main_layout += [html.Div([
        dash_table.DataTable(
            id='ner-table',
            columns=[
                {"name": "Sentence", "id": "sentence"},
                {"name": "Model's NER Tags", "id": "model_ner_tags"},
                {"name": "True NER Tags", "id": "true_ner_tags"},
                {"name": "Precision", "id" : "sent_precision"},
                {"name": "Recall", "id" : "sent_recall"},
                {"name": "F1_score", "id" : "sent_f1_score"},
                {"name": "Accuracy", "id" : "sent_accuracy"}
            ],
            data=[
                {"sentence": ' '.join(data["words"]), 
                "model_ner_tags": ' '.join(data["ner"]), 
                "true_ner_tags": ' '.join(data["true_ner"]),
                "sent_precision" : str(data['precision']),
                "sent_recall" : str(data['recall']),
                "sent_f1_score" : str(data['f1_score']),
                "sent_accuracy" : str(data['accuracy'])}
                for data in dataset[0:200000]
            ],
            row_selectable="single",
            selected_rows=[0],
            filter_action='custom',
            # virtualization = True,
            page_action = 'native',
            page_size=10, 
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '100px'}
            
        ),
        # html.Div([
        #     html.Div(id='model-visualization', className="six columns"),
        #     html.Div(id='true-visualization', className="six columns")
        # ], className="row")
    ])]
    main_layout +=[dbc.Row(dbc.Col(html.H6(children="Model's Visualization"),className='text-secondary'),className='mt-3')]
    main_layout += [dbc.Row(dbc.Col(html.Div(id = 'model-visualization',className='mt-3')))]
    main_layout += [dbc.Row(dbc.Col(html.H6(children='True Visualization'),className='text-secondary'),className='mt-3')]
    main_layout += [dbc.Row(dbc.Col(html.Div(id = 'true-visualization',className='mt-3')))]


    samples_layout = []
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
            return [main_layout, True, False, False]
    

    # Define callback to update visualizations
    @app.callback(
        [Output('model-visualization', 'children'),
        Output('true-visualization', 'children')],
        [Input('ner-table', 'selected_rows')]
    )
    def update_visualizations(selected_rows):
        if selected_rows:
            data = dataset[selected_rows[0]]
            # Create SpaCy Doc object with provided words
            doc = Doc(nlp.vocab, words=data["words"])
            
            # Add entity annotations to the SpaCy Doc object
            entities = data["ner"]
            entity_spans = []
            start = None
            for i, label in enumerate(entities):
                if label.startswith("B-"):
                    if start is not None:
                        entity_spans.append((start, i))
                    start = i
                elif label == "O" and start is not None:
                    entity_spans.append((start, i))
                    start = None
            if start is not None:
                entity_spans.append((start, len(entities)))

            for start, end in entity_spans:
                label = entities[start][2:]
                span = Span(doc, start, end, label=label)
                doc.ents = list(doc.ents) + [span]
            
            # Generate SpaCy visualization for model's NER tags
            model_html_output = displacy.render(doc, style="ent")
            
            # Generate SpaCy visualization for true NER tags
            true_doc = Doc(nlp.vocab, words=data["words"])
            true_entities = data["true_ner"]
            true_entity_spans = []
            start = None
            for i, label in enumerate(true_entities):
                if label.startswith("B-"):
                    if start is not None:
                        true_entity_spans.append((start, i))
                    start = i
                elif label == "O" and start is not None:
                    true_entity_spans.append((start, i))
                    start = None
            if start is not None:
                true_entity_spans.append((start, len(true_entities)))

            for start, end in true_entity_spans:
                label = true_entities[start][2:]
                span = Span(true_doc, start, end, label=label)
                true_doc.ents = list(true_doc.ents) + [span]
            
            true_html_output = displacy.render(true_doc, style="ent")
            
            return html.Div(dcc.Markdown([model_html_output], dangerously_allow_html=True)), html.Div(dcc.Markdown([true_html_output], dangerously_allow_html=True))
        else:
            return dash.no_update, dash.no_update

    app.run_server(port = 8084,debug=True)




if __name__ == '__main__':
    main()




    
    