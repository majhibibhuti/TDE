
# Text Data Explorer

Text Data Explorer is a Python application designed for text analysis and exploration, with support for various natural language processing (NLP) tasks such as translation, transliteration, and Named Entity Recognition (NER). It provides a user-friendly interface for analyzing multilingual text data and evaluating machine learning models.

## Features

- **Translation Module:** Analyze and evaluate machine translation quality using metrics such as BLEU score, ChrF++, and LABSE score across multiple languages.
- **Transliteration Module:** Compute accuracy, Levenshtein distance, Jaccard similarity, and character error rate for transliteration tasks in different languages.
- **NER Module:** Perform Named Entity Recognition (NER) tasks and evaluate model performance using precision, recall, F1-score, and accuracy metrics.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/majhibibhuti/TDE.git
```

2. Navigate to the project directory:

```bash
cd text-data-explorer
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Translation Module

To use the Translation Module, run the main.py script with the following arguments:

```bash
python main.py Translation --port 8050 --manifest manifest.json --vocab vocab.txt --text_base_path /path/to/texts --names_compared pred_text_contextnet pred_text_conformer --prediction_mode True --comparison_mode True --total_tokens 10000 --reference True
```

- `--port`: The serving port for establishing connection.
- `--manifest`: Path to the JSON manifest file containing translation data.
- `--vocab`: Optional vocabulary file to highlight Out-of-Vocabulary (OOV) words.
- `--text_base_path`: A base path for relative paths in the manifest file.
- `--names_compared`: Names of the fields to compare (e.g., pred_text_contextnet pred_text_conformer).
- `--prediction_mode`: Whether prediction mode is enabled (True or False).
- `--comparison_mode`: Whether comparison mode is enabled (True or False).
- `--total_tokens`: The total number of tokens to process (-1 for unlimited).
- `--reference`: Whether to include reference field (True or False).

  Please make sure to give the data path inside the json file. For reference see the manifest.json file in translation folder.
  The data should be in the format specified in samanantar dataset. Please refer to the official doc of samanantar dataset.

### Transliteration Module

To use the Transliteration Module, run the main.py script with the following arguments:

```bash
python main.py Transliteration --port 8050 --data data.json --comparison_mode True --reference True --names_compared pred_text_contextnet pred_text_conformer --total_tokens 10000
```

- `--port`: The serving port for establishing connection.
- `--data`: Path to the data.json file containing the transliteration dataset.
- `--comparison_mode`: Whether comparison mode is enabled (True or False).
- `--reference`: Whether to include reference field (True or False).
- `--names_compared`: Names of the fields to compare (e.g., pred_text_contextnet pred_text_conformer).
- `--total_tokens`: The total number of tokens to process.

  Please make sure to give the data path inside the json file. For reference see the filepath.json file in tranliteration folder.
  The data should be in this format.

  | original_text   | true_transliteration | predicted_transliteration |
|-----------------|----------------------|---------------------------|
| shastragaar     | शस्त्रागार           | शस्त्रागार               |
| bindhya         | बिन्द्या             | बिन्द्या                 |
| kirankant       | किरणकांत            | किरणकांत                |
| yagyopaveet     | यज्ञोपवीत           | यज्ञोपवीत               |
| ratania         | रटानिया             | रटानिया                 |
| vaganyache      | वागण्याचे           | वागण्याचे               |
| deshbharamadhye | देशभरामध्ये        | देशभरामध्ये            |


### NER Module

To use the NER Module, run the main.py script with the following arguments:

```bash
python main.py NER --port 8050 --data data.json
```

- `--port`: The serving port for establishing connection.
- `--data`: Path to the data.json file containing the NER dataset.

  Please make sure to give the data as mentioned in the data.json file in NER Folder.
  The data should be in this format.
