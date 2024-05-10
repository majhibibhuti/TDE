import argparse
from transliteration import transliteration_demo
from translation import translation_demo
from NER import NER_demo
import dask.dataframe as dd

def main():
    parser = argparse.ArgumentParser(description="Python Script Runner")
    
    parser.add_argument("script", choices=["NER","Translation","Transliteration"], help="Choose which script to run")

    # command line arguments for NER

    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument('--data','-d',help = 'Path to the data.json file that contains the NER dataset')


    # command line arguments for Transliteration

    parser.add_argument(
        '--comparison_mode',
        '-cm',
        default = True,
        type = bool,
        choices = [True,False]
    )
    parser.add_argument(
        '--prediction_mode',
        '-pm',
        default=False,
        type = bool,
        help = 'prediction mode is on when the data is pairwise two languages with one as OG language and another as true translated language',
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
        help='names of the languages compared according to the module.',
    )
    parser.add_argument(
        '--total_tokens',
        '-tt',
        default = 10000,
        type = int
    )
    parser.add_argument(
            '--manifest', help='path to JSON manifest file',
        )
    parser.add_argument('--vocab', help='optional vocabulary to highlight OOV words')
    parser.add_argument(
            '--text_base_path',
            default=None,
            type=str,
            help='A base path for the relative paths in manifest. It defaults to manifest path.',
        )

    args,unknown = parser.parse_known_args()
    print(args)

    if args.script == "Transliteration":
        transliteration_demo.main(args)
    elif args.script == "NER":
        NER_demo.main(args)
    elif args.script == "Translation":
        translation_demo.main(args)

if __name__ == "__main__":
    main()
