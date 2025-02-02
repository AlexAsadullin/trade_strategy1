import os

def read_hmm_models():
    model_paths = os.listdir('/home/alex/BitcoinScalper/ML/ansamble/trained_models/Transformer')

def read_lstm_models():
    model_paths = os.listdir('/home/alex/BitcoinScalper/ML/ansamble/trained_models/Transformer')

def read_transformer_models():
    model_paths = os.listdir('/home/alex/BitcoinScalper/ML/ansamble/trained_models/Transformer')

def build_ansamble(model_types: dict):
    """
    model_types alike 
    {'momentum': 'lstm',
    'overlap': 'hmm',
    'trend': 'transformer',
    'some indicator': 'some ai',
    }
    """
