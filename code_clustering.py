# python3 code2vec.py --load ../code2vec_Dataset/models/java14_model/saved_model_iter8.release --predict --export_code_vectors
from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
from tensorflow_model import Code2VecModel

config = Config(set_defaults=True, load_from_args=True, verify=True)
code_model = Code2VecModel(config)

predictor = InteractivePredictor(config, code_model)
predictor.predict()

code_model.close_session()