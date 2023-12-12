from text_module.count_vectorizer import CountVectorizer
from text_module.tf_idf import IDFVectorizer
from text_module.usual_embedding import Usual_Embedding

def build_text_embbeding(config):
    if config['text_embedding']['type']=='count_vector':
        return CountVectorizer(config)
    if config['text_embedding']['type']=='tf_idf':
        return IDFVectorizer(config)
    if config['text_embedding']['type']=='usual_embedding':
        return Usual_Embedding(config)