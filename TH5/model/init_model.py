from model.rnn_model import createRNN_Model,RNN_Model
from model.lstm_model import createLSTM_Model, LSTM_Model
from model.rnn_attention_model import createRNN_Attention_Model, RNN_Attention_Model
from model.gru_model import createGRU_Model, GRU_Model 

def build_model(config, answer_space):
    if config['model']['type_model']=='rnn':
        return createRNN_Model(config,answer_space)
    if config['model']['type_model']=='lstm':
        return createLSTM_Model(config,answer_space)
    if config['model']['type_model']=='rnn_att':
        return createRNN_Attention_Model(config,answer_space)
    if config['model']['type_model']=='gru':
        return createGRU_Model(config,answer_space)
    
def get_model(config, num_labels):
    if config['model']['type_model']=='rnn':
        return RNN_Model(config,num_labels)
    if config['model']['type_model']=='lstm':
        return LSTM_Model(config,num_labels)
    if config['model']['type_model']=='rnn_att':
        return RNN_Attention_Model(config,num_labels)
    if config['model']['type_model']=='gru':
        return GRU_Model(config,num_labels)
