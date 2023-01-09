import random 
import json
import torch
from model import NeuralNet
from nlp_utils import bag_of_words, tokenize, stem

# Setting up Bot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('traindata.json','r') as file:
    intents = json.load(file)

FILE = "data1.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def Chatbot(statement):
    statement = tokenize(statement)
    
    X = bag_of_words(statement,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _,pred = torch.max(output,dim=1)
    tag = tags[pred.item()]
    
    probs = torch.softmax(output,dim=1)
    prob = probs[0][pred.item()]
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                try:
                    return random.choice(intent['responses'])
                except:
                    return "Acknowledged"
            
    else:
        return "I dont understand u"