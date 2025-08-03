import os, glob
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ca_utils
from scipy.io import wavfile
from collections import defaultdict

class VoiceDataset(Dataset):
    #voices

    def __init__(self, root, embeddings):
        super(VoiceDataset, self).__init__()
        self.voices = sorted(glob.glob(os.path.join(root, '*.mp3')))
        self.embeddings = embeddings

    def __getitem__(self, index):
        voice_file_name = self.voices[index]

        with open(voice_file_name[:-4] + '.txt') as file:
            relevant_words = file.read().replace('n', '')

        one_hot_word = [0] * len(self.embeddings)
        for word in relevant_words:
            one_hot_word[self.embeddings.index(word)] = 1

        _, audio_data = wavfile.read(voice_file_name)
        audio_data = audio_data / 32768.0 #normalize

        return audio_data, one_hot_word
    
    def __len__(self):
        return len(self.voices)
    
def train_cnn(model, trainData):
    #params
    epochs = 10
    learningRate = 0.025
    opt = optim.Adam(model.parameters(), lr=learningRate)
    lossFunction = nn.CrossEntropyLoss()

    #training
    gpu = torch.device('cuda:0') #use GPU to train

    model.train() #set model to training mode
    for epoch in range(epochs):
        print("Epoch", epoch + 1, "/", epochs)
        
        for i, data in enumerate(trainData):
            #load the data to train with
            voice_data, one_hot_word = data

            #test
            output = model(voice_data) #send the image through the model
            loss = lossFunction(output, one_hot_word) #calculate the loss compared to the input

            #train
            loss.backward() #backprop
            opt.step() #update the model weights
            opt.zero_grad() #gradients accumulate, reset them
       
    print('Finished training')
    torch.save(model.state_dict(), 'voices.pth')
    return

def test_cnn(model, testloader):
    print('Testing model...')

    preds = np.array([])
    truth = np.array([])

    gpu = torch.device('cuda:0') #use GPU to train
    model.eval() #set model to evaluation mode
    with torch.no_grad(): #performance optimization
        for i, data in enumerate(testloader):
            voice_data, one_hot_word = data

            #test
            output = model(voice_data) #send the image through the model
            prediction = output.data.cpu().numpy()

            #normalize
            prediction[prediction < 0] = 0
            result = (model.Softmax(dim=1))(prediction)
            
            #update stats
            preds = np.append(preds, result)
            truth = np.append(truth, one_hot_word)

    ab = truth != 0

    ape = np.abs((truth - preds) / truth)
    ape = ape[ab]
    mape = np.mean(ape) * 100

    return preds.astype(np.int64), truth.astype(np.int64), mape

def GetData(trainData, testData):
    #data loader
    train_loader = DataLoader(trainData, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(testData, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader, test_loader

def GenerateEmbeddings(words_root, embedding_path):
    word_dict = defaultdict()

    text_files = sorted(glob.glob(os.path.join(words_root, '*.txt')))
    for text_file in text_files:
        with open(text_file) as opened_text:
            relevant_words = opened_text.read().replace('n', '')
            word_list = relevant_words.split()
            for word in word_list:
                word_dict[word] = 1

    os.remove(embedding_path)
    with open(embedding_path, "x") as opened_embeddings:
        for word in word_dict:
            opened_embeddings.write(word)

    return word_dict.items()

def LoadEmbeddings(embedding_path):
    word_dict = defaultdict()

    with open(embedding_path, "r") as file:
        for line in file:
            word_dict[line] = 1

    return word_dict.items()

if __name__ == '__main__':
    embeddings = GenerateEmbeddings('VoiceData\\WordData\\transcription', 'VoiceData\\WordData\\embeddings.txt')
    #embeddings = LoadEmbeddings('VoiceData\\WordData\\embeddings.txt')

    trainData = VoiceDataset('VoiceData\\train', embeddings)
    testData = VoiceDataset('voiceData\\val', embeddings)

    trainLoader, testLoader = GetData(trainData, testData)

    model = ca_utils.ResNet(block=ca_utils.BasicBlock, layers=[1, 1, 1], num_classes=len(embeddings), in_channels=1).to(torch.device('cuda:0')) 
    train_cnn(model, trainLoader)
    
    #model.load_state_dict(torch.load('voices.pth'))

    preds, truth, mape = test_cnn(model, testLoader)
    print("SCORE:" + mape)




