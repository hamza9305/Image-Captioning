
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size ,batch_size=32, num_layers=2,drop_out = 0.15):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers= self.num_layers,
                            batch_first = True, dropout=self.drop_out)
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden = self.init_hidden(self.batch_size)
        
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
    
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embed = self.embedding_layer(captions)
        features = features.view(self.batch_size, 1, -1)
#         print(f"The size of embed= {embed.size()}")
#         print(f"The size of features= {features.size()}")
        embed = torch.cat((features, embed), dim =1)
        lstm_outputs, self.hidden = self.lstm(embed, self.hidden)
        #lstm_outputs = self.dropout(lstm_outputs)
        lstm_outputs_shape = lstm_outputs.shape
        lstm_outputs_shape = list(lstm_outputs_shape)
        lstm_outputs = lstm_outputs.reshape(lstm_outputs.size()[0]*lstm_outputs.size()[1], -1)
        vocab_outputs = self.linear(lstm_outputs)
        vocab_outputs = vocab_outputs.reshape(lstm_outputs_shape[0], lstm_outputs_shape[1], -1)
        
        return vocab_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass