import torch
from torch import nn
import torch.nn.functional as F


class DnnPosTagger(nn.Module):
    def __init__(self, word_embeddings, hidden_dim, num_layers, tag_vocab_size):
        """
        :param word_embeddings: word vectors from dataset, shape: (vocab_size, emb_dim)
        :param hidden_dim: number of hidden dims in LSTM
        :param word_vocab_size: used to create word embeddings (not used here)
        :param tag_vocab_size: size of tag vocab, needed to determine the output size
        """
        super(DnnPosTagger, self).__init__()
        emb_dim = word_embeddings.shape[1]
        print(f'word embeddings shape: {word_embeddings.shape}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True,
                            batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_vocab_size)
        self.name = 'DnnPosTagger'

    def forward(self, word_idx_tensor):
        # get embedding of input
        embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, emb_dim]
        # print(f'embeds shape: {embeds.shape}')
        lstm_out, _ = self.lstm(embeds)
        # print(f'lstm_out shape: {lstm_out.shape}')
        # print(f'lstm_out.view(embeds.shape[1], -1) shape: {lstm_out.view(embeds.shape[1], -1).shape}')
        tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1))  # [seq_length, tag_dim]
        # print(f'tag_space shape: {tag_space.shape}')
        tag_scores = F.log_softmax(tag_space, dim=1)  # [seq_length, tag_dim]
        return tag_scores


class DnnSepParser(nn.Module):
    def __init__(self, word_emb_dim, tag_emb_dim, hidden_dim, num_layers, word_vocab_size, tag_vocab_size, max_sentence_len):
        """
        :param word_emb_dim: dimension of word embedding
        :param tag_emb_dim: dimension of tag embedding
        :param word_vocab_size: used to create word embeddings
        :param hidden_dim: number of hidden layers in LSTM
        :param num_layers: number of stack layers in LSTM
        :param tag_vocab_size: used to create tag embeddings
        :param max_sentence_len: used to determine the output size of MLP
        """
        super(DnnSepParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)
        self.lstm = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=False)
        self.hidden2dep = nn.Linear(hidden_dim * 2, max_sentence_len)
        self.name = 'DnnPosTagger'

    def forward(self, word_idx_tensor, tag_idx_tensor):
        # get embedding of input
        word_embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, word_emb_dim]
        tag_embeds = self.tag_embedding(tag_idx_tensor.to(self.device))  # [batch_size, seq_length, tag_emb_dim]
        concat_emb = torch.cat([word_embeds, tag_embeds], dim=2)  # [batch_size, seq_length, word_emb_dim+tag_emb_dim]
        lstm_out, _ = self.lstm(concat_emb)  # [seq_length, batch_size, 2*hidden_dim]
        tag_space = self.hidden2tag(lstm_out.view(concat_emb.shape[1], -1))  # [seq_length, tag_dim]
        tag_scores = F.log_softmax(tag_space, dim=1)  # [seq_length, tag_dim]
        return tag_scores