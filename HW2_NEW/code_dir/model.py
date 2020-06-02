import torch
from torch import nn
import torch.nn.functional as F


class DnnPosTagger(nn.Module):
    def __init__(self, word_embeddings, hidden_dim, num_layers, word_vocab_size, tag_vocab_size):
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
        # self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True,
                            batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_vocab_size)
        self.name = 'DnnPosTagger'

    def forward(self, word_idx_tensor):
        # get embedding of input
        embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1))  # [seq_length, tag_dim]
        tag_scores = F.log_softmax(tag_space, dim=1)  # [seq_length, tag_dim]
        return tag_scores