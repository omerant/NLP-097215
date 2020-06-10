import torch
from torch import nn
import torch.nn.functional as F
from itertools import product
from chu_liu_edmonds import decode_mst


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
                            batch_first=False)
        self.hidden2first_mlp = nn.Linear(hidden_dim * 2, tag_vocab_size)
        # self.first_mlp2second_mlp = nn.Linear(tag_vocab_size, tag_vocab_size)
        # self.tanh = nn.Tanh()
        self.name = 'DnnPosTagger'

    def forward(self, word_idx_tensor):
        # get embedding of input
        embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, emb_dim]
        # print(f'embeds shape: {embeds.shape}')
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))
        # print(f'lstm_out shape: {lstm_out.shape}')
        # print(f'lstm_out.view(embeds.shape[1], -1) shape: {lstm_out.view(embeds.shape[1], -1).shape}')
        first_mlp_out = self.hidden2first_mlp(lstm_out.view(embeds.shape[1], -1))  # [seq_length, tag_dim]
        tag_scores = F.log_softmax(first_mlp_out, dim=1)  # [seq_length, tag_dim]
        return tag_scores


class DnnSepParser(nn.Module):
    def __init__(self, word_emb_dim, tag_emb_dim, num_layers, word_vocab_size, tag_vocab_size, max_sentence_len):
        """
        :param word_emb_dim: dimension of word embedding
        :param tag_emb_dim: dimension of tag embedding
        :param word_vocab_size: used to create word embeddings
        :param num_layers: number of stack layers in LSTM
        :param tag_vocab_size: used to create tag embeddings
        :param max_sentence_len: used to determine the output size of MLP
        """
        super(DnnSepParser, self).__init__()
        # self.word_count = train_word_count_dict
        self.word_emb_dim = word_emb_dim
        self.tag_emb_dim = tag_emb_dim
        self.hidden_dim = self.word_emb_dim + self.tag_emb_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)
        self.encoder = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=self.hidden_dim, num_layers=num_layers,
                               bidirectional=True, batch_first=False)
        # self.hidden2first_mlp = nn.Linear(self.hidden_dim * 2, max_sentence_len)
        # self.tanh = torch.nn.Tanh()
        # self.first_mlp2second_mlp = nn.Linear(max_sentence_len, max_sentence_len)
        self.tmp1 = nn.Linear(self.hidden_dim * 4, 100)
        self.tmp_tan = nn.Tanh()
        self.tmp2 = nn.Linear(100, 1)
        self.name = 'DnnDepParser' + '_' + 'word_emb-' + str(self.word_emb_dim) + '_' + 'tag_emb-' + str(self.tag_emb_dim) \
                    + '_' + 'num_stack' + str(self.num_layers)

    def forward(self, word_idx_tensor, tag_idx_tensor, calc_mst=False):
        # get embedding of input
        word_embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, word_emb_dim]
        tag_embeds = self.tag_embedding(tag_idx_tensor.to(self.device))  # [batch_size, seq_length, tag_emb_dim]
        concat_emb = torch.cat([word_embeds, tag_embeds], dim=2)  # [batch_size, seq_length, word_emb_dim+tag_emb_dim]
        lstm_out, _ = self.encoder(concat_emb.view(concat_emb.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        # first_mlp_out = self.hidden2first_mlp(lstm_out.view(concat_emb.shape[1], -1))  # [seq_length, tag_dim]
        # second_mlp_out = self.first_mlp2second_mlp(self.tanh(first_mlp_out))
        # dep_scores = F.log_softmax(second_mlp_out, dim=1)  # [seq_length, tag_dim]
        sq = lstm_out.squeeze(dim=1) #[seq_length,2*hidden_dim]
        pairs = [pair for pair in product(sq,sq)] #[seq_length**2,4*hidden_dim
        pairs_cat = [torch.cat(pair).unsqueeze(0) for pair in pairs]
        all_pairs = torch.cat(pairs_cat)
        scores = self.tmp2(self.tmp_tan(self.tmp1(all_pairs))).view(lstm_out.shape[0], lstm_out.shape[0]) #[seq_length,seq_length]
        tmp_scores = F.log_softmax(scores, dim=1)

        # calc tree
        our_heads = None
        if calc_mst:
            with torch.no_grad():
                dep_scores = scores.unsqueeze(0).permute(0, 2, 1)
                dep_scores_2d = dep_scores.squeeze(0)
                our_heads, _ = decode_mst(energy=dep_scores_2d.cpu().numpy(), length=tmp_scores.shape[0],
                                          has_labels=False)
                # print(f'our heads: {our_heads}')
        return tmp_scores, our_heads