import torch
from torch import nn
import torch.nn.functional as F
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
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))
        first_mlp_out = self.hidden2first_mlp(lstm_out.view(embeds.shape[1], -1))  # [seq_length, tag_dim]
        tag_scores = F.log_softmax(first_mlp_out, dim=1)  # [seq_length, tag_dim]
        return tag_scores


class DnnSepParser(nn.Module):
    def __init__(self, word_emb_dim, tag_emb_dim, num_layers, word_vocab_size, tag_vocab_size, hidden_fc_dim,
                 word_embeddings=None, is_advanced=False):
        """
        :param word_emb_dim: dimension of word embedding
        :param tag_emb_dim: dimension of tag embedding
        :param word_vocab_size: used to create word embeddings
        :param num_layers: number of stack layers in LSTM
        :param tag_vocab_size: used to create tag embeddings
        """
        super(DnnSepParser, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.tag_emb_dim = tag_emb_dim
        self.hidden_dim_lstm = self.word_emb_dim + self.tag_emb_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if word_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False, max_norm=2)
        else:
            if is_advanced:
                self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, max_norm=2)
            else:
                self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)
        self.encoder = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=self.hidden_dim_lstm, num_layers=num_layers,
                               bidirectional=True, batch_first=False)
        self.fc1 = nn.Linear(self.hidden_dim_lstm * 4, hidden_fc_dim)
        self.tan = nn.Tanh()
        self.fc2 = nn.Linear(hidden_fc_dim, 1)
        self.name = 'DnnDepParser' + '_' + 'word_emb-' + str(self.word_emb_dim) + '_' + 'tag_emb-' + str(self.tag_emb_dim) \
                    + '_' + 'num_stack' + str(self.num_layers) + '_' + 'hidden_fc_dim-' + str(self.fc1.weight.shape[0])

    def load_embedding(self, word_embeddings, pos_embeddings):
        self.word_embedding = word_embeddings
        self.tag_embedding = pos_embeddings

    def forward(self, word_idx_tensor, tag_idx_tensor, calc_mst=False):
        # get embedding of input
        word_embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, word_emb_dim]
        tag_embeds = self.tag_embedding(tag_idx_tensor.to(self.device))  # [batch_size, seq_length, tag_emb_dim]
        concat_emb = torch.cat([word_embeds, tag_embeds], dim=2)  # [batch_size, seq_length, word_emb_dim+tag_emb_dim]
        lstm_out, _ = self.encoder(concat_emb.view(concat_emb.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]

        lstm_out_b_first = lstm_out.permute(1, 0, 2)
        first_part_out = (lstm_out_b_first @ self.fc1.weight.T[:lstm_out_b_first.shape[2], :] + self.fc1.bias.T).squeeze(0)
        second_part_out = (lstm_out_b_first @ self.fc1.weight.T[lstm_out_b_first.shape[2]:, :] + self.fc1.bias.T).squeeze(0)
        first_part_out1 = first_part_out.unsqueeze(0)
        second_part_out1 = second_part_out.unsqueeze(1)
        first_part_out2 = first_part_out1.repeat(second_part_out.shape[0], 1, 1)
        second_part_out2 = second_part_out1.repeat(1, first_part_out.shape[0], 1)
        Z = first_part_out2 + second_part_out2
        out_1 = Z.view(-1, Z.shape[-1])  # [seq_length**2,hidden_dim_mlp]

        # scores = self.fc2(self.tan(out_1)).view(lstm_out.shape[0], lstm_out.shape[0]).squeeze(0)
        scores = self.fc2(self.tan(out_1)).view(lstm_out.shape[0], lstm_out.shape[0]).squeeze(0)
        tmp_scores = F.log_softmax(scores, dim=1)
        # calc tree
        our_heads = None
        if calc_mst:
            with torch.no_grad():
                dep_scores = scores.unsqueeze(0).permute(0, 2, 1)
                dep_scores_2d = dep_scores.squeeze(0)
                # TODO: add zeros on diagonal
                our_heads, _ = decode_mst(energy=dep_scores_2d.cpu().numpy(), length=tmp_scores.shape[0],
                                          has_labels=False)
                # print(f'our heads: {our_heads}')
        # print(f'tmp_scores.device: {tmp_scores.device}')
        # print(f'our_heads type: {type(our_heads)}')
        # print(f'scores.device: {scores.device}')
        return tmp_scores, our_heads, scores
