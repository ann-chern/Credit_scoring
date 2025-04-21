import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditsRNN(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32):
        super(CreditsRNN, self).__init__()
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                          for feature in features])

        self._gru = nn.GRU(input_size=sum([embedding_projections[x][1] for x in features]),
                             hidden_size=rnn_units, batch_first=True, dropout=0.0, bidirectional=False)
        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(in_features=rnn_units, out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)

    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)

        _, last_hidden = self._gru(concated_embeddings)
        last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, self._hidden_size))

        classification_hidden = self._top_classifier(last_hidden)
        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
    
    
# NEW MODEL
class CreditsRNN_Bi(nn.Module):
    def __init__(self, features, embedding_projections, hidden_size=128, top_classifier_units=32, n_layers=1, dropout_p=0.2, bidirectional=False, method='GRU'):
        super(CreditsRNN_Bi, self).__init__()

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even when bidirectional is True")

        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                          for feature in features])
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._bidirectional = bidirectional
        self._n_directions = 2 if bidirectional else 1
        self._dropout_p = dropout_p
        self._method = method

        classifier_input_size = hidden_size * self._n_directions
        rnn_dropout = dropout_p if n_layers > 1 else 0.0

        if self._method == 'GRU':
          self._net = nn.GRU(input_size=sum([embedding_projections[x][1] for x in features]),
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=bidirectional)
        elif self._method == 'LSTM':
          self._net = nn.LSTM(input_size=sum([embedding_projections[x][1] for x in features]),
                             hidden_size=hidden_size,
                             num_layers=n_layers,
                             batch_first=True,
                             dropout=rnn_dropout,
                             bidirectional=bidirectional)
        else:
            raise ValueError("Method must be 'GRU' or 'LSTM'")

        self._top_classifier = nn.Linear(in_features=classifier_input_size, out_features=top_classifier_units)
        self._activation_function = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
        self._dropout = nn.Dropout(self._dropout_p)

    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = self._dropout(concated_embeddings)

        _, last_hidden = self._net(concated_embeddings)
        last_hidden = last_hidden[0] if self._method == 'LSTM' else last_hidden

        # Last layer (num_layers, num_directions, batch, hidden_size)
        last_hidden = last_hidden.view(self._n_layers, self._n_directions, batch_size, self._hidden_size)
        last_layer_hidden = last_hidden[-1]  # (num_directions, batch, hidden_size)

        combined_hidden = last_layer_hidden.permute(1, 0, 2).reshape(batch_size, -1)
        combined_hidden = self._dropout(combined_hidden)

        classification_hidden = self._top_classifier(combined_hidden)
        activation = self._activation_function(classification_hidden)
        raw_output = self._head(activation)
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
    

# NEW MODEL
class CreditsRNN_Bi_pool_drop(nn.Module):
    def __init__(self, features, embedding_projections, hidden_size=128, top_classifier_units=32, n_layers=1, dropout_p=0.2, bidirectional=False, method='GRU'):
        super(CreditsRNN_Bi_pool_drop, self).__init__()

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even when bidirectional is True")

        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                          for feature in features])

        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._bidirectional = bidirectional
        self._n_directions = 2 if bidirectional else 1
        self._dropout_p = dropout_p
        self._method = method

        rnn_dropout = dropout_p if n_layers > 1 else 0.0

        if self._method == 'GRU':
          self._net = nn.GRU(input_size=sum([embedding_projections[x][1] for x in features]),
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=bidirectional)
        elif self._method == 'LSTM':
          self._net = nn.LSTM(input_size=sum([embedding_projections[x][1] for x in features]),
                             hidden_size=hidden_size,
                             num_layers=n_layers,
                             batch_first=True,
                             dropout=rnn_dropout,
                             bidirectional=bidirectional)
        else:
            raise ValueError("Method must be 'GRU' or 'LSTM'")

        # Calculating input size for classifier
        pool_features = hidden_size * self._n_directions * 2  # max + avg pooling
        classifier_input_size = hidden_size * self._n_directions + pool_features

        self._top_classifier = nn.Linear(in_features=classifier_input_size, out_features=top_classifier_units)
        self._activation_function = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
        self._dropout = nn.Dropout(self._dropout_p)


    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = self._dropout(concated_embeddings)

        output, last_hidden = self._net(concated_embeddings)
        last_hidden = last_hidden[0] if self._method == 'LSTM' else last_hidden

        # Extracting the last layer [n_layers, n_directions, batch, hidden_size]
        last_hidden = last_hidden.view(self._n_layers, self._n_directions, batch_size, self._hidden_size)
        last_layer_hidden = last_hidden[-1]  # [n_directions, batch, hidden_size]

        # Combining directions
        rnn_features = last_layer_hidden.permute(1, 0, 2).reshape(batch_size, -1)  # [batch, n_directions*hidden_size]

        # Pulling along the time axis of outputs RNN (output: [batch, seq_len, n_directions*hidden_size])
        output_pool = output.permute(0, 2, 1)  # [batch, features, seq_len]
        max_pool = F.adaptive_max_pool1d(output_pool, 1).view(batch_size, -1)
        avg_pool = F.adaptive_avg_pool1d(output_pool, 1).view(batch_size, -1)
        pooled_features = torch.cat([max_pool, avg_pool], dim=1)

        # Concatenation of all features
        combined_features = torch.cat([rnn_features, pooled_features], dim=1)
        combined_features = self._dropout(combined_features)

        # Classifier
        classification = self._top_classifier(combined_features)
        activation = self._activation_function(classification)
        raw_output = self._head(activation)
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
    
    
# NEW MODEL
class RNN_pooling(nn.Module):
    def __init__(self, features, embedding_projections, hidden_size=128, top_classifier_units=128, dense_units=32, n_layers=1, dropout_p=0.2, bidirectional=False, method='GRU'):
        super(RNN_pooling, self).__init__()

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even when bidirectional is True")

        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                          for feature in features])
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._bidirectional = bidirectional
        self._n_directions = 2 if bidirectional else 1
        self._dropout_p = dropout_p
        self._method = method

        rnn_input_size = sum(embedding_projections[x][1] for x in features)
        rnn_dropout = dropout_p if n_layers > 1 else 0.0

        if self._method == 'GRU':
          self._net = nn.GRU(input_size=rnn_input_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=bidirectional)
        elif self._method == 'LSTM':
          self._net = nn.LSTM(input_size=rnn_input_size,
                             hidden_size=hidden_size,
                             num_layers=n_layers,
                             batch_first=True,
                             dropout=rnn_dropout,
                             bidirectional=bidirectional)
        else:
            raise ValueError("Method must be 'GRU' or 'LSTM'")

        # Calculation of the feature dimension
        self._emb_pool_size = 2 * rnn_input_size  # max + avg poolingо embed
        self._rnn_hidden_pool_size = 2 * hidden_size * self._n_directions  # max + avg pooling hidden
        self._rnn_last_hidden_size = hidden_size * self._n_directions

        # Total size of the features
        total_features = (self._rnn_last_hidden_size +
                        self._rnn_hidden_pool_size +
                        self._emb_pool_size)

        # Classifier
        self._top_classifier = nn.Linear(in_features=total_features, out_features=top_classifier_units)
        self._activation_function = nn.ReLU()
        self._dense = nn.Linear(in_features=top_classifier_units, out_features=dense_units)
        self._head = nn.Linear(in_features=dense_units, out_features=1)
        self._dropout = nn.Dropout(self._dropout_p)


    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1) # [batch, seq_len, emb_size]
        concated_embeddings = self._dropout(concated_embeddings)

        output, last_hidden = self._net(concated_embeddings)
        last_hidden = last_hidden[0] if self._method == 'LSTM' else last_hidden

        # Extracting the last layer [n_layers, n_directions, batch, hidden_size]
        last_hidden = last_hidden.view(self._n_layers, self._n_directions, batch_size, self._hidden_size)
        last_layer_hidden = last_hidden[-1]  # [n_directions, batch, hidden_size]

        # Combining directions for the last hidden state
        rnn_last_hidden = last_layer_hidden.permute(1, 0, 2).reshape(batch_size, -1)  # [batch, n_directions*hidden_size]

        # Time axis pooling for RNN outputs
        emb_pool = concated_embeddings.permute(0, 2, 1)  # [batch, emb_size, seq_len]
        max_emb_pool = F.adaptive_max_pool1d(emb_pool, 1).view(batch_size, -1)
        avg_emb_pool = F.adaptive_avg_pool1d(emb_pool, 1).view(batch_size, -1)
        emb_pool_features = torch.cat([max_emb_pool, avg_emb_pool], dim=1)

        # Пулинг по временной оси для выходов RNN
        rnn_output = output.permute(0, 2, 1)  # [batch, features, seq_len]
        max_rnn_pool = F.adaptive_max_pool1d(rnn_output, 1).view(batch_size, -1)
        avg_rnn_pool = F.adaptive_avg_pool1d(rnn_output, 1).view(batch_size, -1)
        rnn_pool_features = torch.cat([max_rnn_pool, avg_rnn_pool], dim=1)

        # Concatenation of all features
        combined_features = torch.cat([
            rnn_last_hidden,
            rnn_pool_features,
            emb_pool_features
        ], dim=1)
        combined_features = self._dropout(combined_features)

        # Classifier
        classification = self._top_classifier(combined_features)
        activation = self._activation_function(classification)
        dense_out = self._dense(activation)
        raw_output = self._head(dense_out)
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
    
    
# NEW MODEL
class BidirectRNN(nn.Module):
    def __init__(self, features, embedding_projections, hidden_size=128, top_classifier_units=32, n_layers=1, dropout_p=0.2, bidirectional=False, method='GRU'):
        super(BidirectRNN, self).__init__()

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even when bidirectional is True")

        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                          for feature in features])
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._bidirectional = bidirectional
        self._n_directions = 2 if bidirectional else 1
        self._dropout_p = dropout_p
        self._method = method

        rnn_dropout = dropout_p if n_layers > 1 else 0.0
        input_size = sum(embedding_projections[x][1] for x in features)

        if self._method == 'GRU':
          self._net = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=bidirectional)
        elif self._method == 'LSTM':
          self._net = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=n_layers,
                             batch_first=True,
                             dropout=rnn_dropout,
                             bidirectional=bidirectional)
        else:
            raise ValueError("Method must be 'GRU' or 'LSTM'")

        self._top_classifier = nn.Linear(in_features=hidden_size * self._n_directions,
                                         out_features=top_classifier_units)
        self._activation_function = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
        self._dropout = nn.Dropout(self._dropout_p)


    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        concated_embeddings = self._dropout(concated_embeddings)

        _, last_hidden = self._net(concated_embeddings)
        last_hidden = last_hidden[0] if self._method == 'LSTM' else last_hidden

        # Extracting the last layer [n_layers, n_directions, batch, hidden_size]
        last_hidden = last_hidden.view(self._n_layers, self._n_directions, batch_size, self._hidden_size)
        last_layer_hidden = last_hidden[-1]  # [n_directions, batch, hidden_size]

        # Combining directions
        if self._bidirectional:
            final_hidden = torch.cat([
                last_layer_hidden[0],  # forward
                last_layer_hidden[1]   # backward
            ], dim=-1)
        else:
            final_hidden = last_layer_hidden.squeeze(0)  # [batch, hidden_size]

        final_hidden = self._dropout(final_hidden)

        # Classifier
        classification = self._top_classifier(final_hidden)
        activation = self._activation_function(classification)
        raw_output = self._head(activation)
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)
    