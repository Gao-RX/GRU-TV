import torch
import torch.nn as nn


class GRUAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, input_size, hidden_size, feature_size, is_rep=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.out_layer = nn.Linear(hidden_size, input_size)
        self.encoder_linear = nn.Linear(hidden_size, feature_size)
        self.relu = nn.ReLU()
        self.decoder_linear = nn.Linear(feature_size, hidden_size)
        self.input_size = input_size
        self.is_rep = is_rep

    def forward(self, sequence: torch.Tensor, 
                mask: torch.Tensor, 
                delta: torch.tensor, 
                dt: torch.Tensor):
        h = nn.parameter.Parameter(torch.randn(1, self.hidden_size)).to(sequence.device)
        dh = nn.parameter.Parameter(torch.randn(self.hidden_size)).to(sequence.device)
        prex = nn.parameter.Parameter(torch.zeros(self.input_size)).to(sequence.device)
        mean_value = torch.squeeze(torch.sum(sequence,1))/(1e-6+torch.squeeze(torch.sum((mask!=0),1)))
        mean_value = mean_value.to(sequence.device)
        self.encoder.mean_value = mean_value
        self.decoder.mean_value = mean_value
        for layer in range(sequence.shape[1]):
            t_els = dt[:, layer]
            if t_els <= 0:
                break
            x = sequence[:,layer,...]
            m = mask[:,layer,...]
            d = delta[:,layer,...]
            h_post, dh, prex = self.encoder(h, x, m, d, prex, dh)
            h = h + t_els * dh

        h_post = self.relu(h_post)
        feature = self.encoder_linear(h_post)
        feature = self.relu(feature)
        if self.is_rep:
            return feature
        h = self.decoder_linear(feature)
        output_list = []
        for layer_dec in reversed(list(range(layer))):
            x = sequence[:,layer_dec,...]
            m = mask[:,layer_dec,...]
            d = delta[:,layer_dec,...]
            t_els = dt[:, layer_dec]
            h_post, dh, prex = self.encoder(h, x, m, d, prex, dh)
            h = h + t_els * dh
            out = self.out_layer(h_post)
            output_list.append(out)
        
        out_sequence = torch.cat(list(reversed(output_list)))
        return out_sequence, layer

if __name__ == '__main___':
    pass