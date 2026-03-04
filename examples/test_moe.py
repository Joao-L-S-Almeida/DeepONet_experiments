import numpy as np

from simulai.models import MoEPool
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork
from simulai.templates import NetworkTemplate

class EncoderDecoder(NetworkTemplate):

    def __init__(self, encoder:NetworkTemplate=None, decoder:NetworkTemplate=None):

        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    @property
    def weights(self) -> list:
        return sum([net.weights for net in [self.encoder, self.decoder]], [])

    def forward(self, input_data):

        latent = encoder(input_data)
        output = decoder(latent)

        return output

lr = 1e-3
optimizer_config = {"lr": lr}

optimizer = Optimizer(
    "adam",
    params=optimizer_config,
)

n_inputs_b = 20
n_latent = 100
n_outputs = 10

config_encoder = {
    "layers_units": 7 * [100],  # Hidden layers
    "activations": "tanh",
    "input_size": n_inputs_b,
    "output_size": n_latent,
    "name": "encoder_net",
}

config_decoder = {
    "layers_units": 7 * [100],  # Hidden layers
    "activations": "tanh",
    "input_size": n_latent,
    "output_size": n_outputs,
    "name": "decoder_net",
}


experts_list_encoder = list()
experts_list_decoder = list()

n_experts_encoder = 8
n_experts_decoder = 8
n_epochs = 1_00

for ex in range(n_experts_encoder):
    experts_list_encoder.append(DenseNetwork(**config_encoder))

for ex in range(n_experts_decoder):
    experts_list_decoder.append(DenseNetwork(**config_decoder))

encoder = MoEPool(experts_list=experts_list_encoder, binary_selection=True,
                  input_size=n_inputs_b, devices="cpu")
decoder = MoEPool(experts_list=experts_list_decoder, binary_selection=True,
                  input_size=n_latent, devices="cpu")

encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder)

input_data = np.random.rand(1_000, n_inputs_b)
target_data = np.random.rand(1_000, n_outputs)

params = {"lambda_1": 0.0, "lambda_2": 1e-6}

optimizer.fit(
    op=encoder_decoder,
    input_data=input_data,
    target_data=target_data,
    params=params,
    n_epochs=n_epochs,
    loss="rmse",
    device="cpu",
)
print(encoder_decoder)
print(encoder_decoder(input_data).shape)
