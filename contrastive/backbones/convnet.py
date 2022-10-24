from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(pl.LightningModule):
    r"""3D-ConvNet model class, based on

    Attributes:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate 
        num_classes (int) - number of classification classes
            (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels=1, encoder_depth=3,
                 num_representation_features=256,
                 num_outputs=64, 
                 projection_head_hidden_layers=None,
                 drop_rate=0.1,
                 projection_head_type="linear",
                 batchnorm=False,
                 mode="encoder",
                 memory_efficient=False,
                 in_shape=None):

        super(ConvNet, self).__init__()

        assert mode in {'encoder', 'evaluation', 'decoder'},\
            "Unknown mode selected: %s" % mode


        self.mode = mode
        self.num_representation_features = num_representation_features
        self.num_outputs = num_outputs
        if projection_head_hidden_layers:
            self.projection_head_hidden_layers = projection_head_hidden_layers
        else:
            self.projection_head_hidden_layers = [num_outputs]
        self.drop_rate = drop_rate

        # Decoder part
        self.in_shape = in_shape
        c, h, w, d = in_shape
        self.encoder_depth = encoder_depth
        self.z_dim_h = h//2**self.encoder_depth # receptive field downsampled 2 times
        self.z_dim_w = w//2**self.encoder_depth
        self.z_dim_d = d//2**self.encoder_depth


        modules_encoder = []
        for step in range(encoder_depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0  else 16 * (2**step)
            modules_encoder.append(('conv%s' %step, nn.Conv3d(in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)))
            modules_encoder.append(('norm%s' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' %step, nn.LeakyReLU()))
            modules_encoder.append(('conv%sa' %step, nn.Conv3d(out_channels, out_channels,
                    kernel_size=4, stride=2, padding=1)))
            modules_encoder.append(('norm%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' %step, nn.LeakyReLU()))
            modules_encoder.append(('DropOut%sa' %step, nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels
        # flatten and reduce to the desired dimension

        self.encoder = nn.Sequential(OrderedDict(modules_encoder))


        if (self.mode == "encoder") or (self.mode == 'evaluation'):

            self.features2 = nn.Linear(self.num_features*self.z_dim_h*self.z_dim_w*self.z_dim_d,
                              self.num_representation_features)

            if batchnorm:
                self.hidden_representation = nn.Sequential(OrderedDict([
                    ('linrepr', nn.Linear(self.num_representation_features, self.num_representation_features)),
                    ('normrepr', nn.BatchNorm1d(self.num_representation_features, track_running_stats=False)),
                ]))
            else:
                self.hidden_representation = nn.Sequential(OrderedDict([
                    ('linrepr', nn.Linear(self.num_representation_features, self.num_representation_features)),
                ]))       

            self.backward_linear = nn.Linear(
                self.num_representation_features, self.num_representation_features)

            # build a projection head
            if projection_head_type == "non-linear":
                projection_head = []
                input_size = self.num_representation_features
                for i, dim_i in enumerate(self.projection_head_hidden_layers):
                    output_size = dim_i
                    projection_head.append(('Norm%s' %i, nn.BatchNorm1d(output_size)))
                    projection_head.append(('ReLU%s' %i, nn.ReLU()))
                    input_size = output_size
                projection_head.append(('Output layer' ,nn.Linear(input_size,
                                                                self.num_outputs)))
                projection_head.append(('Norm layer', nn.BatchNorm1d(self.num_outputs)))
                self.projection_head = nn.Sequential(OrderedDict(projection_head))
            elif projection_head_type == "linear":
                self.projection_head = nn.Sequential(
                                        # nn.Linear(self.num_representation_features,
                                        #           self.num_outputs),
                                        nn.ReLU(),
                                        nn.Linear(self.num_outputs,
                                                  self.num_outputs))
            else:
                raise ValueError("projection_head_type must be either \"linear\" or \"non-linear\. "
                                 f"You have set it to: {projection_head_type}")

        elif self.mode == "decoder":
            self.hidden_representation = nn.Linear(
                self.num_features, self.num_representation_features)
            self.develop = nn.Linear(self.num_representation_features,
                                     64 *self.z_dim_h * self.z_dim_w* self.z_dim_d)
            modules_decoder = []
            out_channels = 64
            for step in range(self.depth-1):
                in_channels = out_channels
                out_channels = in_channels // 2
                ini = 1 if step==0 else 0
                modules_decoder.append(('convTrans3d%s' %step, nn.ConvTranspose3d(in_channels,
                            out_channels, kernel_size=2, stride=2, padding=0, output_padding=(ini,0,0))))
                modules_decoder.append(('normup%s' %step, nn.BatchNorm3d(out_channels)))
                modules_decoder.append(('ReLU%s' %step, nn.ReLU()))
                modules_decoder.append(('convTrans3d%sa' %step, nn.ConvTranspose3d(out_channels,
                            out_channels, kernel_size=3, stride=1, padding=1)))
                modules_decoder.append(('normup%sa' %step, nn.BatchNorm3d(out_channels)))
                modules_decoder.append(('ReLU%sa' %step, nn.ReLU()))
            modules_decoder.append(('convtrans3dn', nn.ConvTranspose3d(16, 1, kernel_size=2,
                            stride=2, padding=0)))
            modules_decoder.append(('conv_final', nn.Conv3d(1, 2, kernel_size=1, stride=1)))
            self.decoder = nn.Sequential(OrderedDict(modules_decoder))

        
        if self.mode == "decoder":

            # This loads pretrained weight
            path = "/host/volatile/jc225751/Runs/33_MIDL_2022_reviews/Output/t-0.1/n-004_o-4/logs/default/version_0/checkpoints/epoch=299-step=8399.ckpt"
            pretrained = torch.load(path)
            model_dict = self.state_dict()
            for n, p in pretrained['state_dict'].items():
                if n in model_dict:
                    model_dict[n] = p
            self.load_state_dict(model_dict)

            # This freezes all layers except projection head layers
            layer_counter = 0
            for (name, module) in self.named_children():
                print(f"Module name = {name}")

            for (name, module) in self.named_children():
                if name == 'features':
                    for layer in module.children():
                        for param in layer.parameters():
                            param.requires_grad = False
                        
                        print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, name))
                        layer_counter+=1
            for param in self.hidden_representation.parameters():
                param.requires_grad = False
            print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, "representation"))
            for (name, param) in self.named_parameters():
                print(f"{name}: learning = {param.requires_grad}")

    def forward(self, x):
        # Eventually keep the input images for visualization
        # self.input_imgs = x.detach().cpu().numpy()
        out = self.encoder(x)

        if (self.mode == "encoder") or (self.mode == 'evaluation'):
            out = F.relu(out, inplace=True)
            out = torch.flatten(out, 1)
            out = self.features2(out)
            out_backbone = out

            out = self.hidden_representation(out)
            out_representation = out
            x = self.backward_linear(out)
            out = F.relu(out)

            out = self.projection_head(out)

            out = torch.cat((out, out_representation, out_backbone, x), dim=1)

        elif self.mode == "decoder":
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)

            out = self.hidden_representation(out)    
            out = F.relu(out, inplace=True)
            out = self.develop(out)
            out = out.view(out.size(0), 16 * 2**(self.depth-1), self.z_dim_h, self.z_dim_w, self.z_dim_d)
            out = self.decoder(out)
        return out.squeeze(dim=1)

    def get_current_visuals(self):
        return self.input_imgs
