import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict

class ProfileConv(nn.Module):

    @staticmethod
    def shape2str(shape):
        out = ""
        for i in shape:
            out += f"{i}x"
        return out[:-1]

    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        # self.macs = []
        # self.params = []
        self.profile = []
        self.header = ['type',
                       'input_shape',
                       'input_size',
                       'output_shape',
                       'output_size',
                       'MACs',
                       'weight_shape',
                       'Params'
                      ]


        def hook_conv(module, input, output):
            name = module.__class__.__name__
            weight_shape = self.shape2str(module.weight.size())
            input_shape = self.shape2str(input[0].size())
            input_size = np.prod(input[0].size())
            output_shape = self.shape2str(output.size())
            output_size = np.prod(output.size())
            # self.macs.append(
            #     (name,
            #      input_size,
            #      int(output.size(1) * output.size(2) * output.size(3) *
            #                  module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups),
            #      output_size
            #     )
            # )
            # self.params.append(
            #     (name,
            #      weight_size,
            #      int(module.weight.size(0) * module.weight.size(1) *
            #                    module.weight.size(2) * module.weight.size(3))
            #     )
            # )
            
            self.profile.append(
                (name,
                 input_shape,
                 input_size,
                 output_shape,
                 output_size,
                 int(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups),
                 weight_shape,
                 int(module.weight.size(0) * module.weight.size(1) *
                               module.weight.size(2) * module.weight.size(3))
                )
            )

        def hook_linear(module, input, output):
            name = module.__class__.__name__
            weight_shape = self.shape2str(module.weight.size())
            input_shape = self.shape2str(input[0].size())
            input_size = np.prod(input[0].size())
            output_shape = self.shape2str(output.size())
            output_size = np.prod(output.size())
            # self.macs.append(
            #     (name,
            #      input_size,
            #      int(module.weight.size(0) * module.weight.size(1)),
            #      output_size
            #     )
            # )
            # self.params.append(
            #     (name,
            #      weight_size,
            #      int(module.weight.size(0) * module.weight.size(1))
            #     )
            # )
            self.profile.append(
                (name,
                 input_shape,
                 input_size,
                 output_shape,
                 output_size,
                 int(module.weight.size(0) * module.weight.size(1)),
                 weight_shape,
                 int(module.weight.size(0) * module.weight.size(1))
                )
            )

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(hook_linear))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        # return self.macs, self.params
        return self.profile


if __name__ == '__main__':

    # find the 'out = model(x)' in your code, my method is based on pytorch hook
    # example input
    x = torch.randn(1, 3, 224, 224)
    # example model
    model = torchvision.models.mobilenet_v2(pretrained=False)

    profile = ProfileConv(model)
    MACs, params = profile(x)

    print('number of conv&fc layers:', len(MACs))
    # print(sum(MACs) / 1e9, 'GMACs')
    # print(sum(params) / 1e6, 'M parameters')
