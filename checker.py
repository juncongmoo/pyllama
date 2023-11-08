import time

import matplotlib.pyplot as plt
import torch
from hiq.vis import print_model


def grid(width, height):
    hrange = torch.arange(width).unsqueeze(0).repeat([height, 1]).div(width)
    vrange = torch.arange(height).unsqueeze(1).repeat([1, width]).div(height)
    output = torch.stack([hrange, vrange], 0)
    return output


def checker(width, height, freq):
    hrange = (
        torch.arange(width)
        .reshape([1, width])
        .mul(freq / width / 2.0)
        .fmod(1.0)
        .gt(0.5)
    )
    vrange = (
        torch.arange(height)
        .reshape([height, 1])
        .mul(freq / height / 2.0)
        .fmod(1.0)
        .gt(0.5)
    )
    output = hrange.logical_xor(vrange).float()
    return output


# Note the inputs are grid coordinates and the target is a checkerboard
inputs = grid(512, 512).unsqueeze(0).cuda()
targets = checker(512, 512, 8).unsqueeze(0).unsqueeze(1).cuda()


class Net(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(2, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1, 1),
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.net(x)


net = Net().cuda()
print_model(net)

loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), 0.001)

scaler = torch.cuda.amp.GradScaler()

start_time = time.time()

for i in range(500):
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward() #?????
    scaler.step(opt)
    scaler.update()
print(loss)

print(time.time() - start_time)

plt.subplot(1, 2, 1)
plt.imshow(outputs.squeeze().detach().cpu().float())
plt.subplot(1, 2, 2)
plt.imshow(targets.squeeze().cpu().float())
plt.show()
