import torch
from geomloss import SamplesLoss

sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur=0.05,
    debias=True,
)

prediction = torch.randn(32, 100, 64, requires_grad=True)
target = torch.randn(32, 100, 64)

loss = sinkhorn(prediction, target)

print(loss.shape)
print(loss)

loss.backward()