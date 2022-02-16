# Pytorch tpu 사용방법



```
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
```

torch, xla 버전 맞춰서 설치



```python
import torch_xla
import torch_xla.core.xla_model as xm
```

```python
device = xm.xla_device()
torch.set_default_tensor_type('torch.FloatTensor')
```

```python
#optimizer.step() 대신에 사용
xm.optimizer_step(optimizer, barrier=True)
```