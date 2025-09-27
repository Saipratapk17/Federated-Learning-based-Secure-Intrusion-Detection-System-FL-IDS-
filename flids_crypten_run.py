import threading
import torch
import torch.nn as nn
import crypten
import io
import onnx
from onnx import numpy_helper
from crypten.config import cfg
from omegaconf import OmegaConf
import numpy as np  # ✅ Fix for NameError

# ✅ Define a simple CrypTen-compatible MLP model
class MLP(nn.Module):
    def __init__(self, input_size=25, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# ✅ Safe ONNX export (serial, not inside threads)
def export_onnx_model(model, dummy_input):
    f = io.BytesIO()
    torch.onnx.export(model, dummy_input, f, opset_version=13)
    f.seek(0)

    # Fix any non-float initializers
    onnx_model = onnx.load_model(f)
    for i, tensor in enumerate(onnx_model.graph.initializer):
        np_tensor = numpy_helper.to_array(tensor)
        if not np.issubdtype(np_tensor.dtype, np.floating):
            corrected_tensor = numpy_helper.from_array(np_tensor.astype(np.float32), tensor.name)
            onnx_model.graph.initializer[i].CopyFrom(corrected_tensor)

    cleaned_f = io.BytesIO()
    onnx.save_model(onnx_model, cleaned_f)
    cleaned_f.seek(0)
    return cleaned_f

# ✅ Function to run each CrypTen party
def run_party(rank, world_size, model_buffer):
    # Setup config
    cfg.mpc = OmegaConf.create({"provider": "basic"})
    cfg.debug = OmegaConf.create({"validation_mode": False})
    cfg.encoder = OmegaConf.create({"precision_bits": 16})
    cfg.cuda = False

    # Init CrypTen
    crypten.init_thread(rank=rank, world_size=world_size)
    print(f"✅ CrypTen initialized for rank {rank} with world size {world_size}")

    # Load ONNX into CrypTen
    model_buffer.seek(0)
    crypten_model = crypten.nn.from_onnx(model_buffer)
    encrypted_model = crypten_model.encrypt()

    print(f"✅ Encrypted model ready at rank {rank}")

# ✅ Export ONNX model outside threads to avoid AssertionError
dummy_input = torch.randn(1, 25)
model = MLP()
model_buffer = export_onnx_model(model, dummy_input)

# ✅ Launch threads (1 server + 3 clients)
world_size = 4
threads = []
for rank in range(world_size):
    t = threading.Thread(target=run_party, args=(rank, world_size, model_buffer))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
