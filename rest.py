import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


class MemoryCell(nn.Module):
    def __init__(self, num_mem_slots, embed_dim):
        super(MemoryCell, self).__init__()

        self.num_mem_slots = num_mem_slots
        self.embed_dim = embed_dim

        # Memory update linear layers.
        self.U = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)

        self.prelu_memory = nn.PReLU(init=1)

        init.xavier_normal(self.U.weight)
        init.xavier_normal(self.V.weight)
        init.xavier_normal(self.W.weight)

    def forward(self, inputs, keys):
        # inputs    | seq_len x bsize x hid_dim
        # keys      | bsize * mem_slots x hid_dim

        memories = keys # memory value initialized being equal to key
        memory_inputs = inputs

        for index, sentence in enumerate(memory_inputs):
            # Compute memory updates.

            sentence = sentence.unsqueeze(1).repeat(1, self.num_mem_slots, 1)
            sentence = sentence.view_as(memories)

            memory_gates = F.sigmoid((sentence * (memories + keys)).sum(dim=-1))
            memory_gates = memory_gates.expand_as(memories)

            candidate_memories = self.prelu_memory(self.U(memories) + self.V(sentence) + self.W(keys))

            updated_memories = memories + memory_gates * candidate_memories
            updated_memories = updated_memories / (
                updated_memories.norm(p=2, dim=-1).expand_as(updated_memories) + 1e-12)

            memories = updated_memories

        return memories


class RecurrentEntityNetwork(nn.Module):
    def __init__(self, hidden_dim, dim_obj_qst=37, num_classes=10):
        super(RecurrentEntityNetwork, self).__init__()

        self.embed_dim = hidden_dim
        self.num_mem_slots = 20

        self.embedding = nn.Embedding(self.num_mem_slots, hidden_dim, padding_idx=0)
        init.uniform(self.embedding.weight, a=-(3 ** 0.5), b=3 ** 0.5)

        self.cell = MemoryCell(self.num_mem_slots, hidden_dim)

        # Fully connected linear layers
        self.C = nn.Linear(dim_obj_qst, hidden_dim)
        self.H = nn.Linear(self.num_mem_slots * hidden_dim, hidden_dim)
        self.Z = nn.Linear(hidden_dim, num_classes)

        # Initialize weights.
        init.xavier_normal(self.C.weight)
        init.xavier_normal(self.H.weight)
        init.xavier_normal(self.Z.weight)

    def forward(self, memory_inputs):
        # memory_inputs | seq_len x bsize x dim_obj_qst

        seq_len, bsize, dim_obj_qst = memory_inputs.size()
        memory_inputs = self.C(memory_inputs.view(seq_len * bsize, -1)).view(seq_len, bsize, -1) # check relu performance on top of this

        # Compute memory updates.

        keys = torch.arange(0, self.num_mem_slots)
        keys = torch.autograd.Variable(keys.unsqueeze(0).expand(bsize, self.num_mem_slots).long())

        keys = self.embedding(keys).view(bsize * self.num_mem_slots, -1)

        network_graph = self.cell(memory_inputs, keys)
        network_graph = network_graph.view(bsize, self.num_mem_slots * self.embed_dim)

        outputs = F.relu(self.H(network_graph))
        outputs = self.Z(outputs)

        # logits, bsize x num_classes, change to log_softmax
        return outputs


seq_len = 25
bsize = 64
dim_obj_qst = 37
hidden_dim = 100
num_classes = 10

model = RecurrentEntityNetwork(hidden_dim, dim_obj_qst, num_classes)
memory_inputs = Variable(torch.FloatTensor(seq_len, bsize, dim_obj_qst))

x = model(memory_inputs)
print(x)

