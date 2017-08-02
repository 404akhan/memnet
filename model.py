import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init as init


cuda_exist = torch.cuda.is_available()

class MemoryCell(nn.Module):
    def __init__(self, num_mem_slots, embed_dim):
        super(MemoryCell, self).__init__()

        self.num_mem_slots = num_mem_slots
        self.embed_dim = embed_dim

        # Memory update linear layers.
        self.U = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.W = nn.Linear(embed_dim, embed_dim)
        self.J = nn.Linear(3 * embed_dim, embed_dim)

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

            join = torch.cat([self.U(memories), self.V(sentence), self.W(keys)], dim=1)
            join = self.J(F.relu(join))
            candidate_memories = self.prelu_memory(join)

            updated_memories = memories + memory_gates * candidate_memories
            updated_memories = updated_memories / (
                updated_memories.norm(p=2, dim=-1).expand_as(updated_memories) + 1e-12)

            memories = updated_memories

        return memories


class RecurrentEntityNetwork(nn.Module):
    def __init__(self, hidden_dim, dim_obj_qst=37, num_classes=10, num_mem_slots=10, qst_dim=11):
        super(RecurrentEntityNetwork, self).__init__()

        self.embed_dim = hidden_dim
        self.num_mem_slots = num_mem_slots

        self.embedding = nn.Embedding(self.num_mem_slots, hidden_dim, padding_idx=0)
        init.uniform(self.embedding.weight, a=-(3 ** 0.5), b=3 ** 0.5)

        self.cell = MemoryCell(self.num_mem_slots, hidden_dim)

        # Fully connected linear layers
        self.C = nn.Linear(dim_obj_qst, hidden_dim)
        self.H = nn.Linear(2 * hidden_dim, hidden_dim)
        self.Q = nn.Linear(qst_dim, hidden_dim)

        # Initialize weights.
        init.xavier_normal(self.C.weight)
        init.xavier_normal(self.H.weight)
        init.xavier_normal(self.Q.weight)

    def forward(self, memory_inputs, question_inputs):
        # memory_inputs | seq_len x bsize x dim_obj_qst
        # question_inputs | bsize x qst_dim

        seq_len, bsize, dim_obj_qst = memory_inputs.size()
        memory_inputs = self.C(memory_inputs.view(seq_len * bsize, -1))
        memory_inputs = F.relu(memory_inputs).view(seq_len, bsize, -1) # check relu performance on top of this
        question_inputs = F.relu(self.Q(question_inputs))
        
        # Compute memory updates.

        keys = torch.arange(0, self.num_mem_slots)
        rem1 = keys.unsqueeze(0).expand(bsize, self.num_mem_slots).long()
        if cuda_exist: 
            rem1 = rem1.cuda()
        keys = torch.autograd.Variable(rem1)

        keys = self.embedding(keys).view(bsize * self.num_mem_slots, -1)

        network_graph = self.cell(memory_inputs, keys)
        network_graph = network_graph.view(bsize, self.num_mem_slots, self.embed_dim)

        # Apply attention to the entire acyclic graph using the questions.

        attention_energies = network_graph * question_inputs.unsqueeze(1).expand_as(network_graph)
        attention_energies = attention_energies.sum(dim=-1)

        attention_weights = F.softmax(attention_energies).expand_as(network_graph)

        attended_network_graph = (network_graph * attention_weights).sum(dim=1).squeeze()

        # Condition the fully-connected layer using the questions.

        outputs = F.relu(self.H(torch.cat([question_inputs, attended_network_graph], dim=1)))

        return outputs


class RN(nn.Module):

    def __init__(self,args):
        super(RN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array(self.cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        hidden_dim = 100
        dim_obj_qst = 37
        num_classes = 10
        num_mem_slots = 10
        self.mnet = RecurrentEntityNetwork(hidden_dim, dim_obj_qst, num_classes, num_mem_slots)
        if cuda_exist:
            self.mnet.cuda()

        self.H = nn.Linear(2 * hidden_dim, hidden_dim)
        self.Z = nn.Linear(hidden_dim, num_classes)
        init.xavier_normal(self.H.weight)
        init.xavier_normal(self.Z.weight)


    def cvt_coord(self, i):
        return [(i/5-2)/2., (i%5-2)/2.]


    def forward(self, img, qst):
        # img | 64 x 3 x 75 x 75
        # qst | 64 x 11
        qst_rem = qst
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        ## x = (64 x 24 x 5 x 5)
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,25,1)

        objects = torch.cat([x_flat, qst], 2).permute(1, 0, 2).contiguous()
        # objects | 25 x 64 x 37

        inv_idx = Variable(torch.arange(objects.size(0)-1, -1, -1).long())
        if cuda_exist:
            inv_idx = inv_idx.cuda()
        inv_objects = objects.index_select(0, inv_idx)
        # inv_objects = objects[inv_idx]

        outputs1 = self.mnet(objects, qst_rem)
        outputs2 = self.mnet(inv_objects, qst_rem)
        
        out_concat = torch.cat([outputs1, outputs2], 1)

        logits = F.relu(self.H(out_concat))
        logits = self.Z(logits)

        return F.log_softmax(logits)


    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        

    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy


    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}.pth'.format(epoch))
