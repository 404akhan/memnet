import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, autograd
from torch.utils.data import DataLoader

from babi import BabiDataset, pad_collate
from torch.nn.utils import clip_grad_norm

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
        memories = keys
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
    def __init__(self, hidden_dim, max_num_sentences=150, vocab_size=50):
        super(RecurrentEntityNetwork, self).__init__()

        self.max_num_sentences = max_num_sentences
        self.embed_dim = hidden_dim
        self.num_mem_slots = 20
        self.vocab_size = vocab_size

        self.memory_mask = nn.Parameter(torch.randn(max_num_sentences, 1))
        self.question_mask = nn.Parameter(torch.randn(max_num_sentences, 1))

        self.embedding = nn.Embedding(vocab_size + self.num_mem_slots, hidden_dim, padding_idx=0)
        init.uniform(self.embedding.weight, a=-(3 ** 0.5), b=3 ** 0.5)

        self.cell = MemoryCell(self.num_mem_slots, hidden_dim)

        # Fully connected linear layers.
        self.C = nn.Linear(hidden_dim, hidden_dim)
        self.H = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Z = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.prelu_outputs = nn.ReLU()

        # Initialize weights.
        init.xavier_normal(self.C.weight)
        init.xavier_normal(self.H.weight)
        init.xavier_normal(self.Z.weight)

        self.memory_mask.data.fill_(1)
        self.question_mask.data.fill_(1)

    def forward(self, contexts, questions):
        batch_size, context_length, context_num_words = contexts.size()
        _, question_length = questions.size()

        # List of sentence embeddings for every story in a batch. (num. sentences, batch size, encoder dim.)

        contexts = self.embedding(contexts.view(batch_size, -1))
        contexts = contexts.view(batch_size, context_length, context_num_words, -1)

        questions = self.embedding(questions)

        memory_mask = self.memory_mask[:context_length].unsqueeze(0).unsqueeze(2).expand(*contexts.size())
        question_mask = self.question_mask[:question_length].unsqueeze(0).expand(*questions.size())

        memory_inputs = torch.sum(contexts * memory_mask, dim=2).squeeze().t()
        question_inputs = torch.sum(questions * question_mask, dim=1).squeeze()

        # Compute memory updates.

        keys = torch.arange(self.vocab_size, self.vocab_size + self.num_mem_slots)
        keys = torch.autograd.Variable(keys.unsqueeze(0).expand(batch_size, self.num_mem_slots).long().cuda())

        keys = self.embedding(keys).view(batch_size * self.num_mem_slots, -1)

        network_graph = self.cell(memory_inputs, keys)
        network_graph = self.C(network_graph).view(batch_size, self.num_mem_slots, self.embed_dim)

        # Apply attention to the entire acyclic graph using the questions.

        attention_energies = network_graph * question_inputs.unsqueeze(1).expand_as(network_graph)
        attention_energies = attention_energies.sum(dim=-1)

        attention_weights = F.softmax(attention_energies).expand_as(network_graph)

        attended_network_graph = (network_graph * attention_weights).sum(dim=1).squeeze()

        # Condition the fully-connected layer using the questions.

        outputs = self.prelu_outputs(question_inputs + self.H(attended_network_graph))
        outputs = self.Z(outputs)

        return outputs


HIDDEN_DIM = 100
BATCH_SIZE = 100

NUM_EPOCHS = 250

LOG_FILE = "memnet.txt"

if __name__ == '__main__':
    dataset = BabiDataset()
    vocab_size = len(dataset.QA.VOCAB)

    criterion = nn.CrossEntropyLoss(size_average=False)
    model = RecurrentEntityNetwork(HIDDEN_DIM, 130, vocab_size)
    model.cuda()

    early_stopping_counter = 0
    best_accuracy = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(NUM_EPOCHS):
        dataset.set_mode('train')
        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
        )

        model.train()
        if early_stopping_counter < 20:
            total_accuracy = 0
            num_batches = 0
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                contexts, questions, answers = data
                contexts = autograd.Variable(contexts.long().cuda())
                questions = autograd.Variable(questions.long().cuda())
                answers = autograd.Variable(answers.cuda())

                outputs = model(contexts, questions)

                l2_loss = 0
                for name, param in model.named_parameters():
                    l2_loss += 0.001 * torch.sum(param * param)
                loss = criterion(outputs, answers) + l2_loss

                predictions = F.softmax(outputs).data.max(1)[1]
                correct = predictions.eq(answers.data).cpu().sum()
                acc = correct * 100. / len(contexts)

                loss.backward()
                clip_grad_norm(model.parameters(), 40)
                total_accuracy += acc
                num_batches += 1

                if batch_idx % 20 == 0:
                    print('[Epoch %d] [Training] loss : %f, acc : %f, batch_idx : %d' % (
                        epoch, loss.data[0], total_accuracy / num_batches, batch_idx
                    ))
                optimizer.step()

            dataset.set_mode('valid')
            valid_loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
            )

            model.eval()
            total_accuracy = 0
            num_batches = 0
            for batch_idx, data in enumerate(valid_loader):
                contexts, questions, answers = data
                contexts = autograd.Variable(contexts.long().cuda())
                questions = autograd.Variable(questions.long().cuda())
                answers = autograd.Variable(answers.cuda())

                outputs = model(contexts, questions)

                l2_loss = 0
                for name, param in model.named_parameters():
                    l2_loss += 0.001 * torch.sum(param * param)
                loss = criterion(outputs, answers) + l2_loss

                predictions = F.softmax(outputs).data.max(1)[1]
                correct = predictions.eq(answers.data).cpu().sum()
                acc = correct * 100. / len(contexts)

                total_accuracy += acc
                num_batches += 1

            total_accuracy = total_accuracy / num_batches
            if total_accuracy > best_accuracy:
                best_accuracy = total_accuracy
                best_state = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print('[Epoch %d] [Validate] Accuracy : %f' % (epoch, total_accuracy))
            with open(LOG_FILE, 'a') as fp:
                fp.write('[Epoch %d] [Validate] Accuracy : %f' % (epoch, total_accuracy) + '\n')
            if total_accuracy == 1.0:
                break
        else:
            print('Early Stopping at Epoch %d, Valid Accuracy : %f' % (epoch, best_accuracy))
            break

    dataset.set_mode('test')
    test_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )
    test_acc = 0
    num_batches = 0

    for batch_idx, data in enumerate(test_loader):
        contexts, questions, answers = data
        contexts = autograd.Variable(contexts.long().cuda())
        questions = autograd.Variable(questions.long().cuda())
        answers = autograd.Variable(answers.cuda())

        model.state_dict().update(best_state)

        outputs = model(contexts, questions)

        l2_loss = 0
        for name, param in model.named_parameters():
            l2_loss += 0.001 * torch.sum(param * param)
        loss = criterion(outputs, answers) + l2_loss

        predictions = F.softmax(outputs).data.max(1)[1]
        correct = predictions.eq(answers.data).cpu().sum()
        acc = correct * 100. / len(contexts)

        test_acc += acc
        num_batches += 1
    print('[Epoch %d] [Test] Accuracy : %f' % (epoch, test_acc / num_batches))
    with open(LOG_FILE, 'a') as fp:
        fp.write('[Epoch %d] [Test] Accuracy : %f' % (epoch, test_acc / num_batches) + '\n')