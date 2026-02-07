import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

# ADD THIS FUNCTION:
def sample_clients_dynamic(args, total_clients, online_num, random_state, epoch_index):
    """Dynamic participant sampling"""
    if hasattr(args, 'min_participants') and args.min_participants is not None:
        import random
        min_p = args.min_participants
        max_p = args.max_participants if (hasattr(args, 'max_participants') and args.max_participants) else min_p
        num_to_sample = random.randint(min_p, max_p)
        online_clients = random.sample(total_clients, num_to_sample)
        online_clients.sort()
        print(f"\n🎲 Round {epoch_index}: Sampled {len(online_clients)}/{len(total_clients)} participants: {online_clients}")
        return online_clients
    else:
        return random_state.choice(total_clients, online_num, replace=False).tolist()

class FedAvG(FederatedModel):
    NAME = 'fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedAvG, self).__init__(nets_list,args,transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = sample_clients_dynamic(self.args, total_clients, self.online_num,
                                           self.random_state, self.epoch_index)  # Changed!
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

