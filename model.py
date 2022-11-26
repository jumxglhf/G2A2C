from re import T
import torch
import dgl
import torch.nn.functional as F
import pyro
import copy
import numpy as np


class Node_Generator(torch.nn.Module):
    def __init__(self, in_feats, h_feats, discrete_feat):
        super(Node_Generator, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both')
        self.discrete_feat = discrete_feat
        if discrete_feat:
            self.generator = torch.nn.Linear(3 * h_feats, in_feats)
        else:
            self.generator = torch.nn.Linear(3 * h_feats, 2*in_feats)
        self.activation = torch.nn.LeakyReLU()
        self.in_feats = in_feats


    def forward(self, g, node_index, temperature=0.1):
        h = self.activation(self.conv1(g, g.ndata['feat']))
        h = self.activation(self.conv2(g, h))
        h = torch.cat((h.sum(0), h.max(0).values, h[node_index]))
        feature_dist = self.generator(h)
        if self.discrete_feat:
            feature_dist = torch.sigmoid(feature_dist)
            dist = BernoulliStraightThrough(probs=feature_dist)
            feat = dist.rsample()
            log_prob = dist.log_prob(feat).mean()
            feat = F.normalize(feat, dim=0)
            return feat, feat.mean(), log_prob
        else:
            mu = feature_dist[:self.in_feats]
            sigma = torch.abs(feature_dist[self.in_feats:])
            dist = torch.distributions.Normal(mu, sigma)
            feat = dist.sample()
            log_prob = dist.log_prob(feat).mean()
            return feat, [mu, sigma], log_prob


class Edge_Sampler(torch.nn.Module):
    def __init__(self, in_feats, h_feats, alpha_n=1000000):
        super(Edge_Sampler, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both')
        self.regressor = torch.nn.Linear(in_feats + h_feats * 2, 1)
        self.activation = torch.nn.LeakyReLU()
        self.alpha_n = alpha_n

    def forward(self, g, node_index, edge_set):
        
        num_candidate_dst = g.number_of_nodes() - 1 - len(edge_set)
        mask = [[i for i in range(g.number_of_nodes() - 1) if i not in edge_set]]
        # if khop == 0:
        #     num_candidate_dst = g.number_of_nodes() - 1 - len(edge_set)
        #     mask = [[i for i in range(g.number_of_nodes() - 1) if i not in edge_set]]
        # else:
        #     candidates = dgl.khop_in_subgraph(g, node_index, k=khop)[0].ndata[dgl.NID]
        #     if len(candidates) - len(edge_set) - 1 <= 0:
        #         if g.number_of_nodes() < 10000:
        #             candidates = np.arange(g.number_of_nodes() - 1)
        #         else:
        #             candidates = np.random.permutation(np.arange(g.number_of_nodes() - 1))[:10000]
        #         candidates = [c.item() for c in candidates if c.item() not in edge_set]
        #     else:
        #         candidates = [c.item() for c in candidates if c.item() not in edge_set][:-1]
        #     num_candidate_dst = len(candidates)
        #     mask = candidates

        node_neighbors = g.predecessors(node_index)[:-1] # [:-1] excludes the connection to the target node
        node_neighbors_mask = torch.zeros(g.number_of_nodes() - 1)
        node_neighbors_mask[node_neighbors] = self.alpha_n
        node_neighbors_mask = node_neighbors_mask[mask].to(g.device)

        sampled_feature = g.ndata['feat'][-1].expand((num_candidate_dst, -1))

        h = self.activation(self.conv1(g, g.ndata['feat']))
        h = self.activation(self.conv2(g, h))
        target_node_embedding = h[node_index].expand((num_candidate_dst, -1))        
        candidate_dst_h = h[mask]
        # if khop != 0:
        #     g_ = dgl.node_subgraph(g, mask + [node_index.item()])
        #     h = self.activation(self.conv1(g_, g_.ndata['feat']))
        #     h = self.activation(self.conv2(g_, h))
        #     target_node_embedding = h[-1].expand((num_candidate_dst, -1))
        #     candidate_dst_h = h[:-1]
        # else:
        #     h = self.activation(self.conv1(g, g.ndata['feat']))
        #     h = self.activation(self.conv2(g, h))
        #     target_node_embedding = h[node_index].expand((num_candidate_dst, -1))        
        #     candidate_dst_h = h[mask]
        
        h = torch.cat((candidate_dst_h, sampled_feature, target_node_embedding), dim = 1)
        feature_dist = self.regressor(h).squeeze() + node_neighbors_mask
        feature_dist = F.softmax(feature_dist, dim=0)
        dist = torch.distributions.OneHotCategorical(probs=feature_dist)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        sample_ = torch.zeros(g.number_of_nodes() - 1).to(g.device)
        sample_[mask] = sample
        return sample_, log_prob

class Value_Predictor(torch.nn.Module):
    def __init__(self, in_feats, h_feats, n_class = 7):
        super(Value_Predictor, self).__init__()
        self.gc1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.gc2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both')
        self.regressor = torch.nn.Linear(h_feats, n_class)
        self.projector = torch.nn.Linear(h_feats+n_class, n_class)
        self.activation = torch.nn.LeakyReLU()
        self.n_class = n_class

    def forward(self, g, node_index, label):
        g, node_index = dgl.khop_in_subgraph(g, node_index, 2) 
        h1 = self.activation(self.gc1(g, g.ndata['feat']))
        h1 = self.activation(self.gc2(g, h1))
        logits = F.log_softmax(self.regressor(h1), dim = -1)
        h = torch.cat((h1[node_index], logits[node_index]), dim = -1)
        h = self.projector(h)
        return F.cross_entropy(h.reshape(1, self.n_class), label.reshape(1))


class GCN(torch.nn.Module):
    # code copied from dgl examples
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes, norm='both')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    def train(self, g, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        test_idx = None
        best_model = None
        for e in range(1000):
            # Forward
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_model = copy.deepcopy(model)
                best_val_acc = val_acc
                best_test_acc = test_acc
                test_idx = test_mask.nonzero(as_tuple=True)[0][(pred[test_mask] == labels[test_mask]).nonzero(as_tuple=True)[0]]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Victim model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(best_val_acc.item()*100, best_test_acc.item()*100))
        return best_model, test_idx
        
class SGC(torch.nn.Module):
    
    # code copied from dgl examples
    def __init__(self, in_feats, h_feats, num_classes):
        super(SGC, self).__init__()
        self.conv = dgl.nn.SGConv(in_feats, num_classes, k=2)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h

    def train(self, g, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        test_idx = None
        best_model = None
        for e in range(1000):
            # Forward
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_model = copy.deepcopy(model)
                best_val_acc = val_acc
                best_test_acc = test_acc
                test_idx = test_mask.nonzero(as_tuple=True)[0][(pred[test_mask] == labels[test_mask]).nonzero(as_tuple=True)[0]]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Victim model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(best_val_acc.item()*100, best_test_acc.item()*100))
        return best_model, test_idx

class APPNP(torch.nn.Module):
    # code copied from dgl examples
    def __init__(self, in_feats, h_feats, num_classes):
        super(APPNP, self).__init__()
        self.mlp = torch.nn.Linear(in_feats, num_classes)
        self.conv = dgl.nn.APPNPConv(k=3, alpha=0.5)

    def forward(self, g, in_feat):
        in_feat = self.mlp(in_feat)
        h = self.conv(g, in_feat)
        return h

    def train(self, g, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        test_idx = None
        best_model = None
        for e in range(1000):
            # Forward
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_model = copy.deepcopy(model)
                best_val_acc = val_acc
                best_test_acc = test_acc
                test_idx = test_mask.nonzero(as_tuple=True)[0][(pred[test_mask] == labels[test_mask]).nonzero(as_tuple=True)[0]]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Victim model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(best_val_acc.item()*100, best_test_acc.item()*100))
        return best_model, test_idx

class GAT(torch.nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads=[4, 4]):
        super().__init__()
        self.gat_layers = torch.nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(dgl.nn.GATConv(in_size, hid_size, heads[0], feat_drop=0., attn_drop=0., activation=F.elu))
        self.gat_layers.append(dgl.nn.GATConv(hid_size*heads[0], out_size, heads[1], feat_drop=0., attn_drop=0., activation=None))
        
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

    def train(self, g, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        test_idx = None
        best_model = None
        for e in range(1000):
            # Forward
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_model = copy.deepcopy(model)
                best_val_acc = val_acc
                best_test_acc = test_acc
                test_idx = test_mask.nonzero(as_tuple=True)[0][(pred[test_mask] == labels[test_mask]).nonzero(as_tuple=True)[0]]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Victim model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(best_val_acc.item()*100, best_test_acc.item()*100))
        return best_model, test_idx

class BernoulliStraightThrough(torch.distributions.Bernoulli):
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al, 2013)
    """
    has_rsample = True

    def rsample(self, sample_shape=torch.Size()):

        samples = self.sample(sample_shape)
        probs = self._param  # cached via @lazy_property
        return samples + (probs - probs.detach())

    