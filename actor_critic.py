import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import attack_utils as utils
import model
import copy
from tqdm import tqdm
import dgl


parser = argparse.ArgumentParser(description='GA2C Source Code')

parser.add_argument('--dataset', type=str, default='citeseer', help='dataset to attack')

parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--khop_feat', type=int, default=1, help='order of subgraphs to generate node features')
parser.add_argument('--khop_edge', type=int, default=0, help='order of subgraphs to wire node, 0 for full graph')
parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.99)')

parser.add_argument('--node_budget', type=int, default=1, help='node budget per node')
parser.add_argument('--edge_budget', type=int, default=1, help='edge budget')
parser.add_argument('--feature_budget', type=float, default=1, help='feature budget multiplier, dummy for continuous case')

parser.add_argument('--patience', type=int, default=2, help='patience for early stopping')
parser.add_argument('--accumulation_step', type=int, default=1, help='step to update params')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=int, default=0, help='device ID for GPU')
parser.add_argument('--save_dir', type=str, default='', help='save dir for injected nodes and edges')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--victim_model', type=str, default='gcn', help='model architecture of the victim model')
parser.add_argument('--repeat', type=int, default=5, help='number of runs')

args = parser.parse_args()

device = 'cuda:{}'.format(str(args.device))
hid_dim = args.hid_dim
k_hop_feat = args.khop_feat
k_hop_edge = args.khop_edge
epochs = args.epoch
discount_factor = args.gamma
node_budget = args.node_budget
accumulation_step = args.accumulation_step
patience = args.patience


def finish_episode(step, accumulation_step):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    global action_buffer, reward_buffer, feature_loss
    R = 0
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    # calculate the true value using rewards returned from the environment
    for r in reward_buffer[::-1]:
        # calculate the discounted value
        R = r + discount_factor * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    ep_reward = returns.sum().item()
    if len(returns) != 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(action_buffer, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, R))
    
    policy_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()
    # sum up all the values of policy_losses and value_losses
    loss = policy_loss + value_loss+ feature_loss
    # perform backprop
    loss.backward()

    if step % accumulation_step == 0:
        optimizer.step()
        step = 0
        optimizer.zero_grad()

    del action_buffer, reward_buffer


    return step, ep_reward #, policy_loss.item(), value_loss.item(), feature_loss.item()


final_accs = []


for rep in range(args.repeat):
    print('Currently runing at {} round'.format(str(rep)))
    env = utils.GraphEnv(args.dataset, hid_dim, device, args.victim_model)
    utils.setup_seed(args.seed)

    node_generator = model.Node_Generator(env.feature_dim, hid_dim*2, env.discrete_feat).to(device)
    edge_sampler = model.Edge_Sampler(env.feature_dim, hid_dim*2).to(device)
    value_predictor = model.Value_Predictor(env.feature_dim, hid_dim*2, env.n_class).to(device)

    optimizer = optim.Adam(list(node_generator.parameters()) + \
        list(edge_sampler.parameters()) + list(value_predictor.parameters()), lr=1e-4)
    eps = np.finfo(np.float32).eps.item()

    print('======= Budget Overview =======')
    print('Dataset: {}'.format(args.dataset))
    print('Node Budget: {:.1f}'.format(args.node_budget))
    print('{} Feature {}: {:.2f}'.format('Discrete' if env.discrete_feat else 'Continuous', 'Budget' if env.discrete_feat else 'Mass',args.feature_budget*env.feature_budget))
    print('Edge Budget: {:.1f}'.format(args.edge_budget))
    print('===============================')

    lowerest_acc = 100
    cnt = 0
    best_solution = None

    for e in range(epochs):
        total_success = 0
        step = 0
        unsolved_set = []
        accumulated_ep_reward = 0
        iters = 0
        solution = {}
        shuffled_indices = np.random.permutation(np.arange(len(env.test_idx)))
        if args.verbose:
            pbar = env.test_idx[shuffled_indices]
        else:
            pbar = tqdm(env.test_idx[shuffled_indices])
            
        for node_index in pbar:

            graph = copy.deepcopy(env.g)
            if args.khop_edge != 0:
                graph, node_index = dgl.khop_in_subgraph(graph, node_index, args.khop_edge)

            # Define memories for back-propagation 
            action_buffer = []
            reward_buffer = []
            feature_loss = 0
            step += 1
            iters += 1
            node_attribute_buffer = []
            link_buffer = []

            for _ in range(node_budget):
                # generate an adversarial node
                sg, idx = dgl.khop_in_subgraph(graph, node_index, k_hop_feat)
                feature, num_feat, action_log_prob = node_generator(sg, idx.item())
                # regularize feature bins if excessive features are generated
                
                if env.discrete_feat:
                    feature_loss += F.mse_loss(num_feat, torch.floor(env.feature_budget*args.feature_budget)/env.feature_dim)/node_budget
                else:
                    mu, sigma = num_feat[0], num_feat[1]
                    feature_loss += F.kl_div(mu, env.mu, reduction='batchmean') \
                        + F.kl_div(sigma, env.sigma, reduction='batchmean')

                # buffer current solution
                if e > 0:
                    node_attribute_buffer.append(feature.detach().cpu())
                    link_buffer.append([node_index.item()])

                # for each injected node, wire it to the graph according to our budget
                for i in range(args.edge_budget):
                    # first inject into the victim node
                    if i == 0: 
                        previous_graph = graph.clone()
                        graph = utils.inject_node(graph, feature)
                        graph = utils.wire_edge(graph, node_index) # conencting directly to the target node without learning
                        edge_set = [node_index]
                        reward, success = env.get_reward(node_index, previous_graph, graph)
                        reward_buffer.append(reward)
                        state_value = value_predictor(graph, node_index, graph.ndata['label'][node_index])
                        action_buffer.append([action_log_prob, state_value])
                        dst = node_index
                    else:
                        # edge wiring
                        edge_dist, edge_log_prob = edge_sampler(graph, node_index, edge_set)
                        # state value prediction
                        state_value = value_predictor(graph, node_index, graph.ndata['label'][node_index])
                        action_buffer.append([edge_log_prob + action_log_prob, state_value])

                        # getting rewards for the generated actions
                        previous_graph = graph.clone()
                        dst = (edge_dist == 1).nonzero(as_tuple=True)[0][0]
                        graph = utils.wire_edge(graph, dst)
                        reward, success = env.get_reward(node_index, previous_graph, graph)
                        # memory buffering
                        reward_buffer.append(reward)
                        edge_set.append(dst)
                    if e > 0:
                        link_buffer[-1].append(dst.item())

                    # running out of nodes to connect 
                    if len(edge_set)+1 == graph.number_of_nodes():
                        break

                    # node already cracked
                    if success:
                        break

                if success:
                    total_success += 1
                    break

            # saving working solutions
            if success and e > 0:
                solution[node_index.item()] = {'node_feat': torch.stack(node_attribute_buffer), 'edges': link_buffer}
            else:
                unsolved_set.append(node_index.item())
            step, ep_reward = finish_episode(step, accumulation_step)
            accumulated_ep_reward += ep_reward
            cuurent_acc = (len(env.test_idx)-total_success)*100/ env.total_test_nodes
            if not args.verbose:
                pbar.set_description('Epoch: {}, Current Accuracy: {:.2f} ({}/{}), Episode Reward: {:.2f}'.format(e+1, cuurent_acc, len(env.test_idx)-total_success, env.total_test_nodes, accumulated_ep_reward/iters))
        
        # for next epoch, we only focus on the unsolved nodes
        # env.test_idx = np.array(unsolved_set)

        # early stopping 
        if e > 0:
            if cuurent_acc < lowerest_acc:
                lowerest_acc = cuurent_acc
                cnt = 0
                best_solution = solution
            else:
                cnt += 1
        
        if cnt == patience:
            print('Early Stopping. Current Accuracy is {:.2f}.'.format(lowerest_acc))
            if not args.verbose:
                print('Solution saved to {}'.format(args.save_dir))

                # cache solution
                import pickle 
                with open(f'{args.save_dir}'+f'{args.dataset}.pickle', 'wb') as handle:
                    pickle.dump(best_solution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            final_accs.append(lowerest_acc)
            break

print('Dataset: {} Victim Model: {}, Average ACC: {:.2f}, STD: {:.2f}'.format(args.dataset, args.victim_model, np.mean(final_accs), np.std(final_accs)))