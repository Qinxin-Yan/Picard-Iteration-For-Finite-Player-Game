import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Exponential
import matplotlib.pyplot as plt


device = torch.device('cpu')

# State setup
states = ['DI', 'DS', 'UI', 'US']
state_index = {s: i for i, s in enumerate(states)}
num_states = len(states)

# Game parameters
T = 5.0
N_sim = 10
k_D, k_I = 0.3, 0.5

lamb=0.5

# Infection and recovery parameters
vH = 0.2
q_inf_D = 0.4
q_inf_U = 0.3
q_rec = 1.0

beta_D = 0.4
beta_U = 0.3

num_MC_samples = 5

# Neural network controller
class ControlNet(nn.Module):
    def __init__(self):
        super(ControlNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, t, x_onehot, mu):
        return self.net(torch.cat([t, x_onehot, mu], dim=-1))

def one_hot_state(x):
    vec = torch.zeros(num_states, device=device)
    vec[state_index[x]] = 1.0
    return vec

def sample_initial_state():
    return np.random.choice(states)

def get_jump_rate(x, epsilon, mu):
    if x == 'DS':
        inf_rate = vH * q_inf_D + beta_D * (mu[state_index['DI']] + lamb*mu[state_index['UI']])
        rate = inf_rate + epsilon
        next_states = ['DI', 'US']
        probs = [inf_rate, epsilon]
    elif x == 'US':
        inf_rate = vH * q_inf_U + beta_U * (mu[state_index['DI']] + lamb*mu[state_index['UI']])
        rate = inf_rate + epsilon
        next_states = ['UI', 'DS']
        probs = [inf_rate, epsilon]
    elif x == 'DI':
        rate = q_rec + epsilon
        next_states = ['DS', 'UI']
        probs = [q_rec, epsilon]
    elif x == 'UI':
        rate = q_rec + epsilon
        next_states = ['US', 'DI']
        probs = [q_rec, epsilon]
    else:
        rate, probs, next_states = 0.0, [], []
    return rate, probs, next_states

def gumbel_softmax_sample(logits, tau=1.0):
    noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.nn.functional.softmax((logits + noise) / tau, dim=-1)

def simulate_joint_dynamics(control_net_tagged, control_net_untagged):
    X = [sample_initial_state() for _ in range(N_sim +1)]
    #for i in range(N_sim+1):
    #    X = ['DS'] * ((N_sim+1) // 2) + ['US'] * ((N_sim+1) // 2)
    #    t = 0.0
    #   next_snapshot_idx = 0
    trajs = [[] for _ in range(N_sim + 1)]
    epsilons_all = [[] for _ in range(N_sim + 1)]
    t = torch.tensor(0.0, device=device)

    while t < T:
        counts = torch.zeros(num_states, device=device)
        for x in X:
            counts[state_index[x]] += 1
        mu = counts / (N_sim + 1)
        jump_rates = []
        rate_info = []

        for i in range(N_sim + 1):
            x_i = X[i]
            x_onehot = one_hot_state(x_i)
            t_tensor = torch.tensor([[t / T]], dtype=torch.float32, device=device)

            if i == 0:
                epsilon = control_net_tagged(t_tensor, x_onehot.unsqueeze(0), mu.unsqueeze(0)).squeeze()
                mu_i = mu
            else:
                s_i = x_i
                s_0 = X[0]
                e_si = one_hot_state(s_i)
                e_s0 = one_hot_state(s_0)
                mu_i = mu - (1 / (N_sim + 1)) * e_si + (1 / (N_sim + 1)) * e_s0
                epsilon = control_net_untagged(t_tensor, x_onehot.unsqueeze(0), mu_i.unsqueeze(0)).squeeze()

            epsilons_all[i].append(epsilon)
            rate, probs, next_states = get_jump_rate(x_i, epsilon, mu_i)
            jump_rates.append(rate)
            rate_info.append((probs, next_states))

        Lambda = sum(jump_rates)
        if Lambda == 0:
            break

        U = torch.rand(1, device=device)
        tau = -torch.log(U) / Lambda
        t_next = torch.minimum(t + tau, torch.tensor([T], dtype=torch.float32, device=device))[0]

        logits = torch.log(torch.tensor(jump_rates, device=device) + 1e-8)
        sample_soft = gumbel_softmax_sample(logits)
        who_jumps = torch.argmax(sample_soft).item()

        probs, next_states = rate_info[who_jumps]
        logits2 = torch.log(torch.tensor(probs, device=device) + 1e-8)
        sample_soft2 = gumbel_softmax_sample(logits2)
        jump_index = torch.argmax(sample_soft2).item()

        next_state = next_states[jump_index]
        X[who_jumps] = next_state
        trajs[who_jumps].append((t.clone(), next_state))

        t = t_next

    return trajs, epsilons_all

def estimate_cost(traj, epsilons, k_D, k_I):
    cost = torch.tensor(0.0, device=device)
    for j in range(len(traj) - 1):
        t_j, x_j = traj[j]
        t_next, _ = traj[j + 1]
        tau = torch.tensor(t_next - t_j, device=device, dtype=torch.float32)
        epsilon_j = epsilons[j]
        control_cost = 0.5 * epsilon_j ** 2
        is_D = torch.tensor(x_j in ['DI', 'DS'], dtype=torch.float32, device=device)
        is_I = torch.tensor(x_j in ['DI', 'UI'], dtype=torch.float32, device=device)
        cost += tau * (control_cost + k_D * is_D + k_I * is_I)
    return cost

# Training
control_net = ControlNet().to(device)
for n in range(5):
    print(f"\nPicard Iteration {n}")
    tagged_net = ControlNet().to(device)
    optimizer = optim.Adam(tagged_net.parameters(), lr=1e-2)

    for step in range(51):
        costs = []
        for _ in range(num_MC_samples):
            trajs, eps = simulate_joint_dynamics(tagged_net, control_net)
            cost = estimate_cost(trajs[0], eps[0], k_D, k_I)
            costs.append(cost)
        avg_cost = torch.stack(costs).mean()
        if step%50==0: 
            print(f"  Step {step}: Tagged cost = {avg_cost.item():.4f}")
        optimizer.zero_grad()
        avg_cost.backward()
        optimizer.step()

    control_net.load_state_dict(tagged_net.state_dict())

# Plotting function
def simulate_multiple_trajectories_and_plot(control_net, T=5.0, N_sim=10, num_trajectories=10):
    time_grid = torch.linspace(0, T, steps=100)
    all_trajectories = torch.zeros(num_trajectories, len(time_grid), num_states)

    for traj_idx in range(num_trajectories):
        X = ['DS'] * (N_sim // 4) + ['US'] * (N_sim // 4)+['DI'] * (N_sim // 4) + ['UI'] * (N_sim // 4)
        t = 0.0
        next_snapshot_idx = 0

        while t < T:
            counts = torch.zeros(num_states, device=device)
            for x in X:
                counts[state_index[x]] += 1
            mu = counts / N_sim

            jump_rates = []
            rate_info = []

            for i in range(N_sim):
                x_i = X[i]
                x_onehot = one_hot_state(x_i)
                t_tensor = torch.tensor([[t / T]], dtype=torch.float32, device=device)
                epsilon = control_net(t_tensor, x_onehot.unsqueeze(0), mu.unsqueeze(0)).squeeze().item()
                rate, probs, next_states = get_jump_rate(x_i, epsilon, mu)
                jump_rates.append(rate)
                rate_info.append((probs, next_states))

            Lambda = sum(jump_rates)
            if Lambda == 0:
                break

            U = torch.rand(1, device=device)
            tau = -torch.log(U) / Lambda
            t += tau.item()
            while next_snapshot_idx < len(time_grid) and t >= time_grid[next_snapshot_idx]:
                dist_snapshot = torch.zeros(num_states)
                for x in X:
                    dist_snapshot[state_index[x]] += 1
                dist_snapshot /= N_sim
                all_trajectories[traj_idx, next_snapshot_idx] = dist_snapshot
                next_snapshot_idx += 1

            logits = torch.log(torch.tensor(jump_rates, device=device) + 1e-8)
            sample_soft = gumbel_softmax_sample(logits)
            who_jumps = torch.argmax(sample_soft).item()
            probs, next_states = rate_info[who_jumps]
            logits2 = torch.log(torch.tensor(probs, device=device) + 1e-8)
            sample_soft2 = gumbel_softmax_sample(logits2)
            jump_index = torch.argmax(sample_soft2).item()
            next_state = next_states[jump_index]
            X[who_jumps] = next_state

    mean_trajectory = all_trajectories.mean(dim=0).cpu().numpy()
    std_trajectory = all_trajectories.std(dim=0).cpu().numpy()
    time_grid = time_grid.cpu().numpy()

    plt.figure(figsize=(10, 6))
    for i, state in enumerate(states):
        plt.plot(time_grid, mean_trajectory[:, i], label=state)
        plt.fill_between(time_grid, mean_trajectory[:, i] - std_trajectory[:, i], mean_trajectory[:, i] + std_trajectory[:, i], alpha=0.2)

    plt.title("Average Empirical Distribution Over Time with Std Deviation")
    plt.xlabel("Time")
    plt.ylabel("Proportion of Particles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

simulate_multiple_trajectories_and_plot(control_net)