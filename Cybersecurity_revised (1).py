import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Device and Parameters
# ==========================
device = torch.device("cpu")

states = ['DI', 'DS', 'UI', 'US']
state_index = {s: i for i, s in enumerate(states)}
num_states = len(states)

T = 10.0
N_sim = 24
k_D, k_I = 0.3, 0.5
vH = 0.2
q_inf_D = 0.4
q_inf_U = 0.3
q_rec_D = 0.1
q_rec_U = 0.65
beta_DD = 0.4
beta_UD = 0.4
beta_UU = 0.3
beta_DU = 0.3

num_MC_samples = 5
num_picard_iterations = 5
train_steps = 50

# ==========================
# Neural Network
# ==========================
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

# ==========================
# Helper Functions
# ==========================
def one_hot_state(x):
    vec = torch.zeros(num_states, device=device)
    vec[state_index[x]] = 1.0
    return vec

def gumbel_softmax_sample(logits, tau=1.0):
    noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.nn.functional.softmax((logits + noise) / tau, dim=-1)
 
#initial distribution: uniform distribution
def sample_initial_state():
    return np.random.choice(states)

def compute_empirical_distribution(X):
    counts = torch.zeros(num_states, device=device)
    for x in X:
        counts[state_index[x]] += 1
    return counts / len(X)

def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32, device=device)

def get_jump_rate(x, epsilon, mu):
    epsilon = ensure_tensor(epsilon)

    if x == 'DS':
        inf_rate = vH * q_inf_D + beta_DD * (mu[state_index['DI']] + beta_UD * mu[state_index['UI']])
        inf_rate = ensure_tensor(inf_rate)
        rate = inf_rate + epsilon
        next_states = ['DI', 'US']
        probs = torch.stack([inf_rate, epsilon])
    elif x == 'US':
        inf_rate = vH * q_inf_U + beta_DU * (mu[state_index['DI']] + beta_UU * mu[state_index['UI']])
        inf_rate = ensure_tensor(inf_rate)
        rate = inf_rate + epsilon
        next_states = ['UI', 'DS']
        probs = torch.stack([inf_rate, epsilon])
    elif x == 'DI':
        rate = ensure_tensor(q_rec_D) + epsilon
        probs = torch.stack([ensure_tensor(q_rec_D), epsilon])
        next_states = ['DS', 'UI']
    elif x == 'UI':
        rate = ensure_tensor(q_rec_U) + epsilon
        probs = torch.stack([ensure_tensor(q_rec_U), epsilon])
        next_states = ['US', 'DI']
    else:
        rate = torch.tensor(0.0, device=device)
        probs = torch.zeros(0, device=device)
        next_states = []
    return rate, probs, next_states

# running cost(with quadratic cost on the jump rate)
def tagged_running_cost(state, epsilon):
    is_D = 1.0 if state in ['DI', 'DS'] else 0.0
    is_I = 1.0 if state in ['DI', 'UI'] else 0.0
    return 0.5 * epsilon ** 2 + k_D * is_D + k_I * is_I

# ==========================
# Tagged & Untagged players simulation
# ==========================
def simulate_joint_dynamics_two_clocks_with_cost(tagged_net, untagged_net, N_sim, T):
    X = [sample_initial_state() for _ in range(N_sim + 1)]
    trajs = [[] for _ in range(N_sim + 1)]
    epsilons_all = [[] for _ in range(N_sim + 1)]
    t = torch.tensor(0.0, device=device)
    cost_tagged = torch.zeros(1, device=device)

    def compute_rates():
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
                epsilon = tagged_net(t_tensor, x_onehot.unsqueeze(0), mu.unsqueeze(0)).squeeze()
                mu_i = mu
            else:
                s_i = x_i
                s_0 = X[0]
                e_si = one_hot_state(s_i)
                e_s0 = one_hot_state(s_0)
                mu_i = mu - (1 / (N_sim + 1)) * e_si + (1 / (N_sim + 1)) * e_s0
                with torch.no_grad():
                    epsilon = untagged_net(t_tensor, x_onehot.unsqueeze(0), mu_i.unsqueeze(0)).squeeze()

            epsilons_all[i].append(epsilon)
            rate, probs, next_states = get_jump_rate(x_i, epsilon, mu_i)
            jump_rates.append(rate)
            rate_info.append((probs, next_states))
        return mu, jump_rates, rate_info

    mu, jump_rates, rate_info = compute_rates()
    lambda_tagged = jump_rates[0]
    lambda_un = sum(jump_rates[1:])

    tau_tagged = -torch.log(torch.rand(1, device=device)) / lambda_tagged
    tau_un = -torch.log(torch.rand(1, device=device)) / lambda_un

    time_snapshots = [(0.0, compute_empirical_distribution(X[1:]))]

    while t < T:
        if tau_un < tau_tagged:
            interval = tau_un
            cost_tagged += interval * tagged_running_cost(X[0], epsilons_all[0][-1])
            t = t+ interval
            if t > T:
                break

            un_rates = torch.stack(jump_rates[1:])
            who_jump_un = torch.multinomial(un_rates / un_rates.sum(), 1).item() + 1
            probs, next_states = rate_info[who_jump_un]
            jump_index = torch.multinomial(probs / probs.sum(), 1).item()
            X[who_jump_un] = next_states[jump_index]
            trajs[who_jump_un].append((t.clone(), X[who_jump_un]))

            tau_tagged = tau_tagged-interval
            mu, jump_rates, rate_info = compute_rates()
            lambda_tagged = jump_rates[0]
            lambda_un = sum(jump_rates[1:])
            tau_un =tau_un -torch.log(torch.rand(1, device=device)) / lambda_un

        else:
            interval = tau_tagged
            cost_tagged =cost_tagged+ interval * tagged_running_cost(X[0], epsilons_all[0][-1])
            t = t+ interval
            if t > T:
                break

            probs, next_states = rate_info[0]
            jump_index = torch.multinomial(probs / probs.sum(), 1).item()
            X[0] = next_states[jump_index]
            trajs[0].append((t.clone(), X[0]))

            mu, jump_rates, rate_info = compute_rates()
            lambda_tagged = jump_rates[0]
            lambda_un = sum(jump_rates[1:])
            tau_tagged = -torch.log(torch.rand(1, device=device)) / lambda_tagged
            tau_un = -torch.log(torch.rand(1, device=device)) / lambda_un

        time_snapshots.append((t.item(), compute_empirical_distribution(X[1:])))

    return trajs, epsilons_all, cost_tagged, time_snapshots


untagged_net = ControlNet().to(device)
for p in untagged_net.parameters():
    p.requires_grad = False

for n in range(num_picard_iterations):
    print(f"\nPicard Iteration {n}")
    tagged_net = ControlNet().to(device)
    tagged_net.load_state_dict(untagged_net.state_dict())
    optimizer = optim.Adam(tagged_net.parameters(), lr=1e-2)

    for step in range(train_steps):
        costs = []
        for _ in range(num_MC_samples):
            _, _, cost, _ = simulate_joint_dynamics_two_clocks_with_cost(
                tagged_net, untagged_net, N_sim, T
            )
            costs.append(cost)
        avg_cost = torch.stack(costs).mean()

        if step % 10 == 0:
            print(f"  Step {step}: Tagged cost = {avg_cost.item():.4f}")
        optimizer.zero_grad()
        avg_cost.backward()
        optimizer.step()

    untagged_net.load_state_dict(tagged_net.state_dict())



# Plotting function
def simulate_multiple_trajectories_and_plot(control_net, T=10.0, N_sim=24, num_trajectories=10):
    time_grid = torch.linspace(0, T, steps=100)
    all_trajectories = torch.zeros(num_trajectories, len(time_grid), num_states)

    for traj_idx in range(num_trajectories):
        #Different Initial distribution
        #X = ['DS'] * (N_sim // 4) + ['US'] * (N_sim // 4)+['DI'] * (N_sim // 4) + ['UI'] * (N_sim // 4)
        #X = ['DS'] * (N_sim // 2) + ['US'] * (N_sim // 2)
        X = ['DI']* (N_sim // 1)
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
    plt.savefig("uniform.pdf")
    plt.show()

simulate_multiple_trajectories_and_plot(tagged_net)