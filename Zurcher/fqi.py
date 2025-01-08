def fitted_q_iteration(config):
    """
    Soft Fitted Q-Iteration (Soft FQI) implementation:
      - Action-dependent rewards: 
          - For action 0 (maintenance): r = theta[0] * mileage.
          - For action 1 (replacement): r = theta[1] (negative cost).
      - Bellman target uses softmax: logsumexp(Q(s', a')) instead of max Q(s', a').

    Args:
        config: Dictionary containing FQI parameters.

    Returns:
        q_model: Trained Q-function approximator (MLP).
    """
    # ============= 0. Random seed / directories =============
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    if not os.path.exists('figs/fqi'):
        os.makedirs('figs/fqi', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    if not os.path.exists('logs'):
        os.makedirs('logs', exist_ok=True)

    # ============= 1. Prepare dataset (train + test) =============
    dataset_config = {
        'H': config['H'],
        'num_trajs': config['num_trajs'],
        'maxMileage': config.get('maxMileage', 9999),
        'theta': config['theta'],  # Current theta passed from AVI outer loop
        'beta': config['beta'],
        'numTypes': config.get('numTypes', 1),
        'rollin_type': 'expert',
        'store_gpu': True,
        'shuffle': config['shuffle'],
        'env': config['env'],
        'num_dummies': config['num_dummies'],
        'dummy_dim': config['dummy_dim']
    }

    path_train = build_data_filename(dataset_config, mode='train')
    train_dataset = Dataset(path_train, dataset_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )

    # ============= 2. Create Q-MLP model =============
    model_config = {
        'hidden_sizes': [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], 
    }
    states_dim = config['num_dummies'] + 1  # Mileage + dummies
    actions_dim = 2  # Maintenance (0), Replacement (1)
    q_model = MLP(states_dim, actions_dim, **model_config).to(device)

    optimizer = torch.optim.AdamW(q_model.parameters(), lr=config['lr'], weight_decay=1e-4)
    MSE_loss_fn = nn.MSELoss(reduction='mean')

    # ============= 3. FQI Main Loop =============
    fqi_iterations = config['fqi_num_iterations']
    theta = torch.tensor(config['theta'], device=device)

    for iteration in range(fqi_iterations):
        print(f"\n=== Soft FQI Iteration {iteration+1}/{fqi_iterations} ===")

        #################### 3.a. Build targets: y_t = r_t + beta * logsumexp(Q(s', a')) ####################
        all_states, all_actions, all_targets = [], [], []

        q_model.eval()
        with torch.no_grad():
            for batch in train_loader:
                states = batch['states']  # (batch_size, H, state_dim)
                actions = batch['actions']  # (batch_size, H)
                next_states = batch['next_states']  # (batch_size, H, state_dim)

                # ============= Reward structure =============
                mileage = states[:, :, 0]  # (batch_size, H)
                chosen_r_values = torch.where(
                    actions == 0,  # Maintenance
                    theta[0] * mileage,  # Cost for maintenance
                    theta[1] * torch.ones_like(mileage)  # Cost for replacement
                )  # Shape: (batch_size, H)

                # ============= Softmax over Q(s', a') =============
                # Flatten next_states for batch processing
                next_states_flat = next_states.reshape(-1, next_states.shape[-1])  # (batch_size*H, state_dim)
                q_next = q_model(next_states_flat)  # (batch_size*H, 2)
                logsumexp_q_next = torch.logsumexp(q_next, dim=1)  # (batch_size*H,)
                logsumexp_q_next_2d = logsumexp_q_next.reshape(next_states.shape[:2])  # (batch_size, H)

                # ============= Bellman Target =============
                bellman_target = chosen_r_values + config['beta'] * logsumexp_q_next_2d

                # ============= Prepare (s, a, target) for fitting =============
                states_flat = states.reshape(-1, states.shape[-1])  # (batch_size*H, state_dim)
                actions_flat = actions.reshape(-1)  # (batch_size*H)
                targets_flat = bellman_target.reshape(-1)  # (batch_size*H)

                all_states.append(states_flat.cpu())
                all_actions.append(actions_flat.cpu())
                all_targets.append(targets_flat.cpu())

        # Combine all (s, a, target)
        all_states = torch.cat(all_states, dim=0).float().to(device)  # (num_samples, state_dim)
        all_actions = torch.cat(all_actions, dim=0).long().to(device)  # (num_samples,)
        all_targets = torch.cat(all_targets, dim=0).float().to(device)  # (num_samples,)

        #################### 3.b. Re-Fit Q-Model ####################
        q_model.train()
        dataset_size = all_states.size(0)
        indices = torch.randperm(dataset_size, device=device)

        for epoch in range(config['fqi_epochs_per_iteration']):
            total_loss = 0.0
            for start_idx in range(0, dataset_size, config['batch_size']):
                end_idx = min(start_idx + config['batch_size'], dataset_size)
                batch_idx = indices[start_idx:end_idx]

                s_batch = all_states[batch_idx]  # (batch_size, state_dim)
                a_batch = all_actions[batch_idx]  # (batch_size,)
                y_batch = all_targets[batch_idx]  # (batch_size,)

                # Predict Q(s,a)
                pred_q_values = q_model(s_batch)  # (batch_size, num_actions)
                pred_q_values_chosen = pred_q_values[torch.arange(pred_q_values.size(0)), a_batch]

                # Compute loss
                loss = MSE_loss_fn(pred_q_values_chosen, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"   [Soft FQI Iteration {iteration+1}, Epoch {epoch+1}] Loss: {total_loss:.4f}")

    return q_model
