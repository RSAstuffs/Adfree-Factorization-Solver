#!/usr/bin/env python3
"""
Policy Network for RSA Factorization

A neural network that learns which bits to flip when trying to factor N = p × q.
Trained on small semiprimes (32-64 bit) using reinforcement learning.

Architecture:
- Input: Current state (p_bits, q_bits, N_bits, diff_bits, metadata)
- Output: Probability distribution over which bit to flip

Training:
- Uses REINFORCE with baseline (policy gradient)
- Reward: Reduction in |p*q - N|
- Curriculum: Start with small numbers, increase difficulty
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from collections import deque
import random
import math

# Try to import torch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[PolicyNetwork] PyTorch not available, using numpy-only implementation")


class NumpyPolicyNetwork:
    """
    Pure numpy implementation of policy network for environments without PyTorch.
    Uses simple linear layers with ReLU activation.
    """
    
    def __init__(self, n_bits: int = 32, hidden_dim: int = 128):
        self.n_bits = n_bits
        self.hidden_dim = hidden_dim
        
        # Input: p_bits + q_bits + N_bits + diff_sign + progress = 3*n_bits + 2
        self.input_dim = 3 * n_bits + 2
        # Output: probability for each source qubit (n_bits for p + n_bits for q)
        self.output_dim = 2 * n_bits
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(self.input_dim, hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = np.random.randn(hidden_dim, self.output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(self.output_dim)
        
        # Value head for baseline
        self.Wv = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.bv = np.zeros(1)
        
        # Learning rate
        self.lr = 0.001
        
        # Experience buffer for training
        self.experiences = []
        
        # Training stats
        self.episode_rewards = []
        self.loss_history = []
        
    def _encode_state(self, p: int, q: int, N: int, step: int, total_steps: int) -> np.ndarray:
        """Encode the current state as a feature vector."""
        features = []
        
        # p bits (normalized to 0/1)
        for i in range(self.n_bits):
            features.append(float((p >> i) & 1))
        
        # q bits
        for i in range(self.n_bits):
            features.append(float((q >> i) & 1))
        
        # N bits
        for i in range(self.n_bits):
            features.append(float((N >> i) & 1))
        
        # Difference sign (-1, 0, or 1)
        diff = p * q - N
        if diff > 0:
            features.append(1.0)
        elif diff < 0:
            features.append(-1.0)
        else:
            features.append(0.0)
        
        # Progress (0 to 1)
        features.append(step / max(total_steps, 1))
        
        return np.array(features, dtype=np.float32)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable softmax
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass: returns (action_probs, value)."""
        # Layer 1
        h1 = self._relu(state @ self.W1 + self.b1)
        
        # Layer 2
        h2 = self._relu(h1 @ self.W2 + self.b2)
        
        # Policy head (softmax over actions)
        logits = h2 @ self.W3 + self.b3
        probs = self._softmax(logits)
        
        # Value head
        value = float(h2 @ self.Wv + self.bv)
        
        return probs, value
    
    def select_action(self, p: int, q: int, N: int, step: int, total_steps: int,
                      temperature: float = 1.0) -> Tuple[int, float]:
        """
        Select which bit to flip based on current state.
        
        Returns: (bit_index, log_probability)
        
        bit_index: 0 to n_bits-1 = p bits, n_bits to 2*n_bits-1 = q bits
        """
        state = self._encode_state(p, q, N, step, total_steps)
        probs, value = self.forward(state)
        
        # Apply temperature
        if temperature != 1.0:
            logits = np.log(probs + 1e-10) / temperature
            probs = self._softmax(logits)
        
        # Sample action
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        
        return action, log_prob, value
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                         log_prob: float, value: float):
        """Store experience for training."""
        self.experiences.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value
        })
    
    def train_on_batch(self, gamma: float = 0.99) -> float:
        """Train on collected experiences using REINFORCE with baseline."""
        if len(self.experiences) < 2:
            return 0.0
        
        # Calculate returns (discounted cumulative rewards)
        returns = []
        R = 0
        for exp in reversed(self.experiences):
            R = exp['reward'] + gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy gradient
        policy_loss = 0.0
        value_loss = 0.0
        
        for i, exp in enumerate(self.experiences):
            advantage = returns[i] - exp['value']
            policy_loss -= exp['log_prob'] * advantage
            value_loss += (returns[i] - exp['value']) ** 2
        
        # Simple gradient descent (approximate - real implementation would use backprop)
        # This is a simplified update that adjusts weights based on which actions worked
        for i, exp in enumerate(self.experiences):
            state = exp['state']
            action = exp['action']
            advantage = returns[i] - exp['value']
            
            # Forward pass to get intermediate activations
            h1 = self._relu(state @ self.W1 + self.b1)
            h2 = self._relu(h1 @ self.W2 + self.b2)
            logits = h2 @ self.W3 + self.b3
            probs = self._softmax(logits)
            
            # Policy gradient for output layer
            grad = probs.copy()
            grad[action] -= 1.0  # d_softmax_cross_entropy
            grad *= -advantage  # Scale by advantage
            
            # Update W3, b3
            self.W3 -= self.lr * np.outer(h2, grad)
            self.b3 -= self.lr * grad
            
            # Value head update
            value = float(h2 @ self.Wv + self.bv)
            value_grad = 2 * (value - returns[i])
            self.Wv -= self.lr * 0.5 * value_grad * h2.reshape(-1, 1)
            self.bv -= self.lr * 0.5 * value_grad
        
        # Clear experiences
        total_reward = sum(exp['reward'] for exp in self.experiences)
        self.episode_rewards.append(total_reward)
        self.experiences = []
        
        loss = float(policy_loss + 0.5 * value_loss)
        self.loss_history.append(loss)
        
        return loss
    
    def save(self, filepath: str):
        """Save model weights."""
        np.savez(filepath, 
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 Wv=self.Wv, bv=self.bv,
                 n_bits=self.n_bits,
                 hidden_dim=self.hidden_dim)
        print(f"[PolicyNetwork] Saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        self.Wv = data['Wv']
        self.bv = data['bv']
        self.n_bits = int(data['n_bits'])
        self.hidden_dim = int(data['hidden_dim'])
        print(f"[PolicyNetwork] Loaded from {filepath}")


if HAS_TORCH:
    class TorchPolicyNetwork(nn.Module):
        """
        PyTorch implementation of policy network with proper backpropagation.
        """
        
        def __init__(self, n_bits: int = 32, hidden_dim: int = 256):
            super().__init__()
            self.n_bits = n_bits
            self.hidden_dim = hidden_dim
            
            # Input: p_bits + q_bits + N_bits + diff_magnitude + diff_sign + progress
            self.input_dim = 3 * n_bits + 3
            # Output: probability for each bit position (p and q)
            self.output_dim = 2 * n_bits
            
            # Shared layers
            self.shared = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            # Policy head
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, self.output_dim),
            )
            
            # Value head
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            
            # Optimizer
            self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
            
            # Experience buffer
            self.experiences = []
            self.episode_rewards = []
            self.loss_history = []
        
        def _encode_state(self, p: int, q: int, N: int, step: int, total_steps: int) -> torch.Tensor:
            """Encode state as tensor."""
            features = []
            
            # p bits
            for i in range(self.n_bits):
                features.append(float((p >> i) & 1))
            
            # q bits
            for i in range(self.n_bits):
                features.append(float((q >> i) & 1))
            
            # N bits
            for i in range(self.n_bits):
                features.append(float((N >> i) & 1))
            
            # Difference magnitude (log scale)
            diff = abs(p * q - N)
            if diff > 0:
                log_diff = math.log2(diff + 1) / self.n_bits  # Normalize by bit length
            else:
                log_diff = 0.0
            features.append(log_diff)
            
            # Difference sign
            if p * q > N:
                features.append(1.0)
            elif p * q < N:
                features.append(-1.0)
            else:
                features.append(0.0)
            
            # Progress
            features.append(step / max(total_steps, 1))
            
            return torch.tensor(features, dtype=torch.float32)
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass."""
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            shared_out = self.shared(state)
            
            # Policy: log probabilities
            logits = self.policy_head(shared_out)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Value
            value = self.value_head(shared_out)
            
            return log_probs, value
        
        def select_action(self, p: int, q: int, N: int, step: int, total_steps: int,
                          temperature: float = 1.0) -> Tuple[int, torch.Tensor, torch.Tensor]:
            """Select action using the policy."""
            state = self._encode_state(p, q, N, step, total_steps)
            
            with torch.no_grad():
                log_probs, value = self.forward(state)
                
                # Apply temperature
                if temperature != 1.0:
                    log_probs = log_probs / temperature
                    log_probs = F.log_softmax(log_probs, dim=-1)
                
                probs = torch.exp(log_probs)
                
                # Sample action
                action = torch.multinomial(probs.squeeze(), 1).item()
                
            return action, log_probs[0, action], value.squeeze()
        
        def store_experience(self, state: torch.Tensor, action: int, reward: float,
                             log_prob: torch.Tensor, value: torch.Tensor):
            """Store experience."""
            self.experiences.append({
                'state': state,
                'action': action,
                'reward': reward,
                'log_prob': log_prob,
                'value': value.detach()
            })
        
        def train_on_batch(self, gamma: float = 0.99, entropy_coef: float = 0.01) -> float:
            """Train using PPO-style update."""
            if len(self.experiences) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            R = 0
            for exp in reversed(self.experiences):
                R = exp['reward'] + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Prepare batch
            states = torch.stack([exp['state'] if isinstance(exp['state'], torch.Tensor) 
                                  else self._encode_state_from_dict(exp['state'])
                                  for exp in self.experiences])
            actions = torch.tensor([exp['action'] for exp in self.experiences])
            old_log_probs = torch.stack([exp['log_prob'] for exp in self.experiences])
            old_values = torch.stack([exp['value'] for exp in self.experiences])
            
            # Forward pass
            log_probs, values = self.forward(states)
            
            # Get log probs for taken actions
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Advantages
            advantages = returns - old_values
            
            # Policy loss (with entropy bonus)
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            policy_loss = -(action_log_probs * advantages.detach()).mean()
            policy_loss = policy_loss - entropy_coef * entropy
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            # Record stats
            total_reward = sum(exp['reward'] for exp in self.experiences)
            self.episode_rewards.append(total_reward)
            self.loss_history.append(loss.item())
            
            # Clear experiences
            self.experiences = []
            
            return loss.item()
        
        def _encode_state_from_dict(self, state_dict):
            """Helper to encode state from stored dict."""
            return self._encode_state(
                state_dict['p'], state_dict['q'], state_dict['N'],
                state_dict['step'], state_dict['total_steps']
            )
        
        def save(self, filepath: str):
            """Save model."""
            torch.save({
                'state_dict': self.state_dict(),
                'n_bits': self.n_bits,
                'hidden_dim': self.hidden_dim,
                'episode_rewards': self.episode_rewards,
                'loss_history': self.loss_history,
            }, filepath)
            print(f"[PolicyNetwork] Saved to {filepath}")
        
        def load(self, filepath: str):
            """Load model."""
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint['state_dict'])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.loss_history = checkpoint.get('loss_history', [])
            print(f"[PolicyNetwork] Loaded from {filepath}")


# Use appropriate implementation
PolicyNetwork = TorchPolicyNetwork if HAS_TORCH else NumpyPolicyNetwork


class PolicyNetworkTrainer:
    """
    Trains the policy network on factorization problems.
    Uses curriculum learning: start small, increase difficulty.
    """
    
    def __init__(self, n_bits: int = 32, hidden_dim: int = 256):
        self.n_bits = n_bits
        self.policy = PolicyNetwork(n_bits, hidden_dim)
        
        # Curriculum parameters
        self.current_difficulty = 8  # Start with 8-bit factors
        self.max_difficulty = n_bits // 2  # Max factor size
        self.success_threshold = 0.7  # Success rate to increase difficulty
        self.recent_successes = deque(maxlen=100)
        
        # Training stats
        self.total_episodes = 0
        self.total_successes = 0
    
    def generate_semiprime(self, bit_size: int) -> Tuple[int, int, int]:
        """Generate a random semiprime N = p * q with factors of given bit size."""
        # Generate two random primes of approximately the given bit size
        min_val = 2 ** (bit_size - 1)
        max_val = 2 ** bit_size - 1
        
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        def random_prime(min_v, max_v):
            while True:
                candidate = random.randint(min_v, max_v)
                if candidate % 2 == 0:
                    candidate += 1
                if is_prime(candidate):
                    return candidate
        
        p = random_prime(min_val, max_val)
        q = random_prime(min_val, max_val)
        
        # Ensure p != q for RSA-like semiprimes
        while q == p:
            q = random_prime(min_val, max_val)
        
        N = p * q
        return N, min(p, q), max(p, q)
    
    def run_episode(self, N: int, true_p: int, true_q: int, 
                    max_steps: int = 100) -> Tuple[bool, float]:
        """
        Run one factorization episode using greedy search with policy guidance.
        
        Returns: (success, total_reward)
        """
        # Initialize p and q near sqrt(N)
        sqrt_n = int(N ** 0.5)
        
        # Multiple initialization strategies
        init_strategy = random.choice(['sqrt', 'random', 'biased'])
        if init_strategy == 'sqrt':
            p = sqrt_n
            q = sqrt_n
        elif init_strategy == 'random':
            p = random.randint(3, int(sqrt_n * 1.5)) | 1
            q = random.randint(3, int(sqrt_n * 1.5)) | 1
        else:  # biased toward true values for curriculum
            noise = self.current_difficulty // 2
            p = true_p + random.randint(-noise, noise)
            q = true_q + random.randint(-noise, noise)
        
        # Ensure p, q are positive and odd (for odd N)
        p = max(3, p | 1)
        q = max(3, q | 1)
        
        total_reward = 0.0
        best_diff = abs(p * q - N)
        best_p, best_q = p, q
        
        for step in range(max_steps):
            # Check if solved
            if p * q == N:
                # Big reward for solving!
                reward = 100.0
                state = self.policy._encode_state(p, q, N, step, max_steps)
                self.policy.store_experience(state, 0, reward, 0.0, 0.0)
                total_reward += reward
                return True, total_reward
            
            # Get state
            state = self.policy._encode_state(p, q, N, step, max_steps)
            
            # 30% greedy (try all actions, pick best), 70% policy-guided
            old_diff = abs(p * q - N)
            
            if random.random() < 0.3:
                # Greedy: evaluate all possible single-bit flips
                best_action = None
                best_new_diff = old_diff
                
                for action in range(2 * self.n_bits):
                    test_p, test_q = p, q
                    if action < self.n_bits:
                        test_p ^= (1 << action)
                    else:
                        test_q ^= (1 << (action - self.n_bits))
                    
                    if test_p > 0 and test_q > 0:
                        test_diff = abs(test_p * test_q - N)
                        if test_diff < best_new_diff:
                            best_new_diff = test_diff
                            best_action = action
                
                if best_action is not None:
                    action = best_action
                    log_prob = 0.0
                    value = 0.0
                else:
                    # No improvement possible, use policy
                    temperature = max(0.1, 1.0 - step / max_steps)
                    action, log_prob, value = self.policy.select_action(p, q, N, step, max_steps, temperature)
            else:
                # Policy-guided selection
                temperature = max(0.1, 1.0 - step / max_steps)
                action, log_prob, value = self.policy.select_action(p, q, N, step, max_steps, temperature)
            
            # Apply action (flip a bit)
            if action < self.n_bits:
                # Flip bit in p
                bit_pos = action
                p ^= (1 << bit_pos)
                # Keep p positive and odd
                if p <= 0:
                    p ^= (1 << bit_pos)  # Undo
                if N % 2 == 1 and p % 2 == 0:
                    p |= 1  # Force odd
            else:
                # Flip bit in q
                bit_pos = action - self.n_bits
                q ^= (1 << bit_pos)
                # Keep q positive and odd
                if q <= 0:
                    q ^= (1 << bit_pos)  # Undo
                if N % 2 == 1 and q % 2 == 0:
                    q |= 1  # Force odd
            
            # Calculate reward
            new_diff = abs(p * q - N)
            
            if new_diff == 0:
                reward = 100.0  # Solved!
            elif new_diff < old_diff:
                # Reward proportional to improvement (log scale for big numbers)
                if old_diff > 0:
                    improvement = (old_diff - new_diff) / old_diff
                    reward = improvement * 20.0
                else:
                    reward = 10.0
                if new_diff < best_diff:
                    best_diff = new_diff
                    best_p, best_q = p, q
                    reward += 2.0  # Bonus for new best
            elif new_diff > old_diff:
                # Penalty for getting worse
                if old_diff > 0:
                    worsening = (new_diff - old_diff) / old_diff
                    reward = -min(worsening * 10.0, 5.0)  # Cap penalty
                else:
                    reward = -1.0
            else:
                reward = -0.05  # Small penalty for no change
            
            # Store experience
            self.policy.store_experience(state, action, reward, log_prob, value)
            total_reward += reward
            
            # Check if solved after action
            if p * q == N:
                return True, total_reward
        
        return False, total_reward
    
    def train(self, num_episodes: int = 1000, episode_length: int = 100,
              print_interval: int = 50) -> Dict:
        """
        Train the policy network.
        
        Args:
            num_episodes: Total training episodes
            episode_length: Max steps per episode
            print_interval: How often to print stats
        
        Returns: Training statistics
        """
        print(f"\n{'='*60}")
        print(f"POLICY NETWORK TRAINING")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Initial difficulty: {self.current_difficulty}-bit factors")
        print(f"Max difficulty: {self.max_difficulty}-bit factors")
        print(f"Using: {'PyTorch' if HAS_TORCH else 'NumPy'}")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            # Generate problem at current difficulty
            N, true_p, true_q = self.generate_semiprime(self.current_difficulty)
            
            # Run episode
            success, reward = self.run_episode(N, true_p, true_q, episode_length)
            
            # Train on collected experiences
            loss = self.policy.train_on_batch()
            
            # Update stats
            self.total_episodes += 1
            if success:
                self.total_successes += 1
            self.recent_successes.append(1 if success else 0)
            
            # Check for difficulty increase
            if len(self.recent_successes) >= 50:
                success_rate = sum(self.recent_successes) / len(self.recent_successes)
                if success_rate >= self.success_threshold and self.current_difficulty < self.max_difficulty:
                    self.current_difficulty += 1
                    self.recent_successes.clear()
                    print(f"\n[Curriculum] Difficulty increased to {self.current_difficulty}-bit factors!")
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                recent_rate = sum(list(self.recent_successes)[-print_interval:]) / min(print_interval, len(self.recent_successes))
                avg_reward = sum(self.policy.episode_rewards[-print_interval:]) / min(print_interval, len(self.policy.episode_rewards))
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Difficulty: {self.current_difficulty}-bit | "
                      f"Success Rate: {recent_rate:.1%} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Loss: {loss:.4f}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Total successes: {self.total_successes}")
        print(f"Final difficulty: {self.current_difficulty}-bit factors")
        print(f"Overall success rate: {self.total_successes/max(1, self.total_episodes):.1%}")
        
        return {
            'total_episodes': self.total_episodes,
            'total_successes': self.total_successes,
            'final_difficulty': self.current_difficulty,
            'episode_rewards': self.policy.episode_rewards,
            'loss_history': self.policy.loss_history
        }
    
    def save(self, filepath: str):
        """Save trained policy."""
        self.policy.save(filepath)
    
    def load(self, filepath: str):
        """Load trained policy."""
        self.policy.load(filepath)


def test_policy(policy: PolicyNetwork, N: int, max_steps: int = 500) -> Tuple[int, int, bool]:
    """
    Test a trained policy on a factorization problem.
    
    Returns: (p, q, success)
    """
    sqrt_n = int(N ** 0.5)
    p = sqrt_n
    q = sqrt_n
    
    # Ensure odd
    if N % 2 == 1:
        p = p | 1
        q = q | 1
    
    best_diff = abs(p * q - N)
    best_p, best_q = p, q
    
    for step in range(max_steps):
        if p * q == N:
            return p, q, True
        
        # Select action
        temperature = max(0.05, 0.5 - step / max_steps)
        action, _, _ = policy.select_action(p, q, N, step, max_steps, temperature)
        
        n_bits = policy.n_bits
        
        if action < n_bits:
            p ^= (1 << action)
            if p <= 0:
                p ^= (1 << action)
        else:
            q ^= (1 << (action - n_bits))
            if q <= 0:
                q ^= (1 << (action - n_bits))
        
        # Track best
        diff = abs(p * q - N)
        if diff < best_diff:
            best_diff = diff
            best_p, best_q = p, q
    
    return best_p, best_q, best_p * best_q == N


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Policy Network for RSA Factorization')
    parser.add_argument('--train', action='store_true', help='Train the policy network')
    parser.add_argument('--episodes', type=int, default=2000, help='Training episodes')
    parser.add_argument('--bits', type=int, default=32, help='Max bit size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--save', type=str, default='policy_network.pth', help='Save path')
    parser.add_argument('--load', type=str, default=None, help='Load path')
    parser.add_argument('--test', type=int, default=None, help='Test on specific N')
    
    args = parser.parse_args()
    
    if args.train:
        trainer = PolicyNetworkTrainer(n_bits=args.bits, hidden_dim=args.hidden)
        
        if args.load and os.path.exists(args.load):
            trainer.load(args.load)
        
        stats = trainer.train(num_episodes=args.episodes)
        trainer.save(args.save)
        
        print(f"\nTraining stats saved. Final difficulty: {stats['final_difficulty']}-bit")
    
    elif args.test is not None:
        print(f"Testing policy on N = {args.test}")
        
        if args.load:
            policy = PolicyNetwork(n_bits=args.bits, hidden_dim=args.hidden)
            policy.load(args.load)
        else:
            policy = PolicyNetwork(n_bits=args.bits, hidden_dim=args.hidden)
            print("Warning: Using untrained policy")
        
        p, q, success = test_policy(policy, args.test)
        
        print(f"Result: {p} × {q} = {p * q}")
        print(f"Target: {args.test}")
        print(f"Success: {success}")
        if not success:
            print(f"Diff: {abs(p * q - args.test)}")
    
    else:
        print("Use --train to train or --test N to test")
        print("Example: python policy_network.py --train --episodes 5000 --bits 32")
