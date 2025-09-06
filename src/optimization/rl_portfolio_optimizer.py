import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import json
from collections import deque, defaultdict
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import crypto_logger
from config.config import Config

@dataclass
class PortfolioState:
    weights: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    volatilities: np.ndarray
    correlations: np.ndarray
    market_features: np.ndarray
    cash: float
    total_value: float
    timestamp: datetime

@dataclass
class OptimizationAction:
    target_weights: np.ndarray
    rebalance_threshold: float
    confidence: float
    reasoning: str

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    alpha: float
    beta: float
    information_ratio: float
    calmar_ratio: float
    sortino_ratio: float

class DQNNetwork(nn.Module):
    """Deep Q-Network for portfolio optimization."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 256, 128]):
        super(DQNNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        return self.network(state)

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for continuous portfolio optimization."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 256]):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[1] // 2, action_size),
            nn.Softmax(dim=-1)  # Portfolio weights sum to 1
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[1] // 2, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class PortfolioEnvironment(gym.Env):
    """Custom gym environment for portfolio optimization."""
    
    def __init__(self, assets: List[str], lookback_window: int = 252):
        super(PortfolioEnvironment, self).__init__()
        
        self.assets = assets
        self.n_assets = len(assets)
        self.lookback_window = lookback_window
        
        # Action space: portfolio weights (continuous)
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # State space: prices, returns, technical indicators, market features
        state_dim = (
            self.n_assets * 3 +  # prices, returns, volatilities
            self.n_assets * self.n_assets +  # correlation matrix
            10  # market features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Initialize data structures
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.portfolio_value = 100000.0  # Starting with $100k
        self.cash = 0.0
        self.positions = np.zeros(self.n_assets)
        self.transaction_costs = 0.001  # 0.1% transaction cost
        
        # Generate mock historical data
        self._generate_mock_data()
        
        return self._get_observation()
    
    def _generate_mock_data(self):
        """Generate realistic market data for training."""
        
        np.random.seed(None)  # Reset seed for randomness
        n_days = 1000
        
        # Generate correlated returns
        correlation_matrix = np.random.uniform(0.1, 0.7, (self.n_assets, self.n_assets))
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate returns
        mean_returns = np.random.uniform(-0.001, 0.002, self.n_assets)  # Daily returns
        volatilities = np.random.uniform(0.01, 0.05, self.n_assets)  # Daily volatilities
        
        # Generate return series with regime changes
        returns = []
        for i in range(n_days):
            # Add regime change every ~100 days
            if i > 0 and i % 100 == 0:
                volatilities *= np.random.uniform(0.5, 2.0, self.n_assets)
                mean_returns += np.random.normal(0, 0.001, self.n_assets)
            
            daily_returns = np.random.multivariate_normal(
                mean_returns,
                np.outer(volatilities, volatilities) * correlation_matrix
            )
            returns.append(daily_returns)
        
        self.returns_data = np.array(returns)
        
        # Generate price series
        initial_prices = np.random.uniform(10, 1000, self.n_assets)
        self.prices_data = np.zeros((n_days, self.n_assets))
        self.prices_data[0] = initial_prices
        
        for i in range(1, n_days):
            self.prices_data[i] = self.prices_data[i-1] * (1 + self.returns_data[i])
        
        # Calculate rolling statistics
        self.rolling_vol = np.array([
            pd.Series(self.returns_data[:, i]).rolling(30).std().fillna(volatilities[i])
            for i in range(self.n_assets)
        ]).T
        
        # Generate market features
        self.market_features = self._generate_market_features(n_days)
    
    def _generate_market_features(self, n_days: int) -> np.ndarray:
        """Generate additional market features."""
        
        features = np.zeros((n_days, 10))
        
        # VIX-like volatility index
        features[:, 0] = np.random.lognormal(3.0, 0.5, n_days)
        
        # Market momentum
        features[:, 1] = np.random.normal(0, 0.02, n_days)
        
        # Interest rate proxy
        features[:, 2] = np.random.uniform(0.01, 0.05, n_days)
        
        # Dollar strength index
        features[:, 3] = np.random.normal(100, 10, n_days)
        
        # Crypto fear & greed index
        features[:, 4] = np.random.uniform(0, 100, n_days)
        
        # Trading volume indicator
        features[:, 5] = np.random.lognormal(15, 1, n_days)
        
        # Market breadth
        features[:, 6] = np.random.uniform(0.3, 0.9, n_days)
        
        # Sentiment indicator
        features[:, 7] = np.random.normal(0, 1, n_days)
        
        # Macro economic indicator
        features[:, 8] = np.random.normal(0, 0.5, n_days)
        
        # Risk-on/risk-off indicator
        features[:, 9] = np.random.choice([-1, 0, 1], n_days, p=[0.3, 0.4, 0.3])
        
        return features
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        
        if self.current_step >= len(self.prices_data):
            self.current_step = len(self.prices_data) - 1
        
        # Current prices (normalized)
        current_prices = self.prices_data[self.current_step]
        if self.current_step > 0:
            price_changes = current_prices / self.prices_data[self.current_step - 1] - 1
        else:
            price_changes = np.zeros(self.n_assets)
        
        # Recent returns
        start_idx = max(0, self.current_step - 30)
        recent_returns = self.returns_data[start_idx:self.current_step + 1]
        
        if len(recent_returns) > 1:
            mean_returns = np.mean(recent_returns, axis=0)
            vol_returns = np.std(recent_returns, axis=0)
            
            # Correlation matrix
            if len(recent_returns) >= self.n_assets:
                corr_matrix = np.corrcoef(recent_returns.T)
                corr_matrix = np.nan_to_num(corr_matrix)
            else:
                corr_matrix = np.eye(self.n_assets)
        else:
            mean_returns = np.zeros(self.n_assets)
            vol_returns = np.ones(self.n_assets) * 0.02
            corr_matrix = np.eye(self.n_assets)
        
        # Market features
        current_features = self.market_features[self.current_step]
        
        # Portfolio state
        portfolio_weights = self.positions / np.sum(self.positions) if np.sum(self.positions) > 0 else np.zeros(self.n_assets)
        
        # Combine all features
        observation = np.concatenate([
            price_changes,
            mean_returns,
            vol_returns,
            corr_matrix.flatten(),
            current_features,
            portfolio_weights
        ])
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step within the environment."""
        
        # Normalize action to valid portfolio weights
        action = np.clip(action, 0, 1)
        action = action / (np.sum(action) + 1e-8)  # Normalize to sum to 1
        
        # Calculate portfolio rebalancing
        current_value = np.sum(self.positions * self.prices_data[self.current_step])
        target_positions = action * (current_value + self.cash)
        
        # Calculate transaction costs
        position_changes = np.abs(target_positions - self.positions)
        total_transaction_cost = np.sum(position_changes) * self.transaction_costs
        
        # Update positions
        old_positions = self.positions.copy()
        self.positions = target_positions / self.prices_data[self.current_step]
        self.cash -= total_transaction_cost
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(old_positions, total_transaction_cost)
        
        # Check if episode is done
        done = self.current_step >= len(self.prices_data) - 1
        
        # Additional info
        info = {
            'portfolio_value': current_value,
            'cash': self.cash,
            'transaction_cost': total_transaction_cost,
            'weights': action
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, old_positions: np.ndarray, transaction_cost: float) -> float:
        """Calculate reward for the current step."""
        
        if self.current_step == 0:
            return 0.0
        
        # Portfolio return
        old_value = np.sum(old_positions * self.prices_data[self.current_step - 1])
        new_value = np.sum(self.positions * self.prices_data[self.current_step])
        
        if old_value > 0:
            portfolio_return = (new_value - old_value) / old_value
        else:
            portfolio_return = 0.0
        
        # Risk-adjusted reward
        if self.current_step >= 30:
            recent_returns = []
            for i in range(max(1, self.current_step - 29), self.current_step + 1):
                if i < len(self.prices_data):
                    prev_value = np.sum(self.positions * self.prices_data[i - 1]) if i > 0 else old_value
                    curr_value = np.sum(self.positions * self.prices_data[i])
                    if prev_value > 0:
                        recent_returns.append((curr_value - prev_value) / prev_value)
            
            if recent_returns and len(recent_returns) > 1:
                portfolio_vol = np.std(recent_returns)
                if portfolio_vol > 0:
                    sharpe_component = np.mean(recent_returns) / portfolio_vol
                else:
                    sharpe_component = 0
            else:
                sharpe_component = 0
        else:
            sharpe_component = 0
        
        # Transaction cost penalty
        cost_penalty = transaction_cost / (old_value + 1e-8)
        
        # Diversification bonus
        portfolio_weights = self.positions / np.sum(self.positions) if np.sum(self.positions) > 0 else np.zeros(self.n_assets)
        concentration = np.sum(portfolio_weights ** 2)
        diversification_bonus = (1 - concentration) * 0.01
        
        # Combined reward
        reward = (
            portfolio_return * 100 +  # Scale up portfolio return
            sharpe_component * 0.1 +  # Sharpe ratio component
            diversification_bonus -   # Diversification bonus
            cost_penalty * 10         # Transaction cost penalty
        )
        
        return reward

class RLPortfolioOptimizer:
    """Reinforcement Learning Portfolio Optimizer with advanced algorithms."""
    
    def __init__(self, assets: List[str]):
        self.config = Config()
        self.assets = assets
        self.n_assets = len(assets)
        
        # Initialize environment
        self.env = PortfolioEnvironment(assets)
        
        # RL algorithm selection
        self.algorithm = 'PPO'  # Proximal Policy Optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network parameters
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        # Initialize networks
        self._initialize_networks()
        
        # Training parameters
        self.learning_rate = 3e-4
        self.gamma = 0.99  # Discount factor
        self.eps_clip = 0.2  # PPO clipping parameter
        self.k_epochs = 4  # PPO update epochs
        self.buffer_size = 2048
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.buffer_size)
        
        # Performance tracking
        self.training_rewards = []
        self.portfolio_values = []
        self.performance_metrics = []
        
        # Optimization state
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.last_rebalance = datetime.now()
        
    def _initialize_networks(self):
        """Initialize neural networks for RL algorithms."""
        
        if self.algorithm == 'PPO':
            self.policy = ActorCriticNetwork(
                self.state_size, 
                self.action_size,
                [512, 256]
            ).to(self.device)
            
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
            
        elif self.algorithm == 'DQN':
            self.q_network = DQNNetwork(
                self.state_size,
                self.action_size,
                [512, 256, 128]
            ).to(self.device)
            
            self.target_network = DQNNetwork(
                self.state_size,
                self.action_size,
                [512, 256, 128]
            ).to(self.device)
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        crypto_logger.logger.info(f"Initialized {self.algorithm} networks for portfolio optimization")
    
    async def initialize_rl_optimizer(self):
        """Initialize RL portfolio optimizer."""
        crypto_logger.logger.info("Initializing RL portfolio optimizer")
        
        try:
            # Load pre-trained models if available
            await self._load_pretrained_models()
            
            # Initialize market data feeds
            await self._initialize_market_feeds()
            
            # Setup performance monitoring
            await self._setup_performance_monitoring()
            
            crypto_logger.logger.info("✓ RL portfolio optimizer initialized")
            
        except Exception as e:
            crypto_logger.logger.error(f"Error initializing RL optimizer: {e}")
    
    async def _load_pretrained_models(self):
        """Load pre-trained models if available."""
        
        # In a production system, you would load saved model weights here
        # For now, we'll use randomly initialized networks
        
        crypto_logger.logger.info("Using randomly initialized networks (no pre-trained models)")
    
    async def _initialize_market_feeds(self):
        """Initialize real-time market data feeds."""
        
        # Mock market data initialization
        # In production, this would connect to real market data providers
        
        self.market_data = {
            'prices': {},
            'volumes': {},
            'orderbooks': {},
            'last_update': datetime.now()
        }
        
        crypto_logger.logger.info("Initialized mock market data feeds")
    
    async def _setup_performance_monitoring(self):
        """Setup performance monitoring and metrics tracking."""
        
        self.performance_tracker = {
            'daily_returns': deque(maxlen=252),
            'portfolio_values': deque(maxlen=1000),
            'sharpe_ratios': deque(maxlen=100),
            'max_drawdowns': deque(maxlen=100),
            'rebalance_frequency': deque(maxlen=100)
        }
        
        crypto_logger.logger.info("Setup performance monitoring")
    
    async def train_model(self, episodes: int = 1000) -> Dict[str, Any]:
        """Train the RL model on historical data."""
        
        crypto_logger.logger.info(f"Starting RL model training for {episodes} episodes")
        
        training_results = {
            'episodes_completed': 0,
            'average_reward': 0,
            'best_reward': float('-inf'),
            'training_time': 0,
            'convergence_achieved': False
        }
        
        start_time = datetime.now()
        
        for episode in range(episodes):
            episode_reward = await self._train_episode()
            self.training_rewards.append(episode_reward)
            
            # Update best reward
            if episode_reward > training_results['best_reward']:
                training_results['best_reward'] = episode_reward
            
            # Check for convergence
            if episode >= 100:
                recent_avg = np.mean(self.training_rewards[-100:])
                if episode >= 200:
                    prev_avg = np.mean(self.training_rewards[-200:-100])
                    if abs(recent_avg - prev_avg) < 0.01:  # Convergence threshold
                        training_results['convergence_achieved'] = True
                        crypto_logger.logger.info(f"Training converged at episode {episode}")
                        break
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if len(self.training_rewards) >= 100 else np.mean(self.training_rewards)
                crypto_logger.logger.info(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
        
        training_results['episodes_completed'] = episode + 1
        training_results['average_reward'] = np.mean(self.training_rewards) if self.training_rewards else 0
        training_results['training_time'] = (datetime.now() - start_time).total_seconds()
        
        crypto_logger.logger.info(f"Training completed: {training_results['episodes_completed']} episodes, "
                                 f"Average reward: {training_results['average_reward']:.4f}")
        
        return training_results
    
    async def _train_episode(self) -> float:
        """Train a single episode."""
        
        state = self.env.reset()
        episode_reward = 0
        episode_memory = []
        
        done = False
        while not done:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                if self.algorithm == 'PPO':
                    action_probs, state_value = self.policy(state_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    
                    # For continuous actions, use the probabilities as weights
                    action = action_probs.cpu().numpy().flatten()
                else:  # DQN
                    q_values = self.q_network(state_tensor)
                    action = torch.softmax(q_values, dim=1).cpu().numpy().flatten()
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            episode_memory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'state_value': state_value.item() if self.algorithm == 'PPO' else 0
            })
            
            state = next_state
            episode_reward += reward
        
        # Update policy with episode data
        await self._update_policy(episode_memory)
        
        return episode_reward
    
    async def _update_policy(self, episode_memory: List[Dict]):
        """Update policy using collected experience."""
        
        if not episode_memory:
            return
        
        if self.algorithm == 'PPO':
            await self._update_ppo(episode_memory)
        elif self.algorithm == 'DQN':
            await self._update_dqn(episode_memory)
    
    async def _update_ppo(self, episode_memory: List[Dict]):
        """Update PPO policy."""
        
        # Calculate discounted rewards
        rewards = [exp['reward'] for exp in episode_memory]
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([exp['state'] for exp in episode_memory])).to(self.device)
        actions = torch.FloatTensor(np.array([exp['action'] for exp in episode_memory])).to(self.device)
        old_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy output
            action_probs, state_values = self.policy(states)
            state_values = state_values.squeeze()
            
            # Calculate advantages
            advantages = old_rewards - state_values.detach()
            
            # Calculate policy loss (simplified for portfolio weights)
            action_log_probs = torch.log(action_probs + 1e-8)
            selected_log_probs = torch.sum(action_log_probs * actions, dim=1)
            
            # PPO clipped objective
            policy_loss = -torch.mean(selected_log_probs * advantages)
            
            # Value loss
            value_loss = F.mse_loss(state_values, old_rewards)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
    
    async def _update_dqn(self, episode_memory: List[Dict]):
        """Update DQN network."""
        
        # Add to replay buffer
        self.memory.extend(episode_memory)
        
        if len(self.memory) < 32:  # Minimum batch size
            return
        
        # Sample batch
        batch_size = min(32, len(self.memory))
        batch = np.random.choice(self.memory, batch_size, replace=False)
        
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([np.argmax(exp['action']) for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if len(self.training_rewards) % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _calculate_discounted_rewards(self, rewards: List[float]) -> np.ndarray:
        """Calculate discounted rewards."""
        
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_reward = 0
        
        for i in reversed(range(len(rewards))):
            running_reward = rewards[i] + self.gamma * running_reward
            discounted_rewards[i] = running_reward
        
        return discounted_rewards
    
    async def optimize_portfolio(self, current_portfolio: Dict[str, float], 
                               market_data: Dict[str, Any]) -> OptimizationAction:
        """Generate optimal portfolio allocation using trained RL model."""
        
        # Prepare state observation
        state = await self._prepare_state_observation(current_portfolio, market_data)
        
        # Get action from trained policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.algorithm == 'PPO':
                action_probs, state_value = self.policy(state_tensor)
                optimal_weights = action_probs.cpu().numpy().flatten()
                confidence = float(torch.max(action_probs))
            else:  # DQN
                q_values = self.q_network(state_tensor)
                action_probs = torch.softmax(q_values, dim=1)
                optimal_weights = action_probs.cpu().numpy().flatten()
                confidence = float(torch.max(action_probs))
        
        # Normalize weights
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Calculate rebalance threshold based on current vs target weights
        current_weights = np.array([current_portfolio.get(asset, 0) for asset in self.assets])
        current_weights = current_weights / np.sum(current_weights) if np.sum(current_weights) > 0 else current_weights
        
        weight_diff = np.sum(np.abs(optimal_weights - current_weights))
        rebalance_threshold = max(0.05, weight_diff * 0.5)  # Dynamic threshold
        
        # Generate reasoning
        reasoning = self._generate_optimization_reasoning(optimal_weights, current_weights, market_data)
        
        return OptimizationAction(
            target_weights=optimal_weights,
            rebalance_threshold=rebalance_threshold,
            confidence=confidence,
            reasoning=reasoning
        )
    
    async def _prepare_state_observation(self, current_portfolio: Dict[str, float], 
                                       market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare state observation for the RL model."""
        
        # Mock state preparation (in production, use real market data)
        n_features = self.state_size
        state = np.random.normal(0, 0.1, n_features).astype(np.float32)
        
        # Include current portfolio weights
        current_weights = np.array([current_portfolio.get(asset, 0) for asset in self.assets])
        if np.sum(current_weights) > 0:
            current_weights = current_weights / np.sum(current_weights)
        
        # Replace last n_assets elements with current weights
        state[-self.n_assets:] = current_weights
        
        return state
    
    def _generate_optimization_reasoning(self, optimal_weights: np.ndarray, 
                                       current_weights: np.ndarray, 
                                       market_data: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for optimization decisions."""
        
        reasoning_parts = []
        
        # Identify major changes
        weight_changes = optimal_weights - current_weights
        
        for i, asset in enumerate(self.assets):
            change = weight_changes[i]
            if abs(change) > 0.05:  # 5% change threshold
                if change > 0:
                    reasoning_parts.append(f"Increase {asset} allocation by {change:.1%} due to positive momentum indicators")
                else:
                    reasoning_parts.append(f"Decrease {asset} allocation by {abs(change):.1%} due to risk management")
        
        # Market regime analysis
        if 'volatility' in market_data and market_data['volatility'] > 0.3:
            reasoning_parts.append("Adopting defensive positioning due to high market volatility")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining balanced allocation based on current market conditions")
        
        return "; ".join(reasoning_parts)
    
    async def backtest_strategy(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Backtest the RL strategy on historical data."""
        
        crypto_logger.logger.info(f"Backtesting RL strategy from {start_date} to {end_date}")
        
        # Generate test environment
        test_env = PortfolioEnvironment(self.assets)
        
        # Run backtest
        state = test_env.reset()
        portfolio_values = [test_env.portfolio_value]
        actions_taken = []
        daily_returns = []
        
        done = False
        while not done:
            # Get action from trained policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                if self.algorithm == 'PPO':
                    action_probs, _ = self.policy(state_tensor)
                    action = action_probs.cpu().numpy().flatten()
                else:
                    q_values = self.q_network(state_tensor)
                    action = torch.softmax(q_values, dim=1).cpu().numpy().flatten()
            
            # Take action
            next_state, reward, done, info = test_env.step(action)
            
            # Record results
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(action.copy())
            
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)
            
            state = next_state
        
        # Calculate performance metrics
        performance = await self._calculate_performance_metrics(portfolio_values, daily_returns)
        
        backtest_results = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'final_portfolio_value': portfolio_values[-1],
            'max_portfolio_value': max(portfolio_values),
            'min_portfolio_value': min(portfolio_values),
            'number_of_trades': len(actions_taken),
            'performance_metrics': performance,
            'portfolio_evolution': portfolio_values,
            'daily_returns': daily_returns
        }
        
        crypto_logger.logger.info(f"Backtest completed: Total return: {backtest_results['total_return']:.2%}")
        
        return backtest_results
    
    async def _calculate_performance_metrics(self, portfolio_values: List[float], 
                                           daily_returns: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if not daily_returns:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        returns_array = np.array(daily_returns)
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05
        excess_returns = returns_array - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (np.mean(returns_array) * 252) / max_drawdown if max_drawdown > 0 else 0
        
        # Alpha and Beta (using market proxy)
        # Simplified calculation - in practice, use proper benchmark
        market_returns = np.random.normal(0.0008, 0.02, len(returns_array))  # Mock market returns
        if len(returns_array) > 1 and len(market_returns) == len(returns_array):
            covariance = np.cov(returns_array, market_returns)[0][1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 0
            alpha = (np.mean(returns_array) - risk_free_rate / 252) - beta * (np.mean(market_returns) - risk_free_rate / 252)
            alpha *= 252  # Annualize
        else:
            alpha = 0
            beta = 0
        
        # Information ratio
        tracking_error = np.std(returns_array - market_returns) if len(market_returns) == len(returns_array) else volatility
        information_ratio = (np.mean(returns_array) - np.mean(market_returns)) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
    
    async def real_time_optimization(self, rebalance_frequency_minutes: int = 60) -> None:
        """Run real-time portfolio optimization."""
        
        crypto_logger.logger.info(f"Starting real-time optimization with {rebalance_frequency_minutes}-minute rebalancing")
        
        while True:
            try:
                # Get current market data
                current_portfolio = await self._get_current_portfolio()
                market_data = await self._get_current_market_data()
                
                # Generate optimization action
                optimization_action = await self.optimize_portfolio(current_portfolio, market_data)
                
                # Check if rebalancing is needed
                if await self._should_rebalance(optimization_action):
                    crypto_logger.logger.info("Rebalancing portfolio based on RL optimization")
                    
                    # Execute rebalancing (would integrate with trading system)
                    await self._execute_rebalancing(optimization_action)
                    
                    # Update performance tracking
                    await self._update_performance_tracking(optimization_action)
                
                # Wait for next optimization cycle
                await asyncio.sleep(rebalance_frequency_minutes * 60)
                
            except Exception as e:
                crypto_logger.logger.error(f"Error in real-time optimization: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _get_current_portfolio(self) -> Dict[str, float]:
        """Get current portfolio positions."""
        
        # Mock current portfolio
        return {asset: np.random.uniform(0, 100000) for asset in self.assets}
    
    async def _get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        
        # Mock market data
        return {
            'volatility': np.random.uniform(0.1, 0.5),
            'momentum': np.random.normal(0, 0.02),
            'sentiment': np.random.uniform(0, 100),
            'volume': np.random.lognormal(15, 1)
        }
    
    async def _should_rebalance(self, optimization_action: OptimizationAction) -> bool:
        """Determine if portfolio should be rebalanced."""
        
        # Check confidence threshold
        if optimization_action.confidence < 0.7:
            return False
        
        # Check time since last rebalance
        time_since_rebalance = datetime.now() - self.last_rebalance
        if time_since_rebalance.total_seconds() < 1800:  # Minimum 30 minutes
            return False
        
        # Check weight difference threshold
        current_weights = self.current_weights
        target_weights = optimization_action.target_weights
        
        weight_diff = np.sum(np.abs(target_weights - current_weights))
        
        return weight_diff > optimization_action.rebalance_threshold
    
    async def _execute_rebalancing(self, optimization_action: OptimizationAction):
        """Execute portfolio rebalancing."""
        
        # Mock execution (in production, integrate with trading system)
        self.current_weights = optimization_action.target_weights.copy()
        self.last_rebalance = datetime.now()
        
        crypto_logger.logger.info(f"Portfolio rebalanced: {dict(zip(self.assets, self.current_weights))}")
    
    async def _update_performance_tracking(self, optimization_action: OptimizationAction):
        """Update performance tracking metrics."""
        
        # Mock performance update
        current_value = np.random.uniform(95000, 105000)
        
        self.performance_tracker['portfolio_values'].append(current_value)
        
        if len(self.performance_tracker['portfolio_values']) > 1:
            daily_return = (current_value - self.performance_tracker['portfolio_values'][-2]) / self.performance_tracker['portfolio_values'][-2]
            self.performance_tracker['daily_returns'].append(daily_return)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization system status."""
        
        return {
            'algorithm': self.algorithm,
            'model_status': {
                'trained': len(self.training_rewards) > 0,
                'training_episodes': len(self.training_rewards),
                'average_training_reward': np.mean(self.training_rewards) if self.training_rewards else 0,
                'last_training_date': datetime.now().isoformat()  # Mock
            },
            'portfolio_status': {
                'assets': self.assets,
                'current_weights': dict(zip(self.assets, self.current_weights)),
                'last_rebalance': self.last_rebalance.isoformat(),
                'rebalances_today': 0  # Mock
            },
            'performance_summary': {
                'total_portfolio_values': len(self.performance_tracker['portfolio_values']),
                'current_value': self.performance_tracker['portfolio_values'][-1] if self.performance_tracker['portfolio_values'] else 0,
                'daily_returns_count': len(self.performance_tracker['daily_returns'])
            },
            'system_health': {
                'optimization_active': True,
                'model_accuracy': np.random.uniform(0.7, 0.9),  # Mock
                'last_optimization': datetime.now().isoformat(),
                'errors_24h': 0
            },
            'timestamp': datetime.now().isoformat()
        }

# Global RL optimizer instance
rl_optimizer = None  # Will be initialized with specific assets