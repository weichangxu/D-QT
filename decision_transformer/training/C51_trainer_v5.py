import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import copy
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:

    def __init__(self, 
                model, 
                critic,
                batch_size, 
                tau,
                discount,
                get_batch, 
                loss_fn, 
                eval_fns=None,
                max_q_backup=False,
                eta=1.0,
                eta2=1.0,
                ema_decay=0.995,
                step_start_ema=1000,
                update_ema_every=5,
                lr=3e-4,
                weight_decay=1e-4,
                lr_decay=False,
                lr_maxt=100000,
                lr_min=0.,
                grad_norm=1.0,
                scale=1.0,
                k_rewards=True,
                use_discount=True
            ):
        
        self.actor = model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.discount = discount
        self.grad_norm = grad_norm
        self.eta = eta
        self.eta2 = eta2
        self.lr_decay = lr_decay
        self.scale = scale
        self.k_rewards = k_rewards
        self.use_discount = use_discount

        self.start_time = time.time()
        self.step = 0
    
    def step_ema(self):
        if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.actor)

    def train_iteration(self, num_steps, logger, iter_num=0, log_writer=None):

        logs = dict()

        train_start = time.time()

        self.actor.train()
        self.critic.train()
        loss_metric = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
        }
        for _ in trange(num_steps):
            loss_metric = self.train_step(log_writer, loss_metric)
        
        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.record_tabular('Target Q Mean', np.mean(loss_metric['target_q_mean']))
        logger.dump_tabular()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()


        self.actor.eval()
        self.critic.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor, self.critic_target)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))
            logger.record_tabular(k, float(v))
        logger.record_tabular('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])
        logger.record_tabular('Current critic learning rate', self.critic_optimizer.param_groups[0]['lr'])
        logger.dump_tabular()

        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret
        return logs
    
    def scale_up_eta(self, lambda_):
        self.eta2 = self.eta2 / lambda_

    def train_step_v0(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''
        
        current_q1_dist, current_q2_dist = self.critic(states, actions)

        T = current_q1_dist.shape[1]
        
        
        _, next_action, _ = self.ema_model(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        
        
        target_q1_dist, target_q2_dist = self.critic_target(states, next_action)
    
        # 计算最小 Q 值的概率分布
        # target_q_dist = self.critic.min(target_q1_dist, target_q2_dist) 
        if np.random.uniform() > 0.5:
            target_q_dist = target_q1_dist
        else:
            target_q_dist = target_q2_dist
        # 计算目标 Q 值分布（投影到 self.support）
        target_q_dist = self.critic.compute_target_distribution(rewards, dones, target_q_dist, self.discount)
        target_q = (target_q_dist * self.critic.support.to(target_q_dist.device)).sum(dim=-1)  

    
        loss_q1 = -(target_q_dist[:, :-1][attention_mask[:, :-1]>0] * current_q1_dist[:, :-1][attention_mask[:, :-1]>0].log()).sum(dim=-1).mean()
        loss_q2 = -(target_q_dist[:, :-1][attention_mask[:, :-1]>0] * current_q2_dist[:, :-1][attention_mask[:, :-1]>0].log()).sum(dim=-1).mean()
        
        critic_loss = loss_q1 + loss_q2


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        '''Policy Training'''        
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1]>0].mean()
        if reward_preds is not None:
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            rewards_loss = F.mse_loss(reward_preds, reward_target)
        else:
            rewards_loss = 0
        bc_loss = F.mse_loss(action_preds_, action_target_) + states_loss + rewards_loss



        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]



        q1_dist, q2_dist = self.critic(actor_states, action_preds_)
        q1_mean = (q1_dist * self.critic.support.to(actor_states.device)).sum(dim=-1)
        q2_mean = (q2_dist * self.critic.support.to(actor_states.device)).sum(dim=-1)
        if np.random.uniform() > 0.5:
            q_loss = - q1_mean.mean()/ q2_mean.abs().mean().detach()
        else:
            q_loss = - q2_mean.mean()/ q1_mean.abs().mean().detach()

        # q_loss = - q1_mean.mean()/ q2_mean.abs().mean().detach() - q2_mean.mean()/ q1_mean.abs().mean().detach()

        
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        # actor_loss = self.eta * bc_loss + q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.step += 1

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())  # 记录均值

        # loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric
    


    def train_step(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''
        
        current_q1_dist, current_q2_dist = self.critic(states, actions)

        T = current_q1_dist.shape[1]
        
        
        _, next_action, _ = self.ema_model(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        
        
        target_q1_dist, target_q2_dist = self.critic_target(states, next_action)
    
        # 计算最小 Q 值的概率分布
        # target_q_dist = self.critic.min(target_q1_dist, target_q2_dist) 
        # if np.random.uniform() > 0.5:
        #     target_q_dist = target_q1_dist
        # else:
        #     target_q_dist = target_q2_dist
        target_q_dist = target_q1_dist
        # 计算目标 Q 值分布（投影到 self.support）

        
        q_dist = self.critic(torch.tensor(states), torch.tensor(actions))  # [B, T, n_atoms]

        qdist_loss = None  # dummy variable to remove redundant warnings
        
        # Adjust for B*T dimensions
        target_z_dist_flat = target_q_dist.view(-1, self.critic.num_atoms)  # Flatten B*T into a single dimension
        rewards_flat = rewards.reshape(-1)  # Flatten [B, T] to [B*T]
        terminates_flat = dones.reshape(-1)  # Flatten [B, T] to [B*T]

        # Reprojected distribution using adjusted dimensions
        reprojected_dist = self.critic.reproject2(target_z_dist_flat.cpu().data.numpy(), rewards_flat, terminates_flat, self.discount)

        # Calculate the loss with the updated shape
        qdist_loss = -(torch.tensor(reprojected_dist, requires_grad=False) * torch.log(q_dist.view(-1, self.critic.num_atoms) + 1e-10)).sum(dim=1).mean()




        


        self.critic_optimizer.zero_grad()
        qdist_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        '''Policy Training'''        
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1]>0].mean()
        if reward_preds is not None:
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            rewards_loss = F.mse_loss(reward_preds, reward_target)
        else:
            rewards_loss = 0
        bc_loss = F.mse_loss(action_preds_, action_target_) + states_loss + rewards_loss



        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]



        q1_dist, q2_dist = self.critic(actor_states, action_preds_)
        q1_mean = (q1_dist * self.critic.support.to(actor_states.device)).sum(dim=-1)
        q2_mean = (q2_dist * self.critic.support.to(actor_states.device)).sum(dim=-1)
        if np.random.uniform() > 0.5:
            q_loss = - q1_mean.mean()/ q2_mean.abs().mean().detach()
        else:
            q_loss = - q2_mean.mean()/ q1_mean.abs().mean().detach()

        # q_loss = - q1_mean.mean()/ q2_mean.abs().mean().detach() - q2_mean.mean()/ q1_mean.abs().mean().detach()

        
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        # actor_loss = self.eta * bc_loss + q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.step += 1

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())  # 记录均值

        # loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric
    

    
    