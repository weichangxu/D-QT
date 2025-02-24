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
                exp='QT',
                ema_decay=0.995,
                step_start_ema=1000,
                update_ema_every=5,
                lr=3e-4,
                clr=3e-4,
                weight_decay=1e-4,
                warmup_steps=0,
                lr_decay=False,
                decay_start_step=10000,
                decay_rate=0.1,
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
        self.warmup_steps = warmup_steps

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=clr)

        self.decay_start_step = decay_start_step
        self.decay_rate = decay_rate

        if lr_decay:
            # self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            # self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)
            def lr_lambda(epoch):
                if epoch < self.decay_start_step:
                    return 1.0  # 学习率保持不变
                else:
                    return 0.9 ** ((epoch - self.decay_start_step)*self.decay_rate)
                
            self.actor_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda)
        
        if warmup_steps != 0:
            self.actor_lr_scheduler_warm = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lambda steps: min((steps+1)/warmup_steps, 1))
            self.critic_lr_scheduler_warm = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lambda steps: min((steps+1)/warmup_steps, 1))
            


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
        self.exp = exp
        self.N = 32
    
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
            if self.exp == 'QT' or self.exp == 'QT_pro':
                loss_metric = self.train_step(log_writer, loss_metric)
            elif self.exp == 'C51':
                loss_metric = self.train_step_C51(log_writer, loss_metric)
            elif self.exp == 'IQN_d4pg':
                loss_metric = self.train_step_IQN_d4pg(log_writer, loss_metric)
            elif self.exp == 'IQN_QT':
                loss_metric = self.train_step_IQN_QT(log_writer, loss_metric)
            elif self.exp == 'DIQN_QT':
                loss_metric = self.train_step_DIQN_QT(log_writer, loss_metric)

        
        
        # if self.lr_decay and self.step > self.warmup_steps: 
        #     self.actor_lr_scheduler.step()
        #     # print('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])
        #     self.critic_lr_scheduler.step()


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
                log_writer.add_scalar(f'evaluation/{k}', v, iter_num * num_steps)

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
        current_q1, current_q2 = self.critic.forward(states, actions)

        T = current_q1.shape[1]
        repeat_num = 10

        if self.max_q_backup:
            states_rpt = torch.repeat_interleave(states, repeats=repeat_num, dim=0)
            actions_rpt = torch.repeat_interleave(actions, repeats=repeat_num, dim=0)
            rewards_rpt = torch.repeat_interleave(rewards, repeats=repeat_num, dim=0)
            noise = torch.zeros(1, 1, 1)
            noise = torch.cat([noise, torch.randn(repeat_num-1, 1, 1)], dim=0).repeat(batch_size, 1, 1).to(device) # keep rtg logic
            rtg_rpt = torch.repeat_interleave(rtg, repeats=repeat_num, dim=0)
            rtg_rpt[:, -2:-1] = rtg_rpt[:, -2:-1] + noise * 0.1
            timesteps_rpt = torch.repeat_interleave(timesteps, repeats=repeat_num, dim=0)
            attention_mask_rpt = torch.repeat_interleave(attention_mask, repeats=repeat_num, dim=0)
            _, next_action, _ = self.ema_model(
                states_rpt, actions_rpt, rewards_rpt, None, rtg_rpt[:,:-1], timesteps_rpt, attention_mask=attention_mask_rpt,
            )
        else:
            _, next_action, _ = self.ema_model(
                states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

        if self.k_rewards:
            if self.max_q_backup:
                critic_next_states = states_rpt[:, -1]
                next_action = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action)
                target_q1 = target_q1.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
            else:
                critic_next_states = states[:, -1]
                next_action = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action)
            target_q = torch.min(target_q1, target_q2) # [B, 1]

            not_done =(1 - dones[:, -1]) # [B, 1]
            if self.use_discount:
                rewards[:, -1] = 0.
                mask_ = attention_mask.sum(dim=1).detach().cpu() # [B]
                discount = [i - 1 - torch.arange(i) for i in mask_]
                discount = torch.stack([torch.cat([i, torch.zeros(T - len(i))], dim=0) for i in discount], dim=0) # [B, T]
                discount = (self.discount ** discount).unsqueeze(-1).to(device) # [B, T, 1]
                k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount, dim=1).flip(dims=[1]) # [B, T, 1]

                discount = [torch.arange(i) for i in mask_] # 
                discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
                discount =  (self.discount ** discount).unsqueeze(-1).to(device)
                k_rewards = k_rewards / discount
                
                discount = [i - 1 - torch.arange(i) for i in mask_] # [B]
                discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
                discount = (self.discount ** discount).to(device) # [B, T]
                target_q = (k_rewards + (not_done * discount * target_q).unsqueeze(-1)).detach() # [B, T, 1]
                
            else:
                k_rewards = (rtg[:,:-1] - rtg[:, -2:-1])* self.scale # [B, T, 1]
                target_q = (k_rewards + (not_done * target_q).unsqueeze(-1)).detach() # [B, T, 1]
        else:
            if self.max_q_backup:
                target_q1, target_q2 = self.critic_target(states_rpt, next_action) # [B*repeat, T, 1]
                target_q1 = target_q1.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
                target_q2 = target_q2.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
            else:
                target_q1, target_q2 = self.critic_target(states, next_action) # [B, T, 1]
            target_q = torch.min(target_q1, target_q2) # [B, T, 1]
            target_q = rewards[:, :-1] + self.discount * target_q[:, 1:]
            target_q = torch.cat([target_q, torch.zeros(batch_size, 1, 1).to(device)], dim=1) 


        critic_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0]) \
            + F.mse_loss(current_q2[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0]) 

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
        q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
        # q1_new_action, q2_new_action = self.critic(state_target, action_preds_)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        # actor_loss = self.eta * bc_loss + q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        if self.warmup_steps !=0 and self.step <= self.warmup_steps:
            self.actor_lr_scheduler_warm.step()
            self.critic_lr_scheduler_warm.step()
        
        # print('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])


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
            log_writer.add_scalar('Actor LR', self.actor_optimizer.param_groups[0]['lr'], self.step)
            log_writer.add_scalar('Critic LR', self.actor_optimizer.param_groups[0]['lr'], self.step)


        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric
    
    def train_step_C51(self, log_writer=None, loss_metric={}):
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
        
        current_q1_dist, current_q2_dist = self.critic.forward(states, actions)

        T = current_q1_dist.shape[1]
        
        
        _, next_action, _ = self.ema_model(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        
        
        target_q1_dist, target_q2_dist = self.critic_target(states, next_action)
    
        # 计算最小 Q 值的概率分布
        target_q_dist = self.critic.min(target_q1_dist, target_q2_dist) 
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
    
    
    def train_step_IQN_d4pg(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        T = states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''
        
        
        Q_current, taus = self.critic(states, actions, self.N)

        with torch.no_grad():
        #     _, next_actions, _ = self.actor(
        #     states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        # ) # next_actions: [B, T, act_dim]
            _, next_actions, _ = self.ema_model(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        ) # next_actions: [B, T, act_dim]
            Q_targets_next, _ = self.critic_target(states, next_actions, self.N) # [B*T, n_atom, 1]
            Q_targets_next = Q_targets_next.transpose(1,2) # [B*T, 1 n_atom]
            Q_targets = rewards.view(-1,1).unsqueeze(1) + self.discount * Q_targets_next * (1. - dones.view(-1,1).unsqueeze(1)) # [B*T, 1 n_atom]
        
        

        assert Q_targets.shape == (batch_size * T, 1, self.N)
        assert Q_current.shape == (batch_size * T, self.N, 1)

        td_error = Q_targets - Q_current
        assert td_error.shape == (batch_size * T, self.N, self.N), "wrong td error shape"
        huber_l = self.calculate_huber_loss(td_error, 1.0) # [B*T, n_atom, n_atom]
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0

        critic_loss = quantil_l.sum(dim=1)[attention_mask.reshape(-1) > 0].mean(dim=1).mean()
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

        q_new_action = self.critic.get_qvalues(actor_states, action_preds_)
        q_loss = - q_new_action.mean() * 0.01





        
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


        """ log """
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            # log_writer.add_scalar('Target_Q Mean', target_q_mean.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        # loss_metric['target_q_mean'].append(target_q_mean.mean().item())  # 记录均值

        # loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric
    
    def calculate_huber_loss(self, td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
        return loss
    

    def train_step_IQN_QT(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        T = states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''
        
        
        current_q = self.critic.get_qvalues(states, actions) # [B*T, 1]
        current_q = current_q.reshape(batch_size, -1, 1)

        _, next_action, _ = self.ema_model(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        critic_next_states = states[:, -1]
        next_action = next_action[:, -1]
        target_q = self.critic_target.get_qvalues(critic_next_states, next_action) #[B, 1]

        not_done =(1 - dones[:, -1]) # [B, 1]

        rewards[:, -1] = 0.
        mask_ = attention_mask.sum(dim=1).detach().cpu() # [B]
        discount = [i - 1 - torch.arange(i) for i in mask_]
        discount = torch.stack([torch.cat([i, torch.zeros(T - len(i))], dim=0) for i in discount], dim=0) # [B, T]
        discount = (self.discount ** discount).unsqueeze(-1).to(device) # [B, T, 1]
        k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount, dim=1).flip(dims=[1]) # [B, T, 1]

        discount = [torch.arange(i) for i in mask_] # 
        discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
        discount =  (self.discount ** discount).unsqueeze(-1).to(device)
        k_rewards = k_rewards / discount
        
        discount = [i - 1 - torch.arange(i) for i in mask_] # [B]
        discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
        discount = (self.discount ** discount).to(device) # [B, T]
        target_q = (k_rewards + (not_done * discount * target_q).unsqueeze(-1)).detach() # [B, T, 1]

        critic_loss = F.mse_loss(current_q[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0])

        
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

        q_new_action = self.critic.get_qvalues(actor_states, action_preds_)
        q_loss = - q_new_action.mean() * 0.01

        # 2-17
        # q_loss = - q_new_action.mean()



        # 2-17
        # actor_loss = bc_loss / bc_loss.detach() +  q_loss/ q_loss.detach()

        
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss



        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        if self.warmup_steps !=0:
            if self.step <= self.warmup_steps:
                self.actor_lr_scheduler_warm.step()
                self.critic_lr_scheduler_warm.step()
            else:
                if self.lr_decay:
                    self.actor_lr_scheduler.step()
                    self.critic_lr_scheduler.step()

        else:
            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
       
        
        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.step += 1


        """ log """
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
            log_writer.add_scalar('Actor LR', self.actor_optimizer.param_groups[0]['lr'], self.step)
            log_writer.add_scalar('Critic LR', self.actor_optimizer.param_groups[0]['lr'], self.step)

            # log_writer.add_scalar('Current_Q Mean', current_q.mean().item(), self.step)


        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())  # 记录均值
        # loss_metric['current_q_mean'].append(current_q.mean().item())

        # loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric
    
    def train_step_DIQN_QT(self, log_writer=None, loss_metric={}):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        T = states.shape[1]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device


        '''Q Training'''
        
        
        current_q1, current_q2 = self.critic.get_qvalues(states, actions) # [B*T, 1]
        current_q1 = current_q1.reshape(batch_size, -1, 1)
        current_q2 = current_q2.reshape(batch_size, -1, 1)
        

        _, next_action, _ = self.ema_model(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        critic_next_states = states[:, -1]
        next_action = next_action[:, -1]
        target_q1, target_q2 = self.critic_target.get_qvalues(critic_next_states, next_action) #[B, 1]

        # target_q = torch.min(target_q1, target_q2)
        if np.random.uniform() > 0.5:
            target_q = target_q1
        else:
            target_q = target_q2

        not_done =(1 - dones[:, -1]) # [B, 1]

        rewards[:, -1] = 0.
        mask_ = attention_mask.sum(dim=1).detach().cpu() # [B]
        discount = [i - 1 - torch.arange(i) for i in mask_]
        discount = torch.stack([torch.cat([i, torch.zeros(T - len(i))], dim=0) for i in discount], dim=0) # [B, T]
        discount = (self.discount ** discount).unsqueeze(-1).to(device) # [B, T, 1]
        k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount, dim=1).flip(dims=[1]) # [B, T, 1]

        discount = [torch.arange(i) for i in mask_] # 
        discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
        discount =  (self.discount ** discount).unsqueeze(-1).to(device)
        k_rewards = k_rewards / discount
        
        discount = [i - 1 - torch.arange(i) for i in mask_] # [B]
        discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
        discount = (self.discount ** discount).to(device) # [B, T]
        target_q = (k_rewards + (not_done * discount * target_q).unsqueeze(-1)).detach() # [B, T, 1]

        critic_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0]) \
            + F.mse_loss(current_q2[:, :-1][attention_mask[:, :-1]>0], target_q[:, :-1][attention_mask[:, :-1]>0]) 

        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        if self.warmup_steps !=0:
            if self.step <= self.warmup_steps:
                self.actor_lr_scheduler_warm.step()
                self.critic_lr_scheduler_warm.step()
            else:
                if self.lr_decay:
                    self.actor_lr_scheduler.step()
                    self.critic_lr_scheduler.step()

        else:
            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
       

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

        q1_new_action, q2_new_action = self.critic.get_qvalues(actor_states, action_preds_)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
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


        """ log """
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
            log_writer.add_scalar('Actor LR', self.actor_optimizer.param_groups[0]['lr'], self.step)
            log_writer.add_scalar('Critic LR', self.actor_optimizer.param_groups[0]['lr'], self.step)


        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())  # 记录均值


        return loss_metric
    
    