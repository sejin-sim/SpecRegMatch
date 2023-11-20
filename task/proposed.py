import os, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.tensorboard import SummaryWriter

from dataloaders import make_loaders
from models.MultiOutputReg import Model
from utils.tqdm_config import get_tqdm_config
from utils.saver import Saver

def linear_rampup(cur_epoch, warm_up, lambda_u):
    tmp = np.clip((cur_epoch - warm_up) / int(warm_up*0.8), 0.0, 1.0)

    return lambda_u * float(tmp)

class Trainer(object):
    def __init__(self, args):

        self.args = args
        self.saver = Saver(algorithm = f"proposed_{self.args.result_name}") # result dir name
        self.csv_dir = os.path.join(self.saver.experiment_dir, 'csvs')
        self.plot_dir = os.path.join(self.saver.experiment_dir, 'plot')
        [os.makedirs(f, exist_ok=True) for f in [self.csv_dir, self.plot_dir]]
        
        self.saver.save_experiment_config(self.args)
        self.writer = SummaryWriter(self.saver.experiment_dir)

        self.labeled_statistics = torch.load(os.path.join("./data", f'labeled_statistics_{args.labeled_data_N}_{args.labeled_train_amount}.pth'))
        self.unlabeled_statistics = torch.load(os.path.join("./data", 'unlabeled_statistics.pth'))
        self.train_loader, self.valid_loader, self.test_loader, self.unlabel_loader = make_loaders(args, self.labeled_statistics, self.unlabeled_statistics, num_workers=4)
        
        self.model = Model(args)
        self.model.to(torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu'))
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss(reduction='none')
        
        self.args.best_val_loss = np.inf
        self.args.best_epoch = 0
        self.cnt_train, self.cnt_val, self.cnt_test = 0, 0, 0
        
        # Epoch-level evaluation
        for s in ['train', 'valid', 'test']:
            setattr(self, f'{s}_losses', [])
            
            l_specific = [f'label{str(i)}' for i in range(1, 11)] + ['label_overall']
            for l in l_specific:
                setattr(self, f'{s}_{l}_r2', [])
                setattr(self, f'{s}_{l}_mae', [])
        
    
    def training(self, epoch):                 
        print('[Epoch: %d]' % (epoch))
        
        self.model.train()
        losses_t, losses_x, losses_ul  = 0.0, 0.0, 0.0 
        losses_bt, losses_mx = 0.0, 0.0
        
        total_steps = len(self.train_loader)
        lambda_u = linear_rampup(epoch, self.args.warm_up, self.args.lambda_u)
        lambda_mx = linear_rampup(epoch, self.args.warm_up, self.args.lambda_mx)
        lambda_t = linear_rampup(epoch, self.args.warm_up, self.args.lambda_t)
        
        preds, labels = [], []
        with tqdm(**get_tqdm_config(total=len(self.train_loader),leave=True, color='blue')) as pbar:

            for idx, (labeled, unlabeled) in enumerate(zip(self.train_loader, self.unlabel_loader)):

                self.optimizer.zero_grad()
                
                labeled_x, label = labeled['feature'], labeled['target']
                weak, strong, mix = unlabeled['weak'], unlabeled['strong'], unlabeled['mix']
                
                x_all = torch.cat([labeled_x, weak, strong, mix], axis=0).cuda(self.args.cuda)
                label = label.cuda(self.args.cuda)
                                                        
                # predict
                pred_all, loss_bt = self.model(x_all)
                pred_x, pred_u_w, pred_u_s, pred_u_m = pred_all.split(x_all.size(0) // 4)
                                            
                # labeled_loss
                loss_labeled = self.criterion(pred_x, label)
                loss_x = loss_labeled.mean()
                
                # unlabeled_loss
                loss_unlabeled = self.criterion(pred_u_s, pred_u_w.detach())               
                loss_u = loss_unlabeled.mean()

                # unlabeled_loss + mix
                loss_unlabeled_mix = self.criterion(pred_u_m, pred_u_w.detach())               
                loss_mx = loss_unlabeled_mix.mean()                
                                
                loss = loss_x + (lambda_u * loss_u) + (lambda_mx * loss_mx) + (lambda_t * loss_bt)
                loss.backward()
                self.optimizer.step()
                
                losses_t += loss.item()
                losses_x += loss_x.item()
                losses_ul += (lambda_u * loss_u).item()
                losses_bt += (lambda_t * loss_bt).item()
                losses_mx += (lambda_mx * loss_mx).item()

                preds.append(pred_x.detach())
                labels.append(label.detach())
                
                self.cnt_train += 1
                pbar.set_description("Train(%2d/%2d)-Loss: %.4f|Loss_x: %.4f|Loss_ul: %.4f|Loss_bt: %.4f|Loss_mx: %.4f"
                                     %(idx+1, total_steps, losses_t/(idx+1), losses_x/(idx+1), losses_ul/(idx+1), losses_bt/(idx+1), losses_mx/(idx+1)))
                pbar.update(1)
            
            self.writer.add_scalars(
                    'Training epoch',
                    {'Loss_t': losses_t/(idx+1),
                     'Loss_x': losses_x/(idx+1),
                     'Loss_u': losses_ul/(idx+1),  
                     'lambda_u': lambda_u, 
                     'Loss_bt': losses_bt/(idx+1),                        
                     'lambda_t': lambda_t,          
                     'Loss_mx': losses_mx/(idx+1),                        
                     'lambda_mx': lambda_mx,          
                     },
                    global_step=epoch
                )
            
            losses_t /= (idx+1)     
            self.train_losses.append(losses_t)         
            _, r2s, maes = self.get_regression_measures(preds, labels, epoch=epoch, phase="train", plot=False, inverse_scaling=True)

            for idx, l in zip(range(len(r2s)),
                        [f'label{str(i)}' for i in range(1, 11)] + ['label_overall']):
                getattr(self, f'train_{l}_r2').append(r2s[idx])
                getattr(self, f'train_{l}_mae').append(maes[idx])
            
            pbar.set_description("%5s(%2d/%2d)-Loss: %.4f|R2_mean:%.3f|MAE_mean:%.3f"%('Train', epoch, self.args.epochs, losses_t/(idx+1), np.mean(r2s), np.mean(maes)))
     
        return 3
                
    @torch.no_grad()
    def evaluation(self, epoch: int, phase: str, save: bool=False):
        
        self.model.eval()
        losses = 0.0
        
        if phase == "Valid":
            data_loader, c = self.valid_loader, 'green'
        elif phase == "Test":
            data_loader, c = self.test_loader, 'red'
        
        total_steps = len(data_loader.dataset) // self.args.batch_size
        
        preds, labels = [], []
        with tqdm(**get_tqdm_config(total=len(data_loader), leave=True, color=c)) as pbar:
            
            for idx, labeled in enumerate(data_loader):
                       
                labeled_x, label = labeled['feature'], labeled['target']
                labeled_x = labeled_x.cuda(self.args.cuda)
                label = label.cuda(self.args.cuda)
                
                # predict
                pred, _ = self.model(labeled_x)
                
                loss, _ = self.get_loss(pred, label)
                losses += loss.item()
                
                preds.append(pred.detach())
                labels.append(label.detach())
                           
                self.writer.add_scalars(
                    f'{phase} steps',
                    {'Losses': losses/(idx+1)},
                    global_step=self.cnt_train
                )
                
                _, r2s, maes = self.get_regression_measures(preds, labels, epoch=epoch, phase=phase, plot=False, inverse_scaling=True)

                if phase == "valid":
                    tmp = self.cnt_val
                    r2_mean, mae_mean = self.summary(f'{phase} steps', tmp, labels, preds, epoch, False)
                    self.cnt_val += 1
                else:
                    tmp = self.cnt_test
                    r2_mean, mae_mean = self.summary(f'{phase} steps', tmp, labels, preds, epoch, False)
                    self.cnt_test += 1
                    
                pbar.set_description("%5s(%2d/%2d)-Loss: %.4f|R2_mean:%.3f|MAE_mean:%.3f"%(phase, idx+1, total_steps, losses/(idx+1), r2_mean, mae_mean))
                pbar.update(1)
            
            losses /= (idx+1)
            getattr(self, f"{phase.lower()}_losses").append(losses)

            _, r2s, maes = self.get_regression_measures(preds, labels, epoch, phase, plot=True, inverse_scaling=True) 
            
            for idx, l in zip(range(len(r2s)),
                [f'label{str(i)}' for i in range(1, 11)] + ['label_overall']):
                    getattr(self, f'{phase.lower()}_{l}_r2').append(r2s[idx])
                    getattr(self, f'{phase.lower()}_{l}_mae').append(maes[idx])
          
            pbar.set_description(
                "%5s(%2d/%2d)-Loss: %.4f|R2_mean:%.3f|MAE_mean:%.3f"%(
                    phase, epoch, self.args.epochs, losses, np.mean(r2s), np.mean(maes)
                )
            )   

            if phase == 'Valid':
                if losses < self.args.best_val_loss:
                    self.args.best_val_loss = losses
                    self.args.best_epoch = epoch

                    new_r2s, new_maes = [], []
                    for r, m in zip(r2s, maes):
                        new_r2s.append(str(np.round(r, 4)))
                        new_maes.append(str(np.round(m, 4)))
                    self.args.val_best_r2s = new_r2s
                    self.args.val_best_maes = new_maes
                                         
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.saver.experiment_dir, 'best_model.pth')
                        )
                    self.saver.save_experiment_config(self.args)
                    
                    save = True
            
            if phase == 'Test' and save==True:
                
                new_r2s, new_maes = [], []
                for r, m in zip(r2s, maes):
                    new_r2s.append(str(np.round(r, 4)))
                    new_maes.append(str(np.round(m, 4)))
                self.args.test_best_r2s = new_r2s
                self.args.test_best_maes = new_maes
                
                self.saver.save_experiment_config(self.args)
                
            return save, losses

    def get_loss(self, preds, labels):
        losses = self.criterion(preds, labels.detach())
        mean_loss = losses.mean()
        
        return mean_loss, [loss.mean() for loss in losses.split(1, dim=1)]  

    def summary(self, phase, step, labels, preds, epoch, inverse_scaling):
        
        losses, r2s, maes = self.get_regression_measures(preds, labels, epoch=epoch, phase="train", plot=False, inverse_scaling=inverse_scaling)
                
        result_dict = {}
        l_specific = [f'label{str(i)}' for i in range(1, 11)] + ['label_overall']
        for idx, l in enumerate(l_specific):
            result_dict[f'Loss_{l}'] = losses[idx]
            result_dict[f'R2_{l}'] = r2s[idx]
            result_dict[f'MAE_{l}'] = maes[idx]
        
        self.writer.add_scalars(
            phase, result_dict, global_step=step
        )
        
        return np.mean(r2s), np.mean(maes)

    def get_regression_measures(self,
                            preds: torch.Tensor,
                            labels: torch.Tensor,
                            epoch: int,
                            phase: str,
                            plot: bool,
                            inverse_scaling: bool=False):
        
        preds = torch.cat(preds).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()     
              
        if inverse_scaling:
            for idx, n in enumerate(['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10', 'label_overall']):
                maximum = self.labeled_statistics['max'+f'_{n}']
                minimum = self.labeled_statistics['min'+f'_{n}']
                preds[:, idx] = (maximum-minimum)*preds[:, idx] + minimum
                labels[:, idx] = (maximum-minimum)*labels[:, idx] + minimum
        
        losses, r2s, maes = [], [], []
        for idx in range(self.args.num_output):
            tmp_preds = preds[:, idx]
            tmp_labels = labels[:, idx]
            
            losses.append(mean_squared_error(tmp_labels, tmp_preds))
            r2s.append(r2_score(tmp_labels, tmp_preds))
            maes.append(mean_absolute_error(tmp_labels, tmp_preds))

        if plot == True:
            self.plot_results(preds, labels, phase, epoch)
            
        return losses, r2s, maes

    def plot_results(self, preds: torch.Tensor, labels: torch.Tensor,
                     phase: str, epoch: int):

        preds = pd.DataFrame(preds, columns=['pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6', 'pred7', 'pred8', 'pred9', 'pred10', 'pred_overall'])
        labels = pd.DataFrame(labels, columns=['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10', 'label_overall'])
        
        preds = pd.concat([preds, labels], axis=1)
        preds.to_csv(os.path.join(self.csv_dir, f'preds_labels_{phase}_{str(epoch)}.csv'), index=False)

        for n, l in zip(labels.columns, range(len(labels.columns))):
            min_v = preds.iloc[:, [l, l+self.args.num_output]].min().min()
            max_v = preds.iloc[:, [l, l+self.args.num_output]].max().max()
            
            colors = {
                'train': 'blue',
                'Valid': 'g',
                'Test': 'r'
            }            

            tmp_pred = preds.iloc[:, l]
            tmp_label = preds.iloc[:, l+self.args.num_output]
            
            plt.figure(figsize=(10, 10))
            plt.plot([min_v, min_v], [max_v, max_v], linestyle='--', color='grey')
            plt.scatter(tmp_label, tmp_pred, c=colors[phase], marker='o', alpha=0.5, s=40)
            
            plt.title('Actual-Pred plot', fontsize=15)
            plt.xlim(min_v, max_v)
            plt.ylim(min_v, max_v)
            
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            
            plt.savefig(os.path.join(self.plot_dir, f'actual_pred_plot_{phase}_{str(epoch)}_{n}.png'))
            plt.close()

