import utils.general as utils
import copy
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import logging
import torchvision
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import advertorch
from advertorch.context import ctx_noparamgrad_and_eval

class TrainRunner():
    def __init__(self,**kwargs):

        if (type(kwargs['conf']) == str):
            self.conf = ConfigFactory.parse_file(kwargs['conf'])
            self.conf_filename = kwargs['conf']
        else:
            self.conf = kwargs['conf']
        torch.manual_seed(self.conf.get_int('train.seed'))
        self.nepochs = kwargs['nepochs']
        self.expname =self.conf.get_string("train.expname") +  kwargs['expname']
        loadexpname = self.expname
        self.expname = self.expname + '_continue' if (kwargs['is_continue']) else self.expname
        self.GPU_INDEX = kwargs['gpu_index']
        self.batch_size = kwargs['batch_size']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.eval_frequency = kwargs['eval_frequency']


        utils.mkdir_ifnotexists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../',self.exps_folder_name))

        self.expdir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../',self.exps_folder_name), self.expname)
        loadexpdir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../',self.exps_folder_name), loadexpname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        self.log_dir = log_dir
        utils.mkdir_ifnotexists(log_dir)
        utils.configure_logging(kwargs['debug'],kwargs['quiet'],os.path.join(self.log_dir,'log.txt'))

        self.learning_log_epoch_path = os.path.join(self.log_dir,'learning_epoch_log.csv')
        self.learning_log_step_path = os.path.join(self.log_dir, 'learning_step_log.csv')
        self.debug_log_conf_path = os.path.join(self.log_dir, 'debug_conf.csv')

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_path = os.path.join(self.expdir, self.timestamp, 'checkpoints','model')
        utils.mkdir_ifnotexists(self.model_path)

        self.opt_path = os.path.join(self.expdir, self.timestamp, 'checkpoints', 'optimizer')
        utils.mkdir_ifnotexists(self.opt_path)

        self.sched_path = os.path.join(self.expdir, self.timestamp, 'checkpoints', 'sched')
        utils.mkdir_ifnotexists(self.sched_path)


        self.datadir = utils.concat_home_dir('{0}'.format(self.conf.get_string('train.datapath')))
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        # Backup code
        self.code_path = os.path.join(self.expdir, self.timestamp, 'code')
        utils.mkdir_ifnotexists(self.code_path)
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'training_adv'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'preprocess'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'utils'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'model'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'datasets'))
        utils.mkdir_ifnotexists(os.path.join(self.code_path, 'confs'))
        os.system("""cp -r ./training_adv/* "{0}" """.format(os.path.join(self.code_path, 'training_adv')))
        os.system("""cp -r ./model/* "{0}" """.format(os.path.join(self.code_path, 'model')))
        os.system("""cp -r ./preprocess/* "{0}" """.format(os.path.join(self.code_path, 'preprocess')))
        os.system("""cp -r ./utils/* "{0}" """.format(os.path.join(self.code_path, 'utils')))
        os.system("""cp -r ./datasets/* "{0}" """.format(os.path.join(self.code_path, 'datasets')))
        os.system("""cp -r ./confs/* "{0}" """.format(os.path.join(self.code_path, 'confs')))
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.code_path, 'confs/runconf.conf')))

        logging.info('shell command : {0}'.format(' '.join(sys.argv)))

        debug_conf = {}
        debug_conf['transforms_prop'] = [str(self.conf.get_config('train.trans_props'))]
        debug_conf['transforms'] = [str(self.conf.get_list('train.transforms'))]
        debug_conf['batch_size'] = self.batch_size

        transforms = []
        for tname in self.conf.get_list('train.transforms'):
            if tname in self.conf.get_config('train.trans_props'):
                transforms.append(utils.get_class(tname)(**self.conf.get_config('train.trans_props.{0}'.format(tname))))
            else:
                transforms.append(utils.get_class(tname)())
        debug_conf['train_dataset'] = [self.conf.get_string('train.dataset')]
        ds_train = utils.get_class(self.conf.get_string('train.dataset'))(self.datadir,
                                                                          train=True,
                                                                          download=True,
                                                                                transform=torchvision.transforms.Compose(transforms),
                                                                                )
        ds_test = utils.get_class(self.conf.get_string('train.dataset'))(self.datadir,
                                                                                train=False,
                                                                                download=True,
                                                                                transform=torchvision.transforms.ToTensor())
        self.ds_len = len(ds_train)

        self.train_dataloader = torch.utils.data.DataLoader(ds_train,
                                                      batch_size=self.batch_size,
                                                      num_workers=kwargs['threads'],
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      drop_last=True
                                                     )
        self.test_dataloader = torch.utils.data.DataLoader(ds_test,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=kwargs['threads'],
                                                      drop_last=False)

        debug_conf['network.type'] = [self.conf.get_string('network.type')]
        debug_conf['network.model'] = [self.conf.get_string('network.model')]
        self.network = utils.get_class(self.conf.get_string('network.type'))(self.conf.get_config('network')) #ManifoldNetwork(self.conf.get_config('network'))
        if (kwargs['parallel']):
            self.network = torch.nn.DataParallel(self.network)

        self.network = utils.get_cuda_ifavailable(self.network)

        debug_conf['loss'] =[self.conf.get_string('network.loss')]
        self.loss = utils.get_class(self.conf.get_string('network.loss.loss_type'))(batch_size=self.batch_size, **self.conf.get_config('network.loss.properties'))

        debug_conf['optimizer'] = [self.conf.get_string('train.optimizer')]
        debug_conf['optimizer_prop'] = [self.conf.get_string('train.optimizer_props')]
        self.optimizer =  utils.get_class(self.conf.get_string('train.optimizer'))(params=self.network.parameters(),**self.conf.get_config('train.optimizer_props'))

        self.lr = self.conf.get_float('train.optimizer_props.lr')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,150], gamma=0.5)
        debug_conf['train.scheduler'] = [self.conf.get_string('train.scheduler')]
        if (self.conf.get_string('train.scheduler') == 'our'):
            self.schedulerstep = self.scheduler.step
        elif (self.conf.get_string('train.scheduler') == 'trades'):
            self.schedulerstep = self.adjust_learning_rate
        elif (self.conf.get_string('train.scheduler') == 'trades_mnist'):
            self.schedulerstep = self.adjust_learning_rate_mnist
        elif (self.conf.get_string('train.scheduler') == 'our_mnist'):
            self.schedulerstep = self.adjust_learning_rate_our_mnist
        elif (self.conf.get_string('train.scheduler') == 'none'):
            self.schedulerstep = lambda x:None
        self.startepoch = 0
        if (kwargs['is_continue'] or False):
            old_checkpnts_dir = os.path.join(loadexpdir, kwargs['timestamp'], 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'model', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'optimizer', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, 'sched', str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])
            self.startepoch = saved_model_state["epoch"]

        self.adversary = None
        if self.conf.get_bool('network.adversarial_training.augment'):
            self.adversary = advertorch.attacks.LinfPGDAttack(self.network,
                                                              loss_fn=utils.get_class(self.conf.get_string(
                                                                  'network.adversarial_training.loss_type'))(
                                                                  batch_size=self.batch_size,
                                                                  **self.conf.get_config(
                                                                      'network.adversarial_training.loss_properties')),
                                                              eps=self.conf.get_float(
                                                                  'network.adversarial_training.eps'),
                                                              nb_iter=self.conf.get_int(
                                                                  'network.adversarial_training.nb_iter'),
                                                              eps_iter=self.conf.get_float(
                                                                  'network.adversarial_training.alpha'),
                                                              rand_init=True, clip_min=0.0, clip_max=1.0,
                                                              targeted=False)
        pd.DataFrame(debug_conf).to_csv(self.debug_log_conf_path)

    def save_learning_log(self,epoch_log,step_log):
        pd.DataFrame(epoch_log).to_csv(self.learning_log_epoch_path)
        pd.DataFrame(step_log).to_csv(self.learning_log_step_path)

    def save_checkpoints(self,epoch):
        for name in [epoch,'latest']:
            torch.save({"epoch": epoch, "model_state_dict": self.network.state_dict()},
                       os.path.join(self.model_path, '{0}.pth'.format(name)))
            torch.save({"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                       os.path.join(self.opt_path,'{0}.pth'.format(name)))
            torch.save({"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                       os.path.join(self.sched_path, '{0}.pth'.format(name)))
    def train_step(self,data,epoch):
        input = utils.get_cuda_ifavailable(data[0])
        target = utils.get_cuda_ifavailable(data[1])

        network_output = self.network(input,target=target,epoch=epoch)
        correct = self.network.get_correct(network_output, target).sum().item()
        seen_examples = target.shape[0]

        if self.adversary:
            with ctx_noparamgrad_and_eval(self.network):
                adv_inputs = self.adversary.perturb(input, target)
            network_output = self.network(adv_inputs, target=target)

        loss_res = self.loss(input=network_output, target=target)
        return utils.dict_to_nametuple("TrainStep",
                                       dict(loss_res=loss_res.loss,
                                            correct=correct,
                                            seen_examples=seen_examples,
                                            debug=dict(network_output.debug,**loss_res.debug)))

    def eval_step(self,data):
        input = utils.get_cuda_ifavailable(data[0])
        target = utils.get_cuda_ifavailable(data[1])

        clonemodel = copy.deepcopy(self.network.model)
        network_output = clonemodel(input)

        correct = self.network.get_correct(network_output,target).sum().item()
        seen_examples = target.shape[0]
        return utils.dict_to_nametuple("EvalStep",
                                       dict(correct=correct, seen_examples=seen_examples))

    def adjust_learning_rate_mnist(self, epoch):
        """decrease the learning rate"""
        lr = self.lr
        if epoch >= 55 and epoch < 75:
            lr = lr * 0.1
        elif epoch >= 75 and epoch < 90:
            lr = lr * 0.01
        elif epoch >= 90:
            lr = lr * 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_our_mnist(self, epoch):
        """decrease the learning rate"""
        lr = self.lr
        if epoch >= 40 and epoch < 100:
            lr = lr * 0.1
        elif epoch >= 100 and epoch < 140:
            lr = lr * 0.01
        elif epoch >= 140:
            lr = lr * 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate(self, epoch):
        """decrease the learning rate"""
        lr = self.lr
        if epoch >= 75 and epoch < 90:
            lr = lr * 0.1
        elif epoch >= 90 and epoch < 100:
            lr = lr * 0.01
        elif epoch >= 100:
            lr = lr * 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adv_step(self,data,attack,is_whitebox,epoch,data_index,**kwargs):
        input = utils.get_cuda_ifavailable(data[0])
        target = utils.get_cuda_ifavailable(data[1])
        kwargs['loss_fn'] = utils.get_class(kwargs['loss_fn'])(batch_size=self.batch_size, **kwargs['loss_fn_prop'])
        kwargs.pop('loss_fn_prop', None)

        if (is_whitebox):
            modeladv = self.network.model
        else:
            modeladv = self.blackbox_network.model
        modeladv = copy.deepcopy(modeladv)
        modeleval = copy.deepcopy(self.network.model)
        img_correct_ind = self.network.get_correct(modeleval(input), target=target)
        modeladv.zero_grad()
        adversary = utils.get_class(attack)(modeladv, **kwargs)
        adv_samples = adversary.perturb(input, target)
        network_output = modeleval(adv_samples)

        correct_ind = self.network.get_correct(network_output, target=target)
        correct = correct_ind.sum().item()

        seen_examples = target.shape[0]
        
        if (self.conf.get_bool('train.adv_plot_loss.enabled')): #and  correct < seen_examples and (~correct_ind).any()):

            with torch.no_grad():


                input = input[img_correct_ind]
                adv_samples = adv_samples[img_correct_ind]
                target = target[img_correct_ind]
                t = utils.get_cuda_ifavailable(
                    torch.tensor(
                        np.linspace(0,torch.norm(input[~correct_ind] - adv_samples[~correct_ind],p=np.inf,dim=[1,2,3]).cpu().numpy(),100).T))

                pnts = input[~correct_ind].unsqueeze(1) + t.reshape(t.shape[0], t.shape[1], 1, 1, 1) * (
                            adv_samples[~correct_ind].unsqueeze(1) - input[~correct_ind].unsqueeze(1)) / torch.norm(input[~correct_ind] - adv_samples[~correct_ind], p=np.inf,
                                                                                        dim=[1, 2, 3], keepdim=True).unsqueeze(1)



                loss_fn = utils.get_class(self.conf.get_string('train.adv_plot_loss.loss_fn'))(**self.conf.get_config('train.adv_plot_loss.loss_fn_props'))
                val_pnts = loss_fn(self.network.model(pnts.reshape([-1] + list(pnts.shape[2:]))),
                                                      target=target[~correct_ind].unsqueeze(1).repeat(1,100).reshape(-1)).reshape(-1,100)

                num = 1
                fig = make_subplots(rows=num, cols=1, subplot_titles=list(range(input.shape[0])))



                def gallery(array, ncols=3):
                    nindex, height, width, intensity = array.shape
                    nrows = nindex // ncols
                    assert nindex == nrows * ncols
                    # want result.shape = (height*nrows, width*ncols, intensity)
                    result = (array.reshape(nrows, ncols, height, width, intensity)
                              .swapaxes(1, 2)
                              .reshape(height * nrows, width * ncols, intensity))
                    return result

                for i,x,y,img in zip(range(num),t[:num],val_pnts[:num],pnts[:num]):
                    trace = go.Scatter(x=x.cpu().numpy(), y=y.cpu().numpy(), mode='lines', name='pgd_{0}'.format(i))
                    fig.add_trace(trace,i+1,1)
                    gal = gallery(img[np.linspace(0, 99, 10)].cpu().numpy().transpose([0, 2, 3, 1]), 5)
                    plt.imshow(gal.squeeze(), cmap='gray', vmin=0, vmax=1)
                    plt.savefig(os.path.join(self.plots_dir,'epoch_{0}_data_{1}.png'.format(epoch,data_index)))

                fig.update_layout(
                        autosize=False,
                        width=800,
                        height=800)
                offline.plot(fig,filename=os.path.join(self.plots_dir,'epoch_{0}_data_{1}.html'.format(epoch,data_index)),auto_open=False)



        return utils.dict_to_nametuple("AdvStep",
                                       dict(correct=correct, seen_examples=seen_examples))


    def run(self):

        loss_log_step = []
        loss_log_epoch = []
        correct_real_mean_loss_log = []
        correct_mean_loss_log = []
        effective_distance_log = []
        batch_margin_mean_loss_log = []
        lr_log_epoch = []
        timing_log = []
        train_accuracy_log = []
        test_accuracy_log = []
        step_debug = None
        adv_conf = self.conf.get_config('adversarial.attacks')
        test_adv_accuracy_log = {k + '_accuracy':[] for k in adv_conf.keys()}


        for epoch in range(self.startepoch,self.nepochs):
            start = time.time()

            logging.info("epoch {}...".format(epoch))

            batch_loss = 0
            batch_correct_realmean_loss = 0.0
            batch_correct_mean_loss = 0.0
            batch_margin_mean_loss = 0.0
            batch_effective_mean_distance = 0.0
            correct = 0
            seen_examples = 0

            # Train
            for data_index,data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                step_result = self.train_step(data,epoch)
                if (step_debug is None):
                    step_debug = step_result.debug
                else:
                    for k in step_debug.keys():
                        step_debug[k] = step_debug[k] + step_result.debug[k]
                correct += step_result.correct
                seen_examples += step_result.seen_examples
                self.optimizer.zero_grad()
                if (step_result.loss_res.requires_grad):
                    step_result.loss_res.backward()
                loss_log_step.append(step_result.loss_res.item())
                self.optimizer.step()

                batch_loss += step_result.loss_res.item()
                if 'correct_points_loss_real_mean' in step_result.debug.keys():
                    if (step_result.debug['correct_points_loss_real_mean'][0]):
                        batch_correct_realmean_loss += step_result.debug['correct_points_loss_real_mean'][0]
                        batch_correct_mean_loss += step_result.debug['correct_points_loss_mean'][0]
                if ('correct_effective_zero_points_distance_mean' in step_result.debug.keys()):
                    if (step_result.debug['correct_effective_zero_points_distance_mean'][0]):
                        batch_effective_mean_distance += step_result.debug['correct_effective_zero_points_distance_mean'][0]

                if ('margin_loss_clamped_mean' in step_result.debug.keys()):
                    if (step_result.debug['margin_loss_clamped_mean'][0]):
                        batch_margin_mean_loss += step_result.debug['margin_loss_clamped_mean'][0]

                logging.debug("expname : {0} , "
                              "timestamp: {1} , "
                              "epoch : {2} , "
                              "data_index : {3} , "
                              "loss : {4}".format(self.expname,self.timestamp,epoch,data_index,step_result.loss_res.item()))


            end = time.time()
            seconds_elapsed_epoch = end - start
            lr_log_epoch.append(self.optimizer.param_groups[0]["lr"])
            timing_log.append(seconds_elapsed_epoch)
            loss_log_epoch.append(batch_loss/self.ds_len)
            correct_real_mean_loss_log.append(batch_correct_realmean_loss/self.ds_len)
            correct_mean_loss_log.append(batch_correct_mean_loss / self.ds_len)
            effective_distance_log.append(batch_effective_mean_distance/self.ds_len)
            batch_margin_mean_loss_log.append(batch_margin_mean_loss/self.ds_len)
            train_accuracy_log.append(float(correct)/seen_examples)

            if (epoch % self.eval_frequency == 0):
                # test
                self.network.eval()
                correct = 0
                seen_examples = 0

                attack_res = {k:dict(correct=0) for k in adv_conf.keys()}
                for data_index, data in enumerate(self.test_dataloader):
                    eval_res = self.eval_step(data)
                    correct += eval_res.correct
                    seen_examples += eval_res.seen_examples

                    for attack_name in adv_conf.keys():
                        attack_conf = adv_conf.get_config(attack_name)
                        if attack_conf.get_bool("is_whitebox") or self.conf.get_bool('adversarial.blackbox.enabled'):
                            attack_res[attack_name]['correct'] += self.adv_step(data,attack_conf.get_string('attack'),
                                                                                attack_conf.get_bool('is_whitebox'),
                                                                                epoch,data_index,
                                                                                **attack_conf.get_config('props')).correct
                    logging.debug("expname : {0} , "
                                  "timestamp: {1} , "
                                  "epoch : {2} , "
                                  "data_index : {3} , "
                                  "accuracy : {4}".format(self.expname, self.timestamp, epoch, data_index,
                                                      correct/seen_examples))

                self.network.train()
                self.network.bn_mode = 'org'
                for attack_name in adv_conf.keys():
                    test_adv_accuracy_log[attack_name + "_accuracy"].append(float(attack_res[attack_name]['correct'])/seen_examples)
                test_accuracy_log.append(float(correct)/seen_examples)
            else:
                for attack_name in adv_conf.keys():
                    test_adv_accuracy_log[attack_name + "_accuracy"].append(None)
                test_accuracy_log.append(None)


            epoch_log = dict(epoch=range(self.startepoch, epoch + 1),
                 loss_epoch=loss_log_epoch,
                 correct_real_mean_loss_log = correct_real_mean_loss_log,
                 correct_mean_loss_log = correct_mean_loss_log,
                 effective_distance_log = effective_distance_log,
                 batch_margin_mean_loss_log=batch_margin_mean_loss_log,
                 train_accuracy=train_accuracy_log,
                 test_accuracy=test_accuracy_log,
                 time_elapsed=timing_log,
                 lr_epoch=lr_log_epoch,
                 **test_adv_accuracy_log)
            step_log = dict(loss_step=loss_log_step,**step_debug)

            self.save_learning_log(epoch_log=epoch_log,
                                   step_log=step_log)
            if (epoch) % 10 == 0:
                self.save_checkpoints(epoch)
            self.schedulerstep(epoch)


