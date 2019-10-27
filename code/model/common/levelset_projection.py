import torch
import numpy as np
import utils.general as utils
import abc


class LevelSetProjection(metaclass=abc.ABCMeta):
    def __init__(self,proj_tolerance,proj_max_iters):
        self.proj_tolerance = proj_tolerance
        self.proj_max_iters = proj_max_iters

    @abc.abstractmethod
    def project_points(self,points_init,network,latent,levelset):
        pass

class ZeroInPgdDirection(LevelSetProjection):
    def project_points(self, points_init, model, loss, target,stepsize,epsilon_correct,epsilon_wrong,eps_base,init):
        model.eval()
        model.zero_grad()

        output = model(points_init).detach()

        output = loss(output, target=target)
        eval_prev = output.detach()
        correct = output < 0

        epsilon = epsilon_correct * correct.float() + epsilon_wrong * (~correct).float()
        epsilon = epsilon.view([epsilon.shape[0]] + [1] * len(points_init.shape[1:]))
        stepsize = epsilon_correct * stepsize / eps_base * correct.float() + stepsize * (~correct).float()
        stepsize = stepsize.view([stepsize.shape[0]] + [1] * len(points_init.shape[1:]))


        if (init == 'box'):
            x_adv = points_init + epsilon * (2 * torch.rand_like(points_init) - 1)
        elif(init == 'ball'):
            x_adv = points_init.detach() + 0.001 * utils.get_cuda_ifavailable(torch.randn(points_init.shape))
        else:
            raise NotImplementedError

        x_adv = x_adv.clamp(0,1)
        correct_sign = (2*correct.float() - 1).detach()

        for i in range(self.proj_max_iters):
            x_adv.detach_().requires_grad_()
            with torch.enable_grad():
                eval = correct_sign * loss(model(x_adv),target=target)
                evalsum = eval.sum()
            grad = torch.autograd.grad(evalsum, [x_adv])[0].detach()

            x_adv.detach_()
            x_adv = (x_adv + stepsize * torch.sign(grad.detach()))

            x_adv = torch.min(torch.max(x_adv, points_init - epsilon), points_init + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            model.zero_grad()

        filter_nonsignchanged = (torch.sign(loss(model(x_adv),target=target) * eval_prev) == 1).detach()
        model.zero_grad()

        zero_pnts = {'levelset_points':None}
        debug = {'correct_count':[correct.sum().item()],
                'correct_number_of_pgd_effective':[None],
                'correct_find_zero_number_of_iters':[None],
                'correct_zero_points_eval_mean':[ None],
                'correct_zero_points_eval_max':[None],
                'correct_zero_points_grad_norm_mean':[None],
                'correct_zero_points_grad_norm_max':[None],
                'correct_zero_points_grad_norm_min':[None],
                'correct_effective_zero_points_distance_mean':[None],
                'correct_effective_zero_points_distance_max':[None],
                'correct_effective_zero_points_distance_std':[None],
                'wrong_count':[(~correct).sum().item()],
                'wrong_number_of_pgd_effective':[None],
                'wrong_find_zero_number_of_iters':[None],
                'wrong_zero_points_eval_mean':[None],
                'wrong_zero_points_eval_max':[None],
                'wrong_zero_points_grad_norm_mean':[None],
                'wrong_zero_points_grad_norm_max':[None],
                'wrong_zero_points_grad_norm_min':[None],
                'wrong_effective_zero_points_distance_mean':[None],
                'wrong_effective_zero_points_distance_max':[None],
                'wrong_effective_zero_points_distance_std':[None],
                'effective_search_zeros':[None]}
        if (~filter_nonsignchanged).sum().item() > 0:

            zero_pnts = RegularFalsiProjection(self.proj_tolerance,self.proj_max_iters).project_points(points_init[~filter_nonsignchanged],
                                                                                                         x_adv[~filter_nonsignchanged],
                                                                                                         model,
                                                                                                         loss,
                                                                                                             target[~filter_nonsignchanged])
            eval = zero_pnts['network_eval_on_levelset_points']
            tol = 1.0e-1
            debug['effective_search_zeros'] = [(eval.abs() < tol).sum().item()]
            filter_nonsignchanged[~filter_nonsignchanged] = ~(eval.abs() < tol)
            if (eval.abs() < tol).sum().item() == 0:
                zero_pnts['network_eval_on_levelset_points'] = None
                zero_pnts['levelset_points_Dx'] = None
                zero_pnts['levelset_points'] = None
            else:
                zero_pnts['network_eval_on_levelset_points'] = eval[eval.abs() < tol]
                zero_pnts['levelset_points_Dx'] = zero_pnts['levelset_points_Dx'][eval.abs() < tol]
                zero_pnts['levelset_points'] = zero_pnts['levelset_points'][eval.abs() < tol]

                debug['correct_number_of_pgd_effective'] = [((~filter_nonsignchanged) & correct).sum().item()]
                debug['correct_find_zero_number_of_iters'] = [None]
                debug['correct_zero_points_eval_mean'] = [zero_pnts['network_eval_on_levelset_points'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]].abs().mean().item() if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_zero_points_eval_max'] = [zero_pnts['network_eval_on_levelset_points'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]].abs().max().item() if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_zero_points_grad_norm_mean'] = [torch.norm(zero_pnts['levelset_points_Dx'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]],p=2,dim=[1,2,3]).mean().item() if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_zero_points_grad_norm_max'] = [torch.norm(zero_pnts['levelset_points_Dx'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]],p=2,dim=[1,2,3]).max().item() if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_zero_points_grad_norm_min'] = [torch.norm(zero_pnts['levelset_points_Dx'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]],p=2,dim=[1,2,3]).min().item() if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_effective_zero_points_distance_mean'] = [torch.norm(zero_pnts['levelset_points'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]] -
                                                                                   points_init[(~filter_nonsignchanged) & correct],
                                                                                   p=np.inf,
                                                                                   dim=[1,2,3]).mean().item() if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_effective_zero_points_distance_max'] = [torch.norm(zero_pnts['levelset_points'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]] -
                                                                                  points_init[(~filter_nonsignchanged) & correct],p=np.inf,dim=[1,2,3]).max().item()
                                                                       if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['correct_effective_zero_points_distance_std'] = [torch.norm(zero_pnts['levelset_points'][((~filter_nonsignchanged) & correct)[~filter_nonsignchanged]] -
                                                                                  points_init[(~filter_nonsignchanged) & correct],p=np.inf,dim=[1,2,3]).std().item()
                                                                       if debug['correct_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_number_of_pgd_effective'] = [((~filter_nonsignchanged) & (~correct)).sum().item()]
                debug['wrong_find_zero_number_of_iters'] = [None]
                debug['wrong_zero_points_eval_mean'] = [
                    zero_pnts['network_eval_on_levelset_points'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]].abs().mean().item()
                    if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_zero_points_eval_max'] = [
                    zero_pnts['network_eval_on_levelset_points'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]].abs().max().item()
                    if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_zero_points_grad_norm_mean'] = [torch.norm(zero_pnts['levelset_points_Dx'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]],
                                                                        p=2,dim=[1, 2, 3]).mean().item()
                                                             if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_zero_points_grad_norm_max'] = [torch.norm(zero_pnts['levelset_points_Dx'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]],
                                                                       p=2,dim=[1, 2, 3]).max().item()
                                                            if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_zero_points_grad_norm_min'] = [torch.norm(zero_pnts['levelset_points_Dx'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]],
                                                                       p=2,dim=[1, 2, 3]).min().item()
                                                            if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_effective_zero_points_distance_mean'] = [torch.norm(zero_pnts['levelset_points'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]] -
                                                                                 points_init[(~filter_nonsignchanged) & (~correct)], p=np.inf, dim=[1, 2, 3]).mean().item()
                                                                      if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_effective_zero_points_distance_max'] = [torch.norm(zero_pnts['levelset_points'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]] -
                                                                                points_init[(~filter_nonsignchanged) & (~correct)], p=np.inf, dim=[1, 2, 3]).max().item()
                                                                     if debug['wrong_number_of_pgd_effective'][0] > 0 else None]
                debug['wrong_effective_zero_points_distance_std'] = [torch.norm(zero_pnts['levelset_points'][((~filter_nonsignchanged) & (~correct))[~filter_nonsignchanged]] -
                                                                                points_init[(~filter_nonsignchanged) & (~correct)], p=np.inf, dim=[1, 2, 3]).std().item()
                                                                     if debug['wrong_number_of_pgd_effective'][0] > 0 else None]



        return dict(**zero_pnts,sign_changed=~filter_nonsignchanged,zero_pnts_sign=-correct_sign[~filter_nonsignchanged],debug=debug)

class GenNewtonProjection(LevelSetProjection):
    def project_points(self,points_init,model,latent):
        curr_projection = points_init
        curr_projection.requires_grad_(True)

        if (not latent is None):
            latent = latent.unsqueeze(2).repeat(1, 1, self.num_points)
            curr_projection = torch.cat((latent, curr_projection), 1).contiguous()

        trials = 0
        model.eval()
        while (True):
            model.zero_grad()

            network_eval = model(curr_projection)

            network_eval_sum = network_eval.sum()
            network_eval_sum.backward(retain_graph=True)
            grad = curr_projection.grad
            network_eval = network_eval.detach()

            if (torch.max(torch.abs(network_eval)) > self.proj_tolerance and trials < self.proj_max_iters):

                sum_square_grad = torch.sum(grad ** 2, dim=list(range(len(grad.shape)))[1:], keepdim=True)
                curr_projection = curr_projection - network_eval.view([network_eval.shape[0]] + [1]*(len(grad.shape) - 1)) * (grad / sum_square_grad.clamp_min(1.0e-6))
                curr_projection = curr_projection.detach_().requires_grad_(True)

                if (not latent is None):
                    curr_projection = torch.cat((latent, curr_projection), 1).contiguous()

            else:
                break

            trials = trials + 1
        if (self.proj_max_iters > 0):
            print("iteration : {0} , max : {1} , min {2} , mean {3} , std {4}"
                  .format(trials, torch.max(torch.abs(network_eval), dim=0)[0].mean(),
                          torch.min(torch.abs(network_eval), dim=0)[0].mean(),
                          torch.mean(torch.abs(network_eval), dim=0).mean(),
                          torch.std(torch.abs(network_eval), dim=0).mean()))

        model.train()
        model.zero_grad()
        curr_projection.detach_()

        return {'levelset_points':curr_projection.detach(),
               'levelset_points_Dx':grad.detach(),
               'network_eval_on_levelset_points':network_eval.detach()}


class SearchProjection(LevelSetProjection):

    def project_points(self,points_init,x_adv,model,loss,target):
        model.eval()
        sign_init = torch.sign(loss(model(points_init), target)).detach()
        found_zero = False
        start_t = utils.get_cuda_ifavailable(torch.zeros([points_init.shape[0], 1, 1, 1]))
        end_t = utils.get_cuda_ifavailable(torch.ones([points_init.shape[0], 1, 1,1]))  # torch.norm(pgd_correct_effective_samples - correct_samples,p=np.inf,dim=[1,2,3],keepdim=True)
        iter = 0
        while (not found_zero):
            middle_t = (start_t + end_t) / 2
            zero_pnts = (x_adv + middle_t * (
                    points_init - x_adv)).detach().requires_grad_(
                True)
            model.zero_grad()
            eval = loss(model(zero_pnts), target=target)
            # start_t[(-sign_init) * eval > 0)] = middle_t[((-sign_init)*eval > 0)]
            # end_t[(eval < 0)] = middle_t[(eval < 0)]

            start_t[(sign_init*eval) < 0] = middle_t[(sign_init*eval) < 0]
            end_t[(sign_init*eval) > 0] = middle_t[(sign_init * eval) > 0]


            found_zero = ((torch.abs(eval) > 1.0e-5).sum().item() == 0 or iter >= self.proj_max_iters)
            iter = iter + 1

        eval.sum().backward()
        grad = zero_pnts.grad
        model.train()
        model.zero_grad()

        return {'levelset_points':zero_pnts.detach(),
               'levelset_points_Dx':grad.detach(),
               'network_eval_on_levelset_points':eval.detach()}

class RegularFalsiProjection(LevelSetProjection):

    def project_points(self,points_init,x_adv,model,loss,target):
        model.eval()
        sign_init = torch.sign(loss(model(points_init), target)).detach()
        found_zero = False
        left_t = utils.get_cuda_ifavailable(torch.zeros([points_init.shape[0], 1, 1, 1]))
        right_t = utils.get_cuda_ifavailable(torch.ones([points_init.shape[0], 1, 1,1]))  # torch.norm(pgd_correct_effective_samples - correct_samples,p=np.inf,dim=[1,2,3],keepdim=True)
        iter = 0
        with torch.no_grad():
            while (not found_zero):
                right_val = loss(model(x_adv + right_t * (points_init - x_adv)),target=target).view(points_init.shape[0], 1, 1, 1)
                left_val = loss(model(x_adv + left_t * (points_init - x_adv)),target=target).view(points_init.shape[0], 1, 1, 1)
                next_t = (left_t * right_val - right_t * left_val) / (right_val - left_val)
                zero_pnts = (x_adv + next_t * (points_init - x_adv))
                model.zero_grad()
                eval = loss(model(zero_pnts), target=target)
                # start_t[(-sign_init) * eval > 0)] = middle_t[((-sign_init)*eval > 0)]
                # end_t[(eval < 0)] = middle_t[(eval < 0)]

                # left_t[(sign_init*eval) < 0] = next_t[(sign_init*eval) < 0]abcd

                # right_t[(sign_init*eval) > 0] = next_t[(sign_init * eval) > 0]
                right_t[(sign_init*eval) > 0] = next_t[(sign_init*eval)>0]
                left_t[(sign_init*eval) < 0] = next_t[(sign_init*eval) < 0]



                found_zero = ((torch.abs(eval) > 1.0e-5).sum().item() == 0 or iter >= self.proj_max_iters)
                iter = iter + 1

        zero_pnts.detach_().requires_grad_(True)
        model.zero_grad()
        eval = loss(model(zero_pnts), target=target)
        eval.sum().backward()
        grad = zero_pnts.grad
        model.train()
        model.zero_grad()

        return {'levelset_points':zero_pnts.detach(),
               'levelset_points_Dx':grad.detach(),
               'network_eval_on_levelset_points':eval.detach()}
