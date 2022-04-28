'''import matplotlib.pyplot as plt
import numpy as np
import heapq
import torch


def show_images(images, cols=5, titles=None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def get_features(model, dirs, **kwargs):
    dl = augdataset.get_test_dl(dirs=dirs, **kwargs)
    model.eval()
    all_features = []
    all_fps = []
    for images, fps in dl:
        images = images.cuda()
        features = model(images)
        features = features.detach().cpu().numpy()
        all_features.extend(list(features))
        all_fps.extend(fps)
    return all_fps, all_features


def get_byol_features(model, dirs, **kwargs):
    dl = byoldataset.get_test_dl(dirs=dirs, **kwargs)
    model.eval()
    all_features = []
    all_fps = []
    for images, fps in dl:
        images = images.cuda()
        features, _ = model(images, return_embedding=True)
        features = features.detach().cpu().numpy()
        all_features.extend(list(features))
        all_fps.extend(fps)
    return all_fps, all_features


from torch.optim.optimizer import Optimizer


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If True, prints a message to stdout for
            each update. Default: False.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """
    def __init__(self,
                 optimizer,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 verbose=False,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = []
        self.num_bad_epochs = None
        self.mode_worse = []  # the worse value for the chosen mode
        #         self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        #         self._init_is_better(mode=mode, threshold=threshold,
        #                              threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    """ Status codes to be returned by ReduceLROnPlateau.step() """
    STATUS_WAITING = 1
    STATUS_UPDATED_BEST = 2
    STATUS_REDUCED_LR = 3

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
        else:
            self.num_bad_epochs += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            return self.STATUS_UPDATED_BEST
        else:
            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                return self.STATUS_REDUCED_LR
            else:
                return self.STATUS_WAITING

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, current, best):

        if best is []:
            return True
        best = [best]
        heapq.heappush(best, current)
        if heapq.heappop(best) != current:
            return True
        return False


#     def _init_is_better(self, mode, threshold, threshold_mode):
#         heapq.heappush(self.best_metrics, metrics)
#         if heapq.heappop(self.best_metrics) != metrics:
#             self.save_checkpoint(model)
#             self.counter = 0
#         if mode not in {'min', 'max'}:
#             raise ValueError('mode ' + mode + ' is unknown!')
#         if threshold_mode not in {'rel', 'abs'}:
#             raise ValueError('threshold mode ' + mode + ' is unknown!')
#         if mode == 'min' and threshold_mode == 'rel':
#             rel_epsilon = 1. - threshold
#             self.is_better = lambda a, best: a < best * rel_epsilon
#             self.mode_worse = float('Inf')
#         elif mode == 'min' and threshold_mode == 'abs':
#             self.is_better = lambda a, best: a < best - threshold
#             self.mode_worse = float('Inf')
#         elif mode == 'max' and threshold_mode == 'rel':
#             rel_epsilon = threshold + 1.
#             self.is_better = lambda a, best: a > best * rel_epsilon
#             self.mode_worse = -float('Inf')
#         else:  # mode == 'max' and epsilon_mode == 'abs':
#             self.is_better = lambda a, best: a > best + threshold
#             self.mode_worse = -float('Inf')


class ReduceLROnPlateauWithBacktrack(ReduceLROnPlateau):
    """Load training state from the best epoch and reduce learning
    rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced, and the best state is
    loaded.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        model (torch.nn.Module): Model to be saved and backtracked.
        filename (str): Directory to save the best state.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If True, prints a message to stdout for
            each update. Default: False.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateauWithBacktrack(optimizer, model)
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """
    def __init__(self, optimizer, model, filename, warmup_steps, **kwargs):
        self.filename = filename
        self.model = model
        self.warmup_steps = warmup_steps + 1
        self.warmup_step_curr = 1
        self.warmup_phase = False
        self.init_lrs_for_warmup = []

        super(ReduceLROnPlateauWithBacktrack,
              self).__init__(optimizer=optimizer, **kwargs)
        for group in self.optimizer.param_groups:
            self.init_lrs_for_warmup.append(group['lr'])

    def step(self, metrics, epoch=None):
        if self.warmup_step_curr < self.warmup_steps + 1:
            self.warmup_phase = True
            factor = float(self.warmup_step_curr) / float(
                max(1, self.warmup_steps))
            #             print('warmup with factor ', factor)
            for init_lr, group in zip(self.init_lrs_for_warmup,
                                      self.optimizer.param_groups):
                group['lr'] = init_lr * (0.0001)**(1 - factor)
            self.warmup_step_curr += 1
            return
        self.warmup_phase = False
        #         print(f'stepping on metric: {metrics}')
        status = super(ReduceLROnPlateauWithBacktrack,
                       self).step(metrics=metrics, epoch=epoch)
        if status == self.STATUS_UPDATED_BEST:
            #             print('saving')
            torch.save(
                {
                    'model': self.model.state_dict(),
                    'optim': self.optimizer.state_dict()
                }, self.filename)
        elif status == self.STATUS_REDUCED_LR:
            new_lrs = [group['lr'] for group in self.optimizer.param_groups]
            backtrack_dict = torch.load(self.filename)
            self.optimizer.load_state_dict(backtrack_dict['optim'])
            self.model.load_state_dict(backtrack_dict['model'])
            # Note that new_lr might not be saved_lr * gamma
            for new_lr, group in zip(new_lrs, self.optimizer.param_groups):
                group['lr'] = new_lr
'''