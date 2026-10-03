"""DFL FP32 updates, serializable state, and optional CPU optimizer storage."""
import math
import torch


def optimizer_settings(config):
    """Validate optimizer options independently of the engine's mutable config."""
    lr = config.lr
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 1e-8 <= lr <= 1e-2:
        raise ValueError('lr must be finite and in [1e-8, 1e-2]')
    for name in ('adabelief', 'clipgrad', 'optimizer_on_cpu'):
        if type(getattr(config, name, False)) is not bool:
            raise ValueError(f'{name} must be a boolean')
    if config.lr_dropout not in ('n', 'y', 'cpu'):
        raise ValueError('lr_dropout must be n, y or cpu')
    enabled = config.lr_dropout != 'n'
    return dict(lr=lr, adabelief=config.adabelief,
                dropout=.3 if enabled else 1., cosine=500 if enabled else 0,
                clipnorm=1. if config.clipgrad else 0.,
                optimizer_on_cpu=getattr(config, 'optimizer_on_cpu', False),
                dropout_on_cpu=config.lr_dropout == 'cpu')


class MEOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, config):
        super().__init__(parameters, dict(**optimizer_settings(config), iteration=0))

    def _state_device(self, group, parameter):
        return torch.device('cpu') if group['optimizer_on_cpu'] else parameter.device

    def configure(self, config):
        """Apply mutable settings without resetting moments, steps, or RNG.

        The engine must explicitly recreate the optimizer to change AdaBelief
        after updates. Allocate all state transfers before committing settings.
        """
        settings = optimizer_settings(config)
        self.validate_state()
        for group in self.param_groups:
            if settings['adabelief'] != group['adabelief'] and (
                    group['iteration'] or any(self.state.get(p) for p in group['params'])):
                raise ValueError('Changing adabelief requires an explicit optimizer reset')
        transferred = []
        for group in self.param_groups:
            for parameter in group['params']:
                device = torch.device('cpu') if settings['optimizer_on_cpu'] else parameter.device
                for name, value in self.state.get(parameter, {}).items():
                    transferred.append((parameter, name, value.to(device=device)))
        for parameter, name, value in transferred:
            self.state[parameter][name] = value
        for group in self.param_groups:
            group.update(settings)

    def load_state_dict(self, state_dict):
        # torch.optim casts floating state to each parameter's device. Restore
        # the serialized CPU placement explicitly after loading an offload model.
        if (not isinstance(state_dict, dict) or not isinstance(state_dict.get('state'), dict)
                or not isinstance(state_dict.get('param_groups'), list)
                or any(not isinstance(group, dict) for group in state_dict['param_groups'])):
            raise ValueError('Invalid optimizer state dictionary')
        upgraded = dict(state_dict)
        upgraded['param_groups'] = [dict(group) for group in state_dict['param_groups']]
        for group in upgraded['param_groups']:
            group.setdefault('optimizer_on_cpu', False)
            group.setdefault('dropout_on_cpu', False)
        if len(upgraded['param_groups']) != len(self.param_groups):
            raise ValueError('Optimizer parameter group count mismatch')
        bound, groups = {}, []
        for saved, current in zip(upgraded['param_groups'], self.param_groups):
            identifiers = saved.get('params')
            if not isinstance(identifiers, list) or len(identifiers) != len(current['params']):
                raise ValueError('Optimizer parameter count mismatch')
            for identifier, parameter in zip(identifiers, current['params']):
                if type(identifier) is not int or identifier in bound:
                    raise ValueError('Invalid or duplicate optimizer parameter identifier')
                bound[identifier] = parameter
            groups.append(dict(saved, params=current['params']))
        if any(type(identifier) is not int or identifier not in bound for identifier in upgraded['state']):
            raise ValueError('Optimizer state contains an unknown parameter')
        # Validate source dtype before Torch can silently cast a corrupt FP64 or
        # integer moment back to FP32. A temporary view does not mutate this opt.
        candidate = object.__new__(MEOptimizer)
        candidate.param_groups = groups
        candidate.state = {bound[identifier]:value for identifier,value in upgraded['state'].items()}
        candidate.validate_state(check_device=False)
        super().load_state_dict(upgraded)
        for group in self.param_groups:
            for parameter in group['params']:
                device = self._state_device(group, parameter)
                for name, value in self.state.get(parameter, {}).items():
                    if isinstance(value, torch.Tensor):
                        self.state[parameter][name] = value.to(device=device)
        self.validate_state()

    def validate_state(self, iteration=None, require_complete=False, check_device=True):
        """Reject malformed state; iteration is the number of optimizer updates.

        Missing state is allowed for parameters that have never received a
        gradient (for example a previously frozen LIAE inter_AB). Callers can
        request complete state after checking their trainable parameter set.
        """
        if iteration is not None and (type(iteration) is not int or iteration < 0):
            raise ValueError('Invalid optimizer update count')
        for group in self.param_groups:
            if type(group.get('iteration')) is not int or group['iteration'] < 0:
                raise ValueError('Invalid optimizer iteration')
            if iteration is not None and group['iteration'] != iteration:
                raise ValueError('Optimizer/model update count mismatch')
            for name in ('adabelief', 'optimizer_on_cpu', 'dropout_on_cpu'):
                if type(group.get(name)) is not bool:
                    raise ValueError(f'Invalid optimizer {name}')
            if (type(group.get('dropout')) is not float or group['dropout'] not in (.3, 1.)
                    or type(group.get('cosine')) is not int or group['cosine'] not in (0, 500)):
                raise ValueError('Invalid optimizer dropout/cosine settings')
            if (group['dropout'] == 1.) != (group['cosine'] == 0) or (group['dropout_on_cpu'] and group['dropout'] == 1.):
                raise ValueError('Inconsistent optimizer dropout settings')
            if type(group.get('clipnorm')) is not float or group['clipnorm'] not in (0., 1.):
                raise ValueError('Invalid optimizer clipnorm')
            lr = group.get('lr')
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 1e-8 <= lr <= 1e-2:
                raise ValueError('Invalid optimizer learning rate')
            expected = {'momentum', 'variance'} if group['adabelief'] else {'variance'}
            for parameter in group['params']:
                if parameter.dtype != torch.float32:
                    raise ValueError('ME optimizer requires FP32 parameters')
                state = self.state.get(parameter, {})
                if not isinstance(state, dict):
                    raise ValueError('Invalid optimizer parameter state')
                if not state:
                    if require_complete and parameter.requires_grad and group['iteration']:
                        raise ValueError('Incomplete optimizer state')
                    continue
                if set(state) != expected:
                    raise ValueError('Invalid optimizer tensor names')
                for value in state.values():
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or value.dtype != torch.float32 or (check_device and value.device != self._state_device(group, parameter))
                            or not bool(torch.isfinite(value).all())):
                        raise ValueError('Invalid optimizer tensor state')
                if bool((state['variance'] < 0).any()):
                    raise ValueError('Optimizer variance must be nonnegative')
        parameter_ids = {id(p) for group in self.param_groups for p in group['params']}
        if any(id(p) not in parameter_ids for p in self.state):
            raise ValueError('Optimizer state contains an unknown parameter')

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not supported')
        for group in self.param_groups:
            params = [p for p in group['params'] if p.grad is not None]
            if group['clipnorm']:
                torch.nn.utils.clip_grad_norm_(params, group['clipnorm'], error_if_nonfinite=True)
            lr = group['lr']
            if group['cosine']:
                lr *= (math.cos(group['iteration']*2*math.pi/group['cosine'])+1)/2
            for p in params:
                if p.dtype != torch.float32 or p.grad.dtype != torch.float32 or p.grad.is_sparse:
                    raise ValueError('ME optimizer requires dense FP32 parameters and gradients')
                device = self._state_device(group, p)
                g, state = p.grad.to(device=device), self.state[p]
                if not state:
                    state['variance'] = torch.zeros_like(p, device=device)
                    if group['adabelief']:
                        state['momentum'] = torch.zeros_like(p, device=device)
                v = state['variance']
                if group['adabelief']:
                    m = state['momentum']
                    m.mul_(.9).add_(g, alpha=.1)
                    v.mul_(.999).addcmul_(g-m, g-m, value=.001)
                    delta = m / (v.sqrt()+1e-6)
                else:
                    v.mul_(.9).addcmul_(g, g, value=.1)
                    delta = g / (v.sqrt()+1e-6)
                if group['dropout'] != 1.:
                    if group['dropout_on_cpu']:
                        mask = torch.rand(p.shape, dtype=p.dtype, device='cpu') < group['dropout']
                    else:
                        # Preserve the existing y-mode parameter-device RNG.
                        mask = torch.rand_like(p) < group['dropout']
                    delta *= mask.to(device=device)
                p.add_(delta.to(device=p.device), alpha=-lr)
            group['iteration'] += 1
