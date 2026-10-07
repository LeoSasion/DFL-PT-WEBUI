from .MPRemote import MPRemote


class MPClassFuncOnDemand(MPRemote):
    def __init__(self, class_handle, class_func_name, *, rpc_timeout_sec=300.0, **class_kwargs):
        self.class_handle = class_handle
        self.class_func_name = class_func_name
        self.class_kwargs = class_kwargs
        self.class_func = None
        super().__init__(rpc_timeout_sec=rpc_timeout_sec)

    def _execute(self, args, kwargs):
        if self.class_func is None:
            self.class_func = getattr(self.class_handle(**self.class_kwargs), self.class_func_name)
        return self.class_func(*args, **kwargs)
