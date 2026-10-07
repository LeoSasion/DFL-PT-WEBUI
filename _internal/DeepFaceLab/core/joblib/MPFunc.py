from .MPRemote import MPRemote


class MPFunc(MPRemote):
    def __init__(self, func, *, rpc_timeout_sec=300.0):
        self.func = func
        super().__init__(rpc_timeout_sec=rpc_timeout_sec)

    def _execute(self, args, kwargs):
        return self.func(*args, **kwargs)
