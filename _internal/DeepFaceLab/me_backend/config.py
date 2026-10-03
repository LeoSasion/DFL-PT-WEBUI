from dataclasses import asdict, dataclass, fields, replace
import math
import re


@dataclass(frozen=True)
class MEConfig:
    # Architecture is explicit; runtime/training options never rename weights.
    archi: str = 'liae-ud'
    resolution: int = 128
    ae_dims: int = 256
    e_dims: int = 64
    d_dims: int = 64
    d_mask_dims: int = 22
    face_type: str = 'f'
    batch_size: int = 4
    masked_training: bool = True
    eyes_prio: bool = False
    mouth_prio: bool = False
    loss_function: str = 'SSIM'
    background_power: float = 0.0
    face_style_power: float = 0.0
    bg_style_power: float = 0.0
    blur_out_mask: bool = False
    lr: float = 5e-5
    adabelief: bool = True
    lr_dropout: str = 'n'
    clipgrad: bool = False
    random_warp: bool = True
    random_src_flip: bool = False
    random_dst_flip: bool = True
    random_hsv_power: float = 0.0
    use_rg: bool = False
    use_fp16: bool = False
    optimizer_on_cpu: bool = False
    gan_power: float = 0.0
    gan_patch_size: int = 16
    gan_dims: int = 16
    gan_smoothing: float = 0.1
    gan_noise: float = 0.0
    true_face_power: float = 0.0
    pretrain: bool = False
    uniform_yaw: bool = False
    random_downsample: bool = False
    random_noise: bool = False
    random_blur: bool = False
    random_jpeg: bool = False
    random_color: bool = False
    ct_mode: str = 'none'
    data_workers: int = 0
    retraining_samples: bool = False
    retraining_capacity: int = 128
    retraining_every: int = 16

    def __post_init__(self):
        if not isinstance(self.archi, str) or not re.fullmatch(r'(df|liae)(-[udtc]+)?', self.archi):
            raise ValueError('archi must be df or liae with explicit u/d/t/c options')
        options = self.archi.partition('-')[2]
        if len(set(options)) != len(options):
            raise ValueError('Architecture options must not repeat')
        if type(self.resolution) is not int or not 64 <= self.resolution <= 640 or self.resolution % 32:
            raise ValueError('resolution must be a multiple of 32 between 64 and 640')
        for name, minimum, maximum in [('ae_dims',32,1024),('e_dims',16,256),('d_dims',16,256),('d_mask_dims',16,256),('batch_size',1,256),
                                      ('gan_patch_size',3,self.resolution),('gan_dims',4,128),('data_workers',0,32),
                                      ('retraining_capacity',1,2048),('retraining_every',1,10000)]:
            value=getattr(self,name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
        for f in fields(self):
            if f.type is bool and type(getattr(self,f.name)) is not bool:
                raise ValueError(f'{f.name} must be a JSON boolean')
        if self.face_type not in ('h','mf','f','wf','head'):
            raise ValueError('Unsupported face_type')
        if self.loss_function not in ('SSIM','MS-SSIM','MS-SSIM+L1'):
            raise ValueError('Unknown ME loss_function')
        if self.lr_dropout not in ('n','y','cpu'):
            raise ValueError('lr_dropout must be n, y, or cpu')
        if self.ct_mode not in ('none','rct','lct','mkl','idt','sot','mix','fs-aug','cc-aug'):
            raise ValueError('Unsupported training color transfer mode')
        if self.true_face_power and not self.archi.startswith('df'):
            raise ValueError('TrueFace latent discrimination requires a df architecture')
        if self.retraining_samples and self.retraining_capacity < self.batch_size:
            raise ValueError('retraining_capacity must hold at least one batch')
        for name,lo,hi in [('lr',1e-8,1e-2),('background_power',0.,1.),('face_style_power',0.,100.),('bg_style_power',0.,100.),('random_hsv_power',0.,.3),
                           ('gan_power',0.,10.),('gan_smoothing',0.,.5),('gan_noise',0.,.5),('true_face_power',0.,1.)]:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{name} must be finite and in [{lo}, {hi}]')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data,dict):
            raise ValueError('ME config must be a JSON object')
        unknown=set(data)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Unsupported ME options (not silently ignored): '+', '.join(sorted(unknown)))
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def effective(self):
        """Pretraining uses its own faceset and disables identity/style adversaries."""
        return replace(self, gan_power=0., true_face_power=0., face_style_power=0., bg_style_power=0.) if self.pretrain else self

    def changed_fields(self, other):
        return {field.name for field in fields(self) if getattr(self, field.name) != getattr(other, field.name)}

    def assert_compatible_network(self, other):
        structural = {'archi','resolution','ae_dims','e_dims','d_dims','d_mask_dims','face_type'}
        changed = self.changed_fields(other) & structural
        if changed:
            raise ValueError('Cannot change checkpoint network fields: '+', '.join(sorted(changed)))
