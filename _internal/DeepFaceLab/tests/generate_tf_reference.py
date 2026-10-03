"""Run with a TensorFlow interpreter; consumes pristine TF DFL + local ME ops.

python tests/generate_tf_reference.py --tf-repo ... --me-root ... --output ...
Only writes output NPZ files; never edits either reference source tree.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument('--tf-repo',required=True)
parser.add_argument('--me-root',required=True)
parser.add_argument('--output',required=True)
args = parser.parse_args()
sys.path.insert(0,str(Path(args.tf_repo).resolve()))
import numpy as np
from core.leras import nn
nn.initialize(nn.DeviceConfig.CPU(),data_format='NHWC')
tf = nn.tf

def load_source(name,path):
    spec = importlib.util.spec_from_file_location(name,str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the actual ME loss primitives, not another translation of their equations.
load_source('me_reference_ops',Path(args.me_root)/'core/leras/ops/__init__old.py')
load_source('me_reference_ms',Path(args.me_root)/'core/leras/layers/MsSsim.py')
rng = np.random.RandomState(2026)
resolution = 64
x_np = rng.uniform(0,1,(1,resolution,resolution,3)).astype(np.float32)
y_np = (.7*x_np+.3*rng.uniform(0,1,x_np.shape)).astype(np.float32)
x = tf.placeholder(tf.float32,x_np.shape)
y = tf.placeholder(tf.float32,y_np.shape)
arch = nn.DeepFakeArchi(resolution,opts='ud')
encoder = arch.Encoder(3,16,name='encoder')
inter_AB = arch.Inter(4*4*128,32,64,name='inter_AB')
inter_B = arch.Inter(4*4*128,32,64,name='inter_B')
decoder = arch.Decoder(128,16,16,name='decoder')
enc = encoder(x)
ab,b = inter_AB(enc),inter_B(enc)
swap,srcm = decoder(tf.concat((ab,ab),-1))
dst,dstm = decoder(tf.concat((b,ab),-1))
objective = tf.reduce_mean(tf.square(swap-y))+tf.reduce_mean(srcm)
grad = tf.gradients(objective,x)[0]
blur = nn.gaussian_blur(x,2.)
dssim = nn.dssim(x,y,max_val=1.,filter_size=6)
style = nn.style_loss(x,y,gaussian_blur_radius=8,loss_weight=1.)
# The actual ME MsSsim layer explicitly accepts NCHW even when CPU is NHWC.
ms = nn.MsSsim(1,3,64)(tf.transpose(x,(0,3,1,2)),tf.transpose(y,(0,3,1,2)),1.)
ms_l1 = nn.MsSsim(1,3,64,use_l1=True)(tf.transpose(x,(0,3,1,2)),tf.transpose(y,(0,3,1,2)),1.)
all_weights = encoder.get_weights()+inter_AB.get_weights()+inter_B.get_weights()+decoder.get_weights()
weight_values = {}
assignments = []
for w in all_weights:
    shape = w.shape.as_list()
    scale = .03 if len(shape)>1 else .01
    value = rng.normal(0,scale,shape).astype(np.float32)
    weight_values[w.name] = value
    assignments.append(tf.assign(w,value))
nn.tf_sess.run(assignments)
values = nn.tf_sess.run(dict(encoder=enc,inter_AB=ab,inter_B=b,swap=swap,
    src_mask=srcm,dst=dst,dst_mask=dstm,input_gradient=grad,
    blur=blur,dssim=dssim,style=style,ms=ms,ms_l1=ms_l1),{x:x_np,y:y_np})
values['input'] = x_np
values['target'] = y_np
output = Path(args.output)
output.mkdir(parents=True,exist_ok=True)
np.savez(str(output/'tf_weights.npz'),**weight_values)
np.savez(str(output/'tf_outputs.npz'),**values)
print('Generated actual TensorFlow reference: {} tensors, TF {}'.format(len(all_weights),tf.__version__))
