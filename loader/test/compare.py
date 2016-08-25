# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Verify that different ways of loading datasets lead to the same result.

This test utility accepts the same command line parameters as neon. It
downloads the CIFAR-10 dataset and saves it as individual PNG files. It then
proceeds to fit and evaluate a model using two different ways of loading the
data. Macrobatches are written to disk as needed.

run as follows:
python compare.py -e1 -r0 -w <place where data lives>

"""
import numpy as np
from neon import NervanaObject
from neon.data import ArrayIterator
from neon.initializers import Uniform
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, Rectlin, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.util.persist import get_data_cache_dir
from neon.data.dataloader_transformers import OneHot, TypeCast, ImageMeanSubtract
from ingesters import ingest_cifar10
from aeon import DataLoader
from PIL import Image


bgr_means = [127, 119, 104]


def make_aeon_config(manifest_filename, cache_directory, minibatch_size, do_randomize=False):
    image_decode_cfg = dict(
        height=32, width=32,
        scale=[1.0, 1.0],
        flip_enable=do_randomize,
        center=(not do_randomize))

    return dict(
        manifest_filename=manifest_filename,
        minibatch_size=minibatch_size,
        macrobatch_size=5000,
        cache_directory=cache_directory,
        subset_fraction=1.0,
        shuffle_manifest=do_randomize,
        shuffle_every_epoch=do_randomize,
        type='image,label',
        label={'binary': False},
        image=image_decode_cfg)


def transformers(dl):
    dl = OneHot(dl, nclasses=10, index=1)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = ImageMeanSubtract(dl, index=0, pixel_mean=bgr_means)
    return dl


def load_dataset(basepath, datadir, manifest):
    with open(manifest) as fd:
        lines = fd.readlines()
    assert len(lines) > 0, 'could not read %s' % manifest

    data = None
    for idx, line in enumerate(lines):
        imgfile, labelfile = line.split(',')
        labelfile = labelfile[:-1]
        # Convert from RGB to BGR to be consistent with the data loader
        im = np.asarray(Image.open(imgfile))[:, :, ::-1]
        # Convert from HWC to CHW
        im = np.transpose(im, axes=[2, 0, 1]).ravel()
        if data is None:
            data = np.empty((len(lines), im.shape[0]), dtype='float32')
            labels = np.empty((len(lines), 1), dtype='int32')
        data[idx] = im
        with open(labelfile) as fd:
            labels[idx] = int(fd.read())
    data_view = data.reshape((data.shape[0], 3, -1))
    # Subtract mean values of B, G, R
    data_view -= np.array(bgr_means).reshape((1, 3, 1))
    return (data, labels)


def load_cifar10_imgs(path):
    (X_train, y_train) = load_dataset(path, 'train', train_manifest)
    (X_test, y_test) = load_dataset(path, 'val', val_manifest)
    return (X_train, y_train), (X_test, y_test), 10


def run(train, test):
    init_uni = Uniform(low=-0.1, high=0.1)
    opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                      momentum_coef=0.9,
                                      stochastic_round=args.rounding)
    layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=True),
              Pooling((2, 2)),
              Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=True),
              Pooling((2, 2)),
              Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=True),
              Affine(nout=10, init=init_uni, activation=Softmax())]
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    mlp = Model(layers=layers)
    callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)
    mlp.fit(train, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
    err = mlp.eval(test, metric=Misclassification())*100
    print('Misclassification error = %.2f%%' % err)
    return err


def test_iterator():
    print('Testing data iterator')
    (X_train, y_train), (X_test, y_test), nclass = load_cifar10_imgs(path=image_dir)
    train = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))
    test = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))
    return run(train, test)


def test_loader():
    print('Testing data loader')
    train_config = make_aeon_config(train_manifest, cache_dir, args.batch_size)
    val_config = make_aeon_config(val_manifest, cache_dir, args.batch_size)

    train = transformers(DataLoader(train_config, NervanaObject.be))
    test = transformers(DataLoader(val_config, NervanaObject.be))

    err = run(train, test)
    return err


parser = NeonArgparser(__doc__)
args = parser.parse_args()

image_dir = get_data_cache_dir(args.data_dir, subdir='extracted')
cache_dir = get_data_cache_dir(args.data_dir, subdir='cache')

# Perform ingest if it hasn't already been done and return manifest files
train_manifest, val_manifest = ingest_cifar10(out_dir=image_dir, padded_size=32)

assert test_iterator() == test_loader(), 'The results do not match'
print 'OK'
