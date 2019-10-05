import argparse
import os
import sys
import warnings

#Import logger.
#if __name__ == "__main__":
#    from comet_ml import Experiment, predictor


import keras
import keras.preprocessing.image
import tensorflow as tf
import glob

import sys
wdir = os.getcwd()
#print(dir)
sys.path.insert(0, os.path.join(os.path.dirname(wdir), '..', '..'))
import keras_retinanet.bin
__package__ = "keras_retinanet.bin"


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
from ..utils.image import random_visual_effect_generator

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('./keras_retinanet/backbones', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, \
                  freeze_backbone=False, lr=1e-5, config=None, nms_threshold=None, input_channels=3):
    """ Creates three models (model, training_model, prediction_model).
    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.
    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    modifier = freeze_model if freeze_backbone else None
    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()
    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, \
            modifier=modifier, input_channels=input_channels), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        training_model = model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, \
        modifier=modifier, input_channels=input_channels), weights=weights, skip_mismatch=True)
        training_model = model
    # make prediction model
    print("Making prediction model with nms = %.2f" % nms_threshold )
    prediction_model = retinanet_bbox(model=model, nms_threshold=nms_threshold, anchor_params=anchor_params)
    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, train_generator, validation_generator, args, experiment = None):
    """ Creates the callbacks to use during training.
    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    tensorboard_callback = None
    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)
    if args.evaluation and validation_generator:
        evaluation = Evaluate(validation_generator,
                              #experiment=experiment,
                              tensorboard=tensorboard_callback,
                              weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)
    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{{epoch:02d}}.h5'.format(backbone=args.backbone)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))
     #create the NEON generator
    #NEON_generator = create_NEON_generator(args.batch_size, DeepForest_config)
    #neon_evaluation = NEONmAP(NEON_generator,
    #                          experiment=experiment,
    #                          save_path=args.save_path,
    #                          score_threshold=args.score_threshold,
    #                          DeepForest_config=DeepForest_config)
    #neon_evaluation = RedirectModel(neon_evaluation, prediction_model)
    #callbacks.append(neon_evaluation)
    #comet_loss predictor
    #predictor_callback = predictor.Predictor(experiment, loss_name="loss", patience = 10, best_callback= None, threshold=0.1)
    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }
    #TODO: here you want to include Hierarchical Dimensionality Reduction (https://github.com/GatorSense/hsi_toolkit_py/blob/master/dim_reduction/hdr.py)
    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None
    train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            shuffle_groups=True,
            **common_args
        )
    else:
        validation_generator = None
    return train_generator, validation_generator



def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """
    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))
    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))
    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")
    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))
    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    def csv_list(string):
        return string.split(',')
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    #subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.', default = './snapshots/latest_snapshot.h5', type=str)
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.', default = './keras_retinanet/backbones/universal_deepLidar.h5', type=str)
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    #not sure if I want them here
    parser.add_argument('--annotations', help='Path to CSV file containing annotations for training.', default='./dataset/train.csv', type=str)
    parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.', default='./dataset/classes.csv',type=str)
    parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).', default='./dataset/test.csv', type=str)
    #other args
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=40, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=40)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000) #batch_size
    parser.add_argument('--batchsize',        help='Batch size.', type=int, default=40) #batch_size
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots/')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=200)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=200)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')
    parser.add_argument('--nms_threshold',   help='Parameter regulating non-max suppression for overlapping boxes', type=float, default=0.1)
    parser.add_argument('--input_channels',   help='How many channels in the image?', type=int, default=3)
    #Comet ml image viewer
    parser.add_argument('--save-path',       help='Path for saving eval images with detections (doesn\'t work for COCO).', default="./eval/", type=str)
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.3).', default=0.05, type=float)
    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)
    return check_args(parser.parse_args(args))



def main(args=None, experiment=None):
    # parse arguments
    print("parsing arguments")
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # create object that stores backbone information
    backbone = models.backbone(args.backbone)
    # make sure keras is the minimum required version
    check_keras_version()
    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())
    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)
    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
    #Log number of trees trained on
    if experiment:
        experiment.log_parameter("Number of Training Trees", train_generator.total_trees)
    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
            #nms_threshold=DeepForest_config["nms_threshold"]
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params) #nms_threshold=DeepForest_config["nms_threshold"]
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()
        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            nms_threshold=args.nms_threshold,
            input_channels=args.input_channels,
            lr=args.lr,
            config=args.config
        )
    # print model summary
    print(model.summary())
    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        #train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            #validation_generator.compute_shapes = train_generator.compute_shapes
            validation_generator.compute_anchor_targets = compute_anchor_targets
    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        train_generator,
        validation_generator,
        args
    )
    if not args.compute_val_loss:
        validation_generator = None
    # start training
    history =  training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size()/args.batch_size,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator)
    #return path snapshot of final epoch
    saved_models = glob.glob(os.path.join(args.snapshot_path,"*.h5"))
    saved_models.sort()
    #Return model if found
    if len(saved_models) > 0:
        return saved_models[-1]


if __name__ == '__main__':
    output_model = main(args=None)
