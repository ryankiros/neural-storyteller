"""
Configuration for the generate module
"""

#-----------------------------------------------------------------------------#
# Flags for running on CPU
#-----------------------------------------------------------------------------#
FLAG_CPU_MODE = True

#-----------------------------------------------------------------------------#
# Paths to models and biases
#-----------------------------------------------------------------------------#
paths = dict()

# Skip-thoughts
paths['skmodels'] = '/u/rkiros/public_html/models/'
paths['sktables'] = '/u/rkiros/public_html/models/'

# Decoder
paths['decmodel'] = '/ais/gobi3/u/rkiros/storyteller/romance.npz'
paths['dictionary'] = '/ais/gobi3/u/rkiros/storyteller/romance_dictionary.pkl'

# Image-sentence embedding
paths['vsemodel'] = '/ais/gobi3/u/rkiros/storyteller/coco_embedding.npz'

# VGG-19 convnet
paths['vgg'] = '/ais/gobi3/u/rkiros/vgg/vgg19.pkl'
paths['pycaffe'] = '/u/yukun/Projects/caffe-run/python'
paths['vgg_proto_caffe'] = '/ais/guppy9/movie2text/neural-storyteller/models/VGG_ILSVRC_19_layers_deploy.prototxt'
paths['vgg_model_caffe'] = '/ais/guppy9/movie2text/neural-storyteller/models/VGG_ILSVRC_19_layers.caffemodel'


# COCO training captions
paths['captions'] = '/ais/gobi3/u/rkiros/storyteller/coco_train_caps.txt'

# Biases
paths['negbias'] = '/ais/gobi3/u/rkiros/storyteller/caption_style.npy'
paths['posbias'] = '/ais/gobi3/u/rkiros/storyteller/romance_style.npy'
