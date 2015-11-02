"""
Decoder
"""
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy

from search import gen_sample
from collections import OrderedDict


def load_model(path_to_model, path_to_dictionary):
    """
    Load a trained model for decoding
    """
    # Load the worddict
    with open(path_to_dictionary, 'rb') as f:
        worddict = pkl.load(f)

    # Create inverted dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Load model options
    with open('%s.pkl'%path_to_model, 'rb') as f:
        options = pkl.load(f)
    if 'doutput' not in options.keys():
        options['doutput'] = True

    # Load parameters
    params = init_params(options)
    params = load_params(path_to_model, params)
    tparams = init_tparams(params)

    # Sampler.
    trng = RandomStreams(1234)
    f_init, f_next = build_sampler(tparams, options, trng)

    # Pack everything up
    dec = dict()
    dec['options'] = options
    dec['trng'] = trng
    dec['worddict'] = worddict
    dec['word_idict'] = word_idict
    dec['tparams'] = tparams
    dec['f_init'] = f_init
    dec['f_next'] = f_next
    return dec

def run_sampler(dec, c, beam_width=1, stochastic=False, use_unk=False):
    """
    Generate text conditioned on c
    """
    sample, score = gen_sample(dec['tparams'], dec['f_init'], dec['f_next'],
                               c.reshape(1, dec['options']['dimctx']), dec['options'],
                               trng=dec['trng'], k=beam_width, maxlen=1000, stochastic=stochastic,
                               use_unk=use_unk)
    text = []
    if stochastic:
        sample = [sample]
    for c in sample:
        text.append(' '.join([dec['word_idict'][w] for w in c[:-1]]))

    #Sort beams by their NLL, return the best result
    lengths = numpy.array([len(s.split()) for s in text])
    if lengths[0] == 0:  # in case the model only predicts <eos>
        lengths = lengths[1:]
        score = score[1:]
        text = text[1:]
    sidx = numpy.argmin(score)
    text = text[sidx]
    score = score[sidx]

    return text

def _p(pp, name):
    """
    make prefix-appended name
    """
    return '%s_%s'%(pp, name)

def init_tparams(params):
    """
    initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def load_params(path, params):
    """
    load parameters
    """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive'%kk)
            continue
        params[kk] = pp[kk]
    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer')}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # init state
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=options['dimctx'], nout=options['dim'])

    # Decoder
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder',
                                              nin=options['dim_word'], dim=options['dim'])

    # Output layer
    if options['doutput']:
        params = get_layer('ff')[0](options, params, prefix='ff_hid', nin=options['dim'], nout=options['dim_word'])
        params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])
    else:
        params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim'], nout=options['n_words'])

    return params

def build_sampler(tparams, options, trng):
    """
    Forward sampling
    """
    ctx = tensor.matrix('ctx', dtype='float32')
    ctx0 = ctx

    init_state = get_layer('ff')[1](tparams, ctx, options, prefix='ff_state', activ='tanh')
    f_init = theano.function([ctx], init_state, name='f_init', profile=False)

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][y])

    # decoder
    proj = get_layer(options['decoder'])[1](tparams, emb, init_state, options,
                                            prefix='decoder',
                                            mask=None,
                                            one_step=True)
    next_state = proj[0]

    # output
    if options['doutput']:
        hid = get_layer('ff')[1](tparams, next_state, options, prefix='ff_hid', activ='tanh')
        logit = get_layer('ff')[1](tparams, hid, options, prefix='ff_logit', activ='linear')
    else:
        logit = get_layer('ff')[1](tparams, next_state, options, prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    inps = [y, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=False)

    return f_init, f_next

def linear(x):
    """
    Linear activation function
    """
    return x

def tanh(x):
    """
    Tanh activation function
    """
    return tensor.tanh(x)

def ortho_weight(ndim):
    """
    Orthogonal weight init, for recurrent layers
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.1, ortho=True):
    """
    Uniform initalization from [-scale, scale]
    If matrix is square and ortho=True, use ortho instead
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')

# Feedforward layer
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    Feedforward pass
    """
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU)
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def gru_layer(tparams, state_below, init_state, options, prefix='gru', mask=None, one_step=False, **kwargs):
    """
    Feedforward pass through GRU
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    if one_step:
        rval = _step(*(seqs+[init_state, tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Ux')]]))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info = [init_state],
                                    non_sequences = [tparams[_p(prefix, 'U')],
                                                     tparams[_p(prefix, 'Ux')]],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=False,
                                    strict=True)
    rval = [rval]
    return rval

