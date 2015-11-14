# neural-storyteller

neural-storyteller is a recurrent neural network that generates little stories about images. This repository contains code for generating stories with your own images, as well as instructions for training new models.

<img src="https://github.com/ryankiros/neural-storyteller/blob/master/images/ex1.jpg" height="220px" align="left">
*We were barely able to catch the breeze at the beach , and it felt as if someone stepped out of my mind . She was in love with him for the first time in months , so she had no intention of escaping . The sun had risen from the ocean , making her feel more alive than normal . She 's beautiful , but the truth is that I do n't know what to do . The sun was just starting to fade away , leaving people scattered around the Atlantic Ocean . I d seen the men in his life , who guided me at the beach once more .*

[Samim](http://samim.io/) has made an awesome blog post with lots of results [here](https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed).

Some more results from an older model trained on Adventure books can be found [here](http://www.cs.toronto.edu/~rkiros/adv_L.html).

The whole approach contains 4 components:
* [skip-thought vectors](https://github.com/ryankiros/skip-thoughts)
* [image-sentence embeddings](https://github.com/ryankiros/visual-semantic-embedding)
* [conditional neural language models](https://github.com/ryankiros/skip-thoughts/tree/master/decoding)
* style shifting (described in this project)

The 'style-shifting' operation is what allows our model to transfer standard image captions to the style of stories from novels. The only source of supervision in our models is from [Microsoft COCO](http://mscoco.org/) captions. That is, we did not collect any new training data to directly predict stories given images.

Style shifting was inspired by [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) but the technical details are completely different.

## How does it work?

We first train a recurrent neural network (RNN) decoder on romance novels. Each passage from a novel is mapped to a skip-thought vector. The RNN then conditions on the skip-thought vector and aims to generate the passage that it has encoded. We use romance novels collected from the BookCorpus [dataset](http://www.cs.toronto.edu/~mbweb/).

Parallel to this, we train a visual-semantic embedding between COCO images and captions. In this model, captions and images are mapped into a common vector space. After training, we can embed new images and retrieve captions.

Given these models, we need a way to bridge the gap between retrieved image captions and passages in novels. That is, if we had a function F that maps a collection of image caption vectors **x** to a book passage vector F(**x**), then we could feed F(**x**) to the decoder to get our story. There is no such parallel data, so we need to construct F another way.

It turns out that skip-thought vectors have some intriguing properties that allow us to construct F in a really simple way. Suppose we have 3 vectors: an image caption **x**, a "caption style" vector **c** and a "book style" vector **b**. Then we define F as

F(**x**) = **x** - **c** + **b**

which intuitively means: keep the "thought" of the caption, but replace the image caption style with that of a story. Then, we simply feed F(**x**) to the decoder.

How do we construct **c** and **b**? Here, **c** is the mean of the skip-thought vectors for Microsoft COCO training captions. We set **b** to be the mean of the skip-thought vectors for romance novel passages that are of length > 100.

#### What kind of biases work?

Skip-thought vectors are sensitive to:

- length (if you bias by really long passages, it will decode really long stories)
- punctuation
- vocabulary
- syntactic style (loosely speaking)

For the last point, if you bias using text all written the same way the stories you get will also be written the same way.

#### What can the decoder be trained on?

We use romance novels, but that is because we have over 14 million passages to train on. Anything should work, provided you have a lot of text! If you want to train your own decoder, you can use the code available [here](https://github.com/ryankiros/skip-thoughts/tree/master/decoding) Any models trained there can be substituted here.

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [Lasagne](https://github.com/Lasagne/Lasagne)
* A version of Theano that Lasagne supports

For running on CPU, you will need to install [Caffe](http://caffe.berkeleyvision.org) and its python interface.


## Getting started

You will first need to download some pre-trained models and style vectors. Most of the materials are available in a single compressed file, which you can obtain by running

    wget http://www.cs.toronto.edu/~rkiros/neural_storyteller.zip

Included is a pre-trained decoder on romance novels, the decoder dictionary, caption and romance style vectors, MS COCO training captions and a pre-trained image-sentence embedding model.

Next, you need to obtain the pre-trained skip-thoughts encoder. Go [here](https://github.com/ryankiros/skip-thoughts) and follow the instructions on the main page to obtain the pre-trained model.

Finally, we need the VGG-19 ConvNet parameters. You can obtain them by running

    wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

Note that this model is for non-commercial use only. Once you have all the materials, open `config.py` and specify the locations of all of the models and style vectors that you downloaded.

For running on CPU, you will need to download the VGG-19 prototxt and model by:

    wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
    wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt

 You also need to modify pycaffe and model path in `config.py`, and modify the flag in line 8 as:

    FLAG_CPU_MODE = True

## Generating a story

The images directory contains some sample images that you can try the model on. In order to generate a story, open Ipython and run the following:

    import generate
    z = generate.load_all()
    generate.story(z, './images/ex1.jpg')

If everything works, it will first print out the nearest COCO captions to the image (predicted by the visual-semantic embedding model). Then it will print out a story.

#### Generation options

There are 2 knobs that can be tuned for generation: the number of retrieved captions to condition on as well as the beam search width. The defaults are

    generate.story(z, './images/ex1.jpg', k=100, bw=50)

where k is the number of captions to condition on and bw is the beam width. These are reasonable defaults but playing around with these can give you very different outputs! The higher the beam width, the longer it takes to generate a story.

If you bias by song lyrics, you can turn on the lyric flag which will print the output in multiple lines by comma delimiting. `neural_storyteller.zip` contains an additional bias vector called `swift_style.npy` which is the mean of skip-thought vectors across Taylor Swift lyrics. If you point `path_to_posbias` to this vector in `config.py`, you can generate captions in the style of Taylor Swift lyrics. For example:

    generate.story(z, './images/ex1.jpg', lyric=True)

should output

    You re the only person on the beach right now
    you know
    I do n't think I will ever fall in love with you
    and when the sea breeze hits me
    I thought
    Hey

## Reference

This project does not have any associated paper with it. If you found this code useful, please consider citing:

Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. **"Skip-Thought Vectors."** *arXiv preprint arXiv:1506.06726 (2015).*

    @article{kiros2015skip,
      title={Skip-Thought Vectors},
      author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
      journal={arXiv preprint arXiv:1506.06726},
      year={2015}
    }

If you also use the BookCorpus data for training new models, please also consider citing:

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler.
**"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books."** *arXiv preprint arXiv:1506.06724 (2015).*

    @article{zhu2015aligning,
        title={Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books},
        author={Zhu, Yukun and Kiros, Ryan and Zemel, Richard and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
        journal={arXiv preprint arXiv:1506.06724},
        year={2015}
    }
