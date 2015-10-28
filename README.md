# neural-storyteller

neural-storyteller is a recurrent neural network that generates little stories about images. This repository contains code for generating stories with your own images, as well as instructions for training new models.

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

## Getting started

You will first need to download some pre-trained models and style vectors. Most of the materials are available in a single compressed file, which you can obtain by running

    wget http://www.cs.toronto.edu/~rkiros/neural_storyteller.zip
    
Included is a pre-trained decoder on romance novels, the decoder dictionary, caption and romance style vectors, MS COCO training captions and a pre-trained image-sentence embedding model.

Next, you need to obtain the pre-trained skip-thoughts encoder. Go [here](https://github.com/ryankiros/skip-thoughts) and follow the instructions on the main page to obtain the pre-trained model.

Finally, we need the VGG-19 ConvNet parameters. You can obtain them by running

    wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

Note that this model is for non-commercial use only. Once you have all the materials, open `generate.py` and specify the locations of all of the models and style vectors that you downloaded.

## Generating a story

The images directory contains some sample images that you can try the model on. In order to generate a story, open Ipython and run the following:

    import generate
    z = generate.load_all()
    generate.story(z, './images/ex1.jpg')

If everything works, it will first print out the nearest COCO captions to the image (predicted by the visual-semantic embedding model). Then it will print out a story.

#### Generation options

There are 2 knobs that can be tuned for generation: the number of retrieved captions to condition on as well as the beam search width. The defaults are

    generate.story(z, './images/ex1.jpg', k=1000, bw=50)

where k is the number of captions to condition on and bw is the beam width. These are reasonable defaults but playing around with these can give you very different outputs! The higher the beam width, the longer it takes to generate a story.

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
