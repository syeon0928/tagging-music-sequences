# Advanced Machine Learning Final Project

## Tagging music sequences.

![](mp_musictag.png)

Music plays an important role in our lives, while the landscape of contemporary music is vast. In order to understand music taste and build recommender systems for music, we need to learn to tag music first. In this project, we want to build a classifier that can tag music pieces with a genre or category after listening to an arbitrary long example. For this, we want to consider the following datsets:
- [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- [The MagnaTagATune Dataset (MTAT)](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset), and
- for advanced studies: [the Free Music Archive (FMA)](http://freemusicarchive.org/).

### Main goals:

- Research literature about sound and music pre-processing, transformation, and representation. What type of pre-processing is best for music pieces, i.e. what is the state-of-the-art of spectrograms vs. raw waveform?
- Train an encoding model (deep recurrent and/or CNN network) with appropriate representation to classify sequences of music pieces. Your options are vast as you can consider all the tools that we covered in class: GRUs? CNNs? Variational Encoders? Combinations thereof? Make use of recent examples from literature! Can you identify an architecture (and meta-parameter settings) that can be trained to tag/classify considerably well?
- Study the performance for edge cases, such as particularly short input sequences or music pieces for rare genres/categories. Can you identify characteristics of such edge cases that make performance particularly high or low?
- Identify differences in quantitative performance and qualitative characteristics (look into how your model decides in edge cases) between different pre-processing options.

### Optional:

- Build your music tagger by training only on one of the datasets and comparing generalisation on the other. Given that you took good care of appropriate representation and pre-processing for both, can you explain the performance differences?
- Look into pre-trained options (e.g. from paperswithcode.com) and finetune your extended models. How is performance (quantitative and qualitative) different?
