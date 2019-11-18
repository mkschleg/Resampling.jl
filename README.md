
A repository containing the julia code used for creating the majority of results in "Importance Resampling for Off-policy Prediction" published in NeurIPS 2019.


## Core Idea

Our contribution consists of a new approach to using importance weights for off-policy prediction for learning general value functions. Specifically, we introduce Importance Resampling, the first application of samplie importance-resampling to reinforcement learning. Importance Resampling samples mini-batches from an experience replay buffer according to a probability mass function defined with probabilities proportional to a transitions importance weights of policies (\pi/\mu). We provide theory on the consistency of Importance Sampling under typical conditions in RL, and we show improved sample efficiency as compared to reweighting.


## Other Resources:

These resources were used at the NeurIPS conference, occuring in December of 2019.

- [Poster](https://github.com/mkschleg/Resampling.jl/raw/gh-pages/resources/poster.pdf)
- [Slides](https://github.com/mkschleg/Resampling.jl/blob/gh-pages/resources/IR_neurips2019.pdf)

## Authors

- [Matthew Schlegel](mkschleg.github.io)
- Wes Chung
- Jian Qian
- Daniel Graves
- [Martha White](http://webdocs.cs.ualberta.ca/~whitem/index.html)

## Acknowledgments

We would like to thank Huawei for their support, and especially for allowing a portion of this work to be completed during Matthew's internship in the summer of 2018. We also would like to acknowledge University of Alberta, Alberta Machine Intelligence Institute, and NSERC for their continued funding and support.
