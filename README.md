# CLiC: Concept Learning in Context
<!--<a href="https://arxiv.org/abs/2303.10735v3"><img src="https://img.shields.io/badge/arXiv-2303.10753-b31b1b.svg" height=20.5></a>-->
<a href="https://mehdi0xc.github.io/clic"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a>

> **Mehdi Safaee, Aryan Mikaeili, Or Patashnik, Daniel Cohen-Or, Ali Mahdavi-Amiri**
>   
>This paper addresses the challenge of learning a local visual pattern of an object from one image, and generating images depicting objects with that pattern. Learning a localized concept
      and placing it on an object in a target image is a nontrivial task, as the objects may have different orientations and shapes. Our approach builds upon recent advancements in visual concept learning.
      It involves acquiring a visual concept (e.g., an ornament) from a source image and subsequently applying it to an object (e.g., a chair) in a target image. Our key idea is to perform in-context concept learning,
      acquiring the local visual concept within the broader context of the objects they belong to. To localize the concept learning, we employ soft masks that contain both the concept within the mask and the surrounding image area.
      We demonstrate our approach through object generation within an image, showcasing plausible embedding of in-context learned concepts. We also introduce methods for directing acquired concepts to specific locations within target images,
      employing cross-attention mechanisms, and establishing correspondences between source and target objects. The effectiveness of our method is demonstrated through quantitative and qualitative experiments, along with comparisons against baseline techniques.

<p align="center">
<img src="https://github.com/Mehdi0xC/clic/blob/page/resources/teaser.png" width="800px"/>
</p>

# Description

Official implementation of the paper:  CLiC: Concept Learning in Context

# How to run the code
- `learning.py` should be used to learn a concept from a source image and saving the parameters.
- `transfer.py` should be used to transfer a learned concept to a target image. 
- Both source and target images should have masks specifying the desired regions. 
- High-level configs are being obtained from `configs/default.yaml` but can be overridden by providing arguments.
- View or run `reproduce.sh` for more information on how to execute the code. 

# Citation
To cite the paper please use the following bibtex:

``` 
@article{safaee2023clic,
    title={CLiC: Concept Learning in Context},
    author={Mehdi Safaee and Aryan Mikaeili and Or Patashnik and Daniel Cohen-Or and Ali Mahdavi-Amiri},
    journal={ArXiv},
    year={2023}
}
```