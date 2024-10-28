# Neural Adaptation Primer

This repository provides a tutorial on how to model neural
adaptation. It complements the primer on "[Neural
adaptation](https://doi.org/10.1016/j.cub.2020.11.054)" by [Jan
Benda](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/biologie/institute/neurobiologie/lehrbereiche/neuroethologie/)
published in Current Biology, 2021. Feel free to use and distribute
the scripts and figures for teaching. The code, figures, and
descriptions are provided under the [GNU General Public License
v3.0](LICENSE).

For each of the topics listed below, there is a folder `TOPIC/`
containing the following files:
- `README.md`: a tutorial on the topic and how to code the models.
- `TOPIC.py`: standalone python script containing the functions
  explained in the README.md file that can be run as a demo.
- `TOPICplots.py`: python script using the functions in `TOPIC.py` to
  generate the figures needed for the `README.md` file.
- `TOPIC-*.png`: figure files for `README.md` generated by
  `TOPICplots.py`.

For running the demo script, change into the directory of the topic
and run the script. For example, to run the demo for the adapting
leaky integrate-and-fire model in 'lifac/' do:
```
cd lifac
python3 lifac.py
```
Or open the 'lifac/lifac.py' script in your favorite IDE and run it
from there.


## Citation

Cite this repository via the associated manuscript:

> Benda, Jan (2021): Neural adaptation. *Current Biology* 31(3), R110-R116.
> [doi:10.1016/j.cub.2020.11.054](https://doi.org/10.1016/j.cub.2020.11.054)


## Requirements

The `python` scripts run in python version 3, using the following packages:

- numpy
- scipy >= 1.2.0
- matplotlib >= 2.2.0


## Tutorial

### Leaky integrate-and-fire with adaptation current

The leaky integrate-and-fire model is a simple model of a spiking
neuron. Augmented with a generic adaptation current it reproduces many
features of intrinsically adapting neurons. [Read more in
`lifac/`.](lifac/README.md)


### Spike-frequency adaptation models

Spike-frequency adaptation is a phenomenon observed on the level of,
well, spike frequencies. Modeling adaptation on the level of spike
freuqencies is thus a natural choice. [Read more in
`sfa/`.](sfa/README.md)


### Adaptation high-pass filter

Spike-frequency adaptation basically adds a high-pass filter to the
neuron's input-output function. This filter operation interacts with
the non-linear *f-I* curves of the neuron. [Read
more in `filter/`.](filter/README.md)


### Adaptation to stimulus mean and variance

Subtractive adaptation is perfectly suited to make the neuron's
response invariant with respect to the mean of the
stimulus. Invariance to the stimulus variance, however, requires
thresholding to extract the amplitude modulation and divisive
adaptation. [Read more in `meanvariance/`.](meanvariance/README.md)


### Stimulus-specific adaptation

Adaptation in parallel pathways leads to stimulus-specific
adaptation. [Read more in `ssa/`.](ssa/README.md)


### Resolving ambiguities

Absolute stimulus intensity is ambiguously encoded by an adapting
neuron. Nonetheless, matched intrinsic adaptation allows down-stream
neurons to robustly encode absolute stimulus intensity. [Read more in
`ambiguities/`.](ambiguities/README.md)


### Generating sparse codes

Efficient codes are both temporally and spatially sparse. Intrinsic
adaptation together with lateral inhibition generate such sparse
codes. [Read more in `sparse/`.](sparse/README.md)


## Contributing

You are welcome to improve the code and the explanations. Or even add
another chapter.

Fork the repository, work on your suggestions and make a pull request.

For minor issues, e.g. a reference you want me to add, or fixing
little quirks, please open an issue.