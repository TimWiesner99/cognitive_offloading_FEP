# Simulating Cognitive Offloading under the Free Energy Principle
This repository belongs to the bachelor's thesis "Simulating Cognitive Offloading under the Free Energy Principle" (2022) written at the AI department of Radboud University (NL). For questions or to request the paper, please contact me under tim.wiesner@student.ru.nl.

### Abstract
Cognitive Offloading is the very common psychological process of using our body or physical environment to reduce the mental load of a cognitive task. Turning one's head to be able to read some text more easily, called external normalization, is one example of this. Computational models can help us formalize, simulate and understand cognitive behaviours like these. Recent works in active inference have used the free energy principle to build models that closely connect perception to action. In this work, two computational models, formulated under the free energy principle, are proposed to model cognitive offloading, or more specifically, external normalization. While they are not more than a proof of concept, results show that the free energy principle is indeed a useful framework for modelling cognitive processes.

### Notes
The files in `parallel_decoder` belong to the first model described in the paper as "Continuous Action model without Transition Model". The second model "Discrete Action model with Transition Model" is contained in the directory `discretized_model`. Neural networks are already trained and can be read from their respective files in `/networks`. Place the original MNIST files in a `./data` folder to have them be read by the custom data loader.
