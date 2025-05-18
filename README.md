# Automata Learning from Preference and Equivalence Queries

The project website can be found at [https://eric-hsiung.github.io/remap](https://eric-hsiung.github.io/remap).

This is the experimental code repository for the paper **[Automata Learning from Preference and Equivalence Queries](https://arxiv.org/pdf/2308.09301)** (CAV 2025).

It contains an implementation of the REMAP algorithm, unit tests, and code for running experiments.
It additionally contains a fork where certain bugs in the [reward_machines](reward_machines) (Icarte et al. 2018) repository have been corrected.
The corrections are specification of a deterministic reward machine, specifically for CraftWorld. For specific details about the corrections, please
refer to the Appendix of the [paper](https://arxiv.org/pdf/2308.09301).

The main README for REMAP is found at [remap/README](remap/README).

To cite this work:
```
@misc{hsiung2023remap,
        title={Automata Learning from Preference and Equivalence Queries}, 
        author={Eric Hsiung and Joydeep Biswas and Swarat Chaudhuri},
        year={2023},
        eprint={2308.09301},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
      }
```
