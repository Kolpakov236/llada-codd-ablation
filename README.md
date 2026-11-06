# LLaDA + CoDD: a GSM8K ablation study

We tested which parts of [CoDD](https://arxiv.org/abs/2603.00045) affect math accuracy when decoding with LLaDA-8B-Instruct. The ablations separate the choice of token positions, the choice of token values, and dependencies between the hidden states used by the probabilistic circuit.

| Variant | Notebook | Accuracy, 200 tasks | Accuracy, 300 tasks |
| --- | --- | ---: | ---: |
| LLaDA baseline | [f1](f1_base_full512.ipynb) | 72.0% | 67.7% |
| CoDD position order only | [f2](f2_order_full512.ipynb) | 65.0% | 62.7% |
| CoDD token choices only | [f3](f3_words_full512.ipynb) | 66.0% | 63.3% |
| CoDD with independent hidden states | [f4](f4_indep_full512.ipynb) | 61.5% | 61.5% |
| Full CoDD | [f5](f5_codd_full512.ipynb) | 63.5% | 65.3% |

Each run used the same tasks across its five variants, a 512-token generation budget, margin remasking, and 128 steps. The 200-task run used a random subset; the repeat used the first 300 test tasks. Full CoDD did not improve accuracy in either run. These results describe our setup and subsets. They do not establish that CoDD is worse in general.

## Running the notebooks

1. Use a Linux x86-64 CUDA machine with Python 3.10. The notebooks were prepared for Yandex DataSphere and an NVIDIA L4.
2. Clone the [upstream CoDD implementation](https://github.com/liuanji/CoDD) into `CoDD/`. The local reference checkout used commit `d9760f41630c3d16c5106da31734e203e99c74c7`.
3. Prepare the local LLaDA weights, GSM8K data, and Python environment. See the header and `--help` in [prepare_offline.py](prepare_offline.py) for the offline bundle workflow. For variants f2 through f5, use [download_hmm.py](download_hmm.py) for the probabilistic circuit checkpoint and verify it against [README_HMM.md](README_HMM.md).
4. Open one of the f1 through f5 notebooks and check the paths in its first code cell. Run the environment checks and `smoke` before `full`. The checked-in notebooks implement the 200-task configuration. Repeating the 300-task run requires changing the task selector and matching configuration checks across all five variants.

The repository contains notebooks and scripts. Model weights, datasets, raw run logs, and the offline environment are separate. The `exp*.ipynb` notebooks record earlier experimental setups.

## Authors and materials

Makar Kolpakov, Vadim Medvedko, Dmitry Sidimekov, and Daniil Ignatiev.

- [Project slides and results (Russian)](https://disk.yandex.ru/i/F-eEzP8RkChOmA)
- [Related project: Mamba in Tiny Recursive Model on ARC-AGI-2 (Russian)](https://disk.yandex.ru/i/2SSg3KvFAeyMIQ)
