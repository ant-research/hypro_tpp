
# HYPRO

Pytorch implementation for [HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences](https://arxiv.org/abs/2210.01753), NeurIPS 2022.


## How to Run

### Environment Requirements

First, please make sure you have an environment compatible with the following requirement 

```bash
torch == 1.11.0
numpy
pandas
```

Lower version of pytorch should also be working but we have not tested it. 

### Data Preparation

You can obtain all the three benchmarks from [Google Drive](https://drive.google.com/drive/folders/13e5jCkprJGB6jiVtIrU-XaCzSws5PPfB). All the datasets are well pre-processed and can be used easily.

```
mkdir -p data/{data_name}
```
**Please unzipped the files and put them in the `./data/{data_name}` directory**


### Training and Eval Example

Assume we are running over the Taxi data and setup the config files.


Step 1: we need to train the chosen TPP, attNHP, as the generator with the config `configs/taxi/attnhp_train.yaml`:

```
python main_train_generator.py
```
NOTE: in `configs/taxi/attnhp_train.yaml`, one needs to setup data and model specs, where we have put default params there.



Step 2: we use it to generate samples with the config `configs/taxi/attnhp_gen.yaml` 

```
python main_gen_seq.py
```
NOTE: in `configs/taxi/attnhp_gen.yaml`, one needs to setup generation specs and update `pretrained_model_dir` to be the dir of the model trained in Step 1.


Then we train the discriminator with the config `configs/taxi/attnhp_disc_bce.yaml` 

```
python main_train_disc.py
```
NOTE: in `configs/taxi/attnhp_disc_bce.yaml`, it is optionally to update the `pretrained_model_dir` to be the dir of the model trained in Step 1. For taxi data it is recommended to do this. For other data not.


Finally we evaluate the hybrid model with the config `configs/taxi/joint_eval.yaml`

```
python main_joint_eval.py
```
NOTE: in `configs/taxi/joint_eval.yaml`, one needs to update the dir of the generator and discriminator trained in Step 1 and Step 3.


## Citing


If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@article{xue2022hypro,
  title={HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences},
  author={Xue, Siqiao and Shi, Xiaoming and Zhang, Y James and Mei, Hongyuan},
  booktitle = neurips,
  year={2022},
  url={https://arxiv.org/abs/2210.01753}
}
```

## Credits

The following repositories are used in our code, either in close to original form or as an inspiration:

- [Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes)
- [Neural Hawkes Particle Smoothing](https://github.com/hongyuanmei/neural-hawkes-particle-smoothing)
- [Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)
