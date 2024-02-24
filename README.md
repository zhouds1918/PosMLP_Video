# PosMLP_Video 

### Installation

Please follow the installation instructions in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.

### Training

1. Download the pretrained models into the pretrained folder.

2. Simply run the training code as followed:
  ```shell
 python3 tools/run_net.py --cfg configs/SSV1/SSV1_MLP_S16.yaml DATA.PATH_PREFIX /data/zhouds/some_some_v1/img OUTPUT_DIR /data/zhouds/tiaoshi
  ```


**[Note]:**

- You can change the configs files to determine which type of the experiments.

- For more config details, you can read the comments in `slowfast/config/defaults.py`.

- To avoid **out of memory**, you can use `torch.utils.checkpoint` (will be updated soon):



### Testing

We provide testing example as followed:

```
### SomethingV1&V2
```shell
python3 tools/run_net.py   --cfg configs/SSV1/SSV1_MLP_S16.yaml DATA.PATH_PREFIX your_data_path TEST.NUM_ENSEMBLE_VIEWS 1 TEST.NUM_SPATIAL_CROPS 3 TEST.CHECKPOINT_FILE_PATH your_model_path OUTPUT_DIR your_output_dir
```

Specifically, we need to set the number of crops&clips and your checkpoint path then run multi-crop/multi-clip test:


 Set the number of crops and clips:

   **Multi-clip testing for Kinetics**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 4
   TEST.NUM_SPATIAL_CROPS 1
   ```

   **Multi-crop testing for Something-Something**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 1
   TEST.NUM_SPATIAL_CROPS 3
   ```

 You can also set the checkpoint path via:

   ```shell
   TEST.CHECKPOINT_FILE_PATH your_model_path
   ```

## Acknowledgement

This repository is built based on [SlowFast](https://github.com/facebookresearch/SlowFast) and [Uniformer](https://github.com/Sense-X/UniFormer) repository.



 
