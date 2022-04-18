# Table of Contents

- [Real example: `admin_torch` on WMT'14 En-De](#admin_torch-on-WMT14-En-De)
- [Comparison with original Admin and DeepNet](#comparison-with-original-admin-and-deepnet-on-wmt17-en-de)

# Real example: `admin-torch` on WMT'14 En-De 

As an example, we apply `admin_torch` to `fairseq` and train Transformer on WMT'14 En-De. 

> Note: the efforts to incorporate `admin-torch` into fairseq are summarized as [this commit](https://github.com/LiyuanLucasLiu/fairseq/commit/33ad76ae5dc927bc32b9594f9728a367c45680bb):

## 1. Pre-processing

### 1.1. Data Preparation

please refer to [the Transformer-Clinic repo](https://github.com/LiyuanLucasLiu/Transformer-Clinic/blob/master/pre-process/wmt14en-de.sh) for data preparation. 

### 1.2. Package Install

```
pip install admin_torch==0.1.0
pip uninstall fairseq
pip install https://github.com/LiyuanLucasLiu/fairseq/archive/refs/tags/admin-torch.zip
```

## 2. Training and Evaluation

### 2.1. Training
```
bash train_wmt_en-de.sh $PATH-to-WMT14 $NUBMER_LAYER $OUTPUT_PATH
```
Not that `$PATH-to-WMT14` is the path to the `wmt14_en_de_joined_dict` data
folder from data preparation. `$NUMBER_LAYER` is the encoder/decoder layer number.
`$OUTPUT_PATH` is the path where you want to save your checkpoints. 

### 2.2. Evaluation
```
bash eval_wmt_en-de.sh $PATH-to-WMT14 NONE $OUTPUT_PATH
```
Not that `$PATH-to-WMT14` is the path to the `wmt14_en_de_joined_dict` data folder
from data preparation. `$OUTPUT_PATH` is the path used in the training step. 

## 3. Pre-trained Weights

| Layer Number | BLEU  | PATH |
|--------------|-------|------|
| 6L-6L        | 27.84 | TBD |
| 18L-18L      | 28.91 | TBD |
| 100L-100L*   | 29.65 | TBD |

*: trained with the [huge-batch-size setting](#omparison-with-original-admin-and-deepnet-on-wmt17-en-de),
but only for 40 epochs, due to the huge cost of the training. 

## 4. Discussion on the `admin-torch` setting. 

`admin-torch.as_module` can be configured by changing `output_change_scale` and
`as_parameter`. `output_change_scale` can be set to `O(1)` for additional stability, but
results in a performance drop in our experiments. `as_parameter` can be set to `False` to
make `omega` (the shortcut connection scaler) as a constant (no updates). Their performance are listed
as below:

|    Layer Number | Output Change | Omega           | BLEU  |
|-----------------|---------------|-----------------|-------|
| 6L-6L           | O(1)          | as a constant   | 27.71 |
| 6L-6L           | O(1)          | as a parameter  | 27.79 |
| 6L-6L           | O(logn)       | as a constant   | 27.83 |
| 6L-6L           | O(logn)       | as a parameter  | 27.84 |
| 18L-18L         | O(1)          | as a constant   | 28.66 |
| 18L-18L         | O(1)          | as a parameter  | 28.89 |
| 18L-18L         | O(logn)       | as a constant   | 28.78 |
| 18L-18L         | O(logn)       | as a parameter  | 28.91 |

# Comparison with original Admin and DeepNet on WMT'17 En-De

We choose to make comparisons with DeepNet and the original Admin implementation on WMT'17 En-De,
the dataset used in the DeepNet paper. 

We noticed that the training configuration in the DeepNet paper is different from the setting used
in the original Admin repo. Their major difference is the batch size (i.e., regular batch size and
huge batch size). We refer the setting used in the DeepNet paper as `Huge batch size (128x4096)`, 
and they refer the setting with changed batch size as `Regular batch size (8x4096)`. 

We can find that they can only work on their own settings. 

|               | Regular batch size (8x4096) |  Huge batch size (128x4096) |
|---------------|--------------------|------------------|
| [Original Admin](https://github.com/LiyuanLucasLiu/Transformer-Clinic)| ✅ | ❌ |
| [DeepNet](https://arxiv.org/abs/2203.00555) | ❌ | ✅ |
| `admin-torch` | ✅ | ✅ |

Here, we re-implemented admin as `admin-torch`, and we can find that the new `admin-torch`
implementation works well on both settings. 

All implementations are publicly released (elaborated as below). 


## 1. Data Preparation
Please refer to the DeepNet paper for data preparation. Here we used the same data shared by the 
DeepNet team. 

## 2. Original Admin and DeepNet

### 2.1. Implementation Download and Code Install
```
pip uninstall fairseq
git clone https://github.com/LiyuanLucasLiu/Transformer-Clinic.git
cd Transformer-Clinic/fairseq
pip install --editable .
```
 
### 2.2. Training

#### 2.2.1. Original Admin
```
# Before running the training, the original admin requires to do a profilling 
# of the network. The profilling result for 100L-100L is included in this repo
# (i.e., example/profile.ratio.init). The command to generate this profilling 
# can be found at https://github.com/LiyuanLucasLiu/Transformer-Clinic/blob/master/nmt-experiments/wmt14_en-de.md#100l-100l-admin-without-any-hyper-parameter-tuning

# regular batch size (4096 x 8)
bash train_wmt_en-de.sh $PATH-to-WMT17 100 $OUTPUT_PATH_REG "--init-type adaptive"

# huge batch size (4096 x 128)
bash train_wmt_en-de_huge.sh $PATH-to-WMT17 100 $OUTPUT_PATH_HUG "--init-type adaptive"

# evaluate 
bash eval_wmt_en-de.sh $PATH-to-WMT17 none $OUTPUT_PATH_HUG/REG 45 10
```

#### 2.2.2. DeepNet
```
# regular batch size (4096 x 8)
bash train_wmt_en-de.sh $PATH-to-WMT17 100 $OUTPUT_PATH_REG "--init-type deepnet"

# huge batch size (4096 x 128)
bash train_wmt_en-de_huge.sh $PATH-to-WMT17 100 $OUTPUT_PATH_HUG "--init-type deepnet"

# evaluate 
bash eval_wmt_en-de.sh $PATH-to-WMT17 none $OUTPUT_PATH_HUG/REG 45 10
```

## 3 `torch-admin`

### 3.1 Package Install

```
pip install admin_torch==0.1.0
pip uninstall fairseq
pip install https://github.com/LiyuanLucasLiu/fairseq/archive/refs/tags/admin-torch.zip
```

### 3.2 Training and Evaluation

```
# regular batch size (4096 x 8)
bash train_wmt_en-de.sh $PATH-to-WMT17 100 $OUTPUT_PATH_REG

# huge batch size (4096 x 128)
bash train_wmt_en-de_huge.sh $PATH-to-WMT17 100 $OUTPUT_PATH_HUG

# evaluate 
bash eval_wmt_en-de.sh $PATH-to-WMT17 none $OUTPUT_PATH_HUG/REG 45 10
```