# ADRec

## Quick Start
1. Install the dependencies by running the following command:

```
git checkout dev
uv sync
```

2. Run the following command to start training the model with a specified category:
    
```
CUDA_VISIBLE_DEVICES=0 uv run main.py --category=Sports_and_Outdoors
```

Available categories:
* `Sports_and_Outdoors`
* `Beauty`
* `Toys_and_Games`
* `CDs_and_Vinyl`

Note that:
1. The datasets will be automatically downloaded once the `category` argument is specified.
2. All hyperparameters can be specified via command line arguments. Please refer to:
    * `genrec/default.yaml`
    * `genrec/datasets/AmazonReviews2014/config.yaml`
    * `genrec/models/TigerDiff/config.yaml`

We provide the default training script `train_tigerdiff_cd.sh`, `train_tigerdiff_sports.sh`, `train_tigerdiff_beauty.sh`, and `train_tigerdiff_toys.sh` for training the model on different categories.

> [!IMPORTANT]
> TODO: Tune the hyperparameters.


