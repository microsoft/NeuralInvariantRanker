# Ranking LLM-Generated Loop Invariants for Program Verification

Official code release of out EMNLP 2023 work [NeuralInvariantRanker (iRank)](https://aclanthology.org/2023.findings-emnlp.614.pdf).


## About the Ranker
The tool is used for ranking LLM-generated Loop Invariants. We evaluate the invariants from two different LLMs ([`gpt-3.5-turbo`](data/llm-genrated-solutions/gpt-3.5/) and [`gpt-4`](data/llm-genrated-solutions/gpt-4/)). 




## News
- January 2024: Official code release. 

## Step by step guide
- [**Step 1**](#requirement) - Make sure all the requirements are successfully installed.  
- [**Step 2**](#about-the-data) - Process the data. 
- [**Step 3**](#initial-embeddings) - Gather the initial embeddings.
- [**Step 4**](#training-the-ranker) - Train rankers. 
- [**Step 5**](#evaluation-and-visualization) - Evaluate and Visualize the results of the ranker.


## Requirement
The requirements of this project are as follows. Note that, we evaluated the project in [`PyTorch-1.13.1` (with cuda 11.6)](https://pytorch.org/get-started/previous-versions/#v1131) and [`Transformers-4.29.2`](https://github.com/huggingface/transformers/releases/tag/v4.29.2). Other versions of the libraries may need modification of the code, specially the [`trainer.py`](src/ranker/trainer.py) (and perhaps other places). To setup the appropriate library versions, run `pip install -r requirements.txt`. 

## About the data
We provide to version of the data. Follow the steps below to extract the datasets

1. **Training data :** Training data is available as a zip file in the [`data/ranker_data/data.zip`](data/ranker_data/data.zip). Extract the ranker data with `cd data/ranker_data; unzip data.zip;`
2. **Evaluation data :** Evaluation data is available in [`data/llm-genrated-solutions/solutions.zip`](data/llm-genrated-solutions/solutions.zip). Extract the evaluation data with `cd data/llm-genrated-solutions; unzip solutions.zip;`. 

Training data is formatted as `jsonl` file, where each line corresponds to an example. The format of an example is as follows
```json
{
    "problem_id": "<unique id of the problem>",
    "code": "code of the problem, i.e., problem description",
    "positives": [
        {
            "code": "<a candidate invariant>",
            "score": "<score between 0 and 1, denoting how good a candidate invariant is, higher is better.>"
        }
        ...
    ],
    "negatives": [
        {
            "code": "<a candidate invariant>",
            "score": "<score between 0 and 1, denoting how good a candidate invariant is, higher is better.>"
        }
        ...
    ]
}
```
<!-- For example
```json
{
    "problem_id": "ALIA-others__inc-array", 
    "code": "(set-logic ALIA)\n\n(synth-inv inv_fun ((x (Array Int Int))))\n\n(define-fun pre_fun ((x (Array Int Int))) Bool\n    (= 1 (select x 0)))\n(define-fun trans_fun ((x (Array Int Int)) (x! (Array Int Int))) Bool\n    (and (< (select x 0) 100) (= x! (store x 0 (+ (select x 0) 2)))))\n(define-fun post_fun ((x (Array Int Int))) Bool\n    (or (not (>= (select x 0) 100)) (= (select x 0) 101)))\n\n(inv-constraint inv_fun pre_fun trans_fun post_fun)\n\n(check-synth)\n\n",
    "positives": [
        {
            "code": "(define-fun inv_fun ((x (Array Int Int))) Bool (and (>= (select x 0) 1) (<= (select x 0) 101) (= (mod (select x 0) 2) 1)))", 
            "score": 1.0
        }
        ...
    ],
    "negatives": [
        {
            "code": "(define-fun inv_fun ((x (Array Int Int))) Bool (and (<= 1 (select x 0)) (<= (+ (select x 0) 2) 100) (= (mod (select x 0) 2) 1)))", 
            "score": 0.75
        }
        ...
    ]
}
```
Note that, for all positives examples in our training data, score is 1.00, and for the negatives the score denotes how many  -->

## Initial Embeddings
We collect initial embeddings from two different [OpenAI embedding](https://platform.openai.com/docs/guides/embeddings) models. Before starting the training, the initial embeddings must be gathered. We provide [this scrupt](data/get_initial_embeddings.py) to gather the initial embeddings. To run the script, 
```
python data/get_initial_embeddings.py \
    --model_name <MODEL_NAME, default = ada_002> \
    --use_azure --azure_config <AZURE_CONFIG> \ 
    --token_per_minute TOKEN_PER_MINUTE \
    --output_dir <directory where the embeddings will be saved as MODEL_NAME.json, default = data/embeddings>
```
Note that, this script can both use [OpenAI api](https://github.com/openai/openai-python/releases/tag/v1.4.0) and [Azure OpenAI api](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview). For using OpenAI api, the API key must be exported as `OPENAI_API_KEY` environment variable. Run `export OPENAI_API_KEY=<Your API Key>`. For using Azure Open AI api, a azure config json file is needed with the above script (`--azure_config`). The Config file should be formatted as, 
```json
{
    "API_BASE": "<base url of your azure openai endpoint>",
    "API_KEY": "<your key>",
    "ENGINE": "<model/deployment name>",
    "API_VERSION": "<version of the api>"
}
```
While in the paper we experimented with two different embedding models, _i.e.,_ `text-embedding-ada-002` and `davinci-similarity`, the `davinci-similarity` model is deprecated, and hence cannot be collected with the initial embedding script. Please download the initial embeddings from [this link](), if you want to reproduce our results. 

For any other embedding models than OpenAI embeddings, implement your Embedder extending the [`Embedder`](data/get_initial_embeddings.py#L30). Your class should contain the following method 
```py
    def embed(
        self, 
        texts: List[str], 
        existing_embeddings: Optional[Dict[str, List[float]]] = {}
    ) -> Dict[str, List[float]]:
```

## Training the ranker
To train the ranker, 
```sh
bash train_ranker.sh <model name (ada_002/davinci)> <fold id 0/1/2/3/4>
```
For example, running
```sh
bash train_ranker.sh ada_002 0
```
will train a ranker based on the training data in [`data/ranker_data/fold_0`](data/ranker_data/fold_0) and the initial embedding from `text-embedding-ada-002` (initial embeddings should already be computed as [`data/embeddings/ada_002.json`](data/embeddings/ada_002.json)).

### Training parameters:
The training parameters (_e.g.,_ `batch_size`, `max_training_steps`, `evaluation_strategy`, etc. are controlled by an external `json` file.) The file is passed to the training in [`script/train_ranker.sh`](script/train_ranker.sh). By default, [`configs/ranker_config.json`](configs/ranker_config.json) is used for the training. Change this file for experimenting with other parameters, or create a new file and update [this line](script/train_ranker.sh#L25) with appropriate config file path. 

## Evaluation and Visualization
After the training is done, we provide [this notebook](data/gather_metrics.ipynb) to evaluate and visualize the LLM-generated invariants. Note that, we performed **5-fold cross validation** for evaluating the ranker. To get results from this notebook, make sure to train models for **all 5 folds** and both **`ada_002`** amd **`dainci`** models. 

Alternatively, for ranking other problems and corresponding candidate invariants, we refer to [this function](src/ranker/ranker.py#L76). This function signature is as follows
```py
    def rank_invariants(
        self, 
        model: nn.Module,
        examples: List[Dict[str, Any]],
        _cache: Dict[str, List[float]] = None,
        bar_tqdm=None
    ):
```
The `model` is the trained ranker model, `examples` are a list of ranking examples formatter as below, and `_cache` is the initial embeddings dictionary, that **must** contain the initial embeddings for the problem and invariants in the examples. Each of the example for this function should be
```
{
    "problem": <text of the problem description>,
    "invariants": [
        {
            "code": "<text of the candidate invariant>"
        },
        ...
    ]
}
```
The function will reorder the `"invariants"` list in each of the examples and add two more keys (_i.e.,_ `"code_vector"`, and `"similarity"`) for each invariant. In addition, the ranker will populate the `"code_vector"` for the `"problem"` as well. These `"code_vector"`s are the trained embeddings by the ranker model. 

## Invariant Generation
While the focus of this project to train ranker for invariants, we provide the prompt templete that we used to generate invariants using LLM in [`src/codex_models/chat.py`](src/codex_models/chat.py#L98). 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
