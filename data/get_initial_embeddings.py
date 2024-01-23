import json 
import argparse
import os
from time import sleep
import numpy as np
import tiktoken
import openai
from tqdm import tqdm
import threading
from typing import List, Dict, Optional

used_tokens_in_one_minute = 0
this_file_dir = os.path.dirname(os.path.abspath(__file__))

class TokenCounterResetter(threading.Thread):
    def __init__(self):
        super().__init__()
        self._kill = threading.Event()

    def run(self):
        while True:
            is_killed = self._kill.wait(60)
            if is_killed:
                break
            global used_tokens_in_one_minute
            used_tokens_in_one_minute = 0
            
    def kill(self):
        self._kill.set()

class Embedder:
    def __init__(self, model_name, **kwargs) -> None:
        if model_name == 'ada_002':
            self.model = "text-embedding-ada-002"
        else:
            raise NotImplementedError("Model not supported")
        self.use_azure = False
        self.azure_config = None
        if "use_azure" in kwargs:
            self.use_azure = kwargs["use_azure"]
            if self.use_azure:
                assert "azure_config" in kwargs
                self.azure_config = json.load(open(kwargs["azure_config"]))
        self.openai_client = self.setup_api()
        if "token_per_minute" in kwargs:
            self.token_per_minute = kwargs["token_per_minute"]
        else:
            self.token_per_minute = 10000
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        if "max_prompt_size" in kwargs:
            self.max_prompt_size = kwargs["max_prompt_size"]
        else:
            self.max_prompt_size = 8191
            
    def setup_api(self, num_trials=3):
        if self.use_azure:
            assert self.azure_config is not None, "Azure config is not provided"
            config = self.azure_config
            openai_client = openai.AzureOpenAI(
                azure_endpoint=config["API_BASE"],
                api_version=config["API_VERSION"],
                api_key=config["API_KEY"],
                timeout=120,
                max_retries=num_trials
            )
        else:
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY is not set")
            openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=120,
                max_retries=num_trials
            )
        return openai_client
    
    def get_batch_embeddings(self, text_list):
        while True:
            try:
                response = self.openai_client.embeddings.create(
                    input=text_list,
                    model="text-embedding-ada-002"
                )
                break
            except Exception as e:
                print(e)
                sleep(1)
                continue
        embeddings = [d.embedding for d in response.data]
        return embeddings, response.usage.total_tokens
    
    def embed(
        self, 
        texts: List[str], 
        existing_embeddings: Optional[Dict[str, List[float]]] = {}
    ) -> Dict[str, List[float]]:
        prepared_inputs = []
        for t in texts:
            if t in existing_embeddings:
                continue
            tokens = self.tokenizer.encode(t)[:self.max_prompt_size]
            input_text = self.tokenizer.decode(tokens)
            prepared_inputs.append((t, input_text, len(tokens)))
        batches  = []
        batch = []
        running_tokens = 0
        for pinp in prepared_inputs:
            if running_tokens + pinp[2] > self.token_per_minute:
                batches.append(batch)
                batch = []
                running_tokens = 0
            batch.append(pinp)
            running_tokens += pinp[2]
        if len(batch) > 0:
            batches.append(batch)
        bar = tqdm(batches, desc="Used tokens in last one minute: %5d" % 0)
        for b in bar:
            original_texts = [x[0] for x in b]
            input_texts = [x[1] for x in b]
            batch_embeddings, tokens_count = self.get_batch_embeddings(input_texts)
            global used_tokens_in_one_minute
            used_tokens_in_one_minute += tokens_count
            for oi, be in zip(original_texts, batch_embeddings):
                existing_embeddings[oi] = be
            sleep_counter = 0    
            while used_tokens_in_one_minute != 0:
                sleep_counter += 1
                bar.set_description(
                    "Used tokens in last one minute: %5d. Sleeping for %s seconds" % (used_tokens_in_one_minute, sleep_counter)
                )
                sleep(1)
        return existing_embeddings

def gather_unique_code_from_ranker_data(rd_file):
    lines = open(rd_file).readlines()
    all_codes = set()
    for l in lines:
        d = json.loads(l)
        all_codes.add(d["code"])
        for p in d["positives"]:
            all_codes.add(p["code"])
        for n in d["negatives"]:
            all_codes.add(n["code"])
    return all_codes  

def gather_all_inputs():
    all_inputs = set()
    ranker_data_dir = os.path.join(this_file_dir, 'ranker_data')
    assert os.path.exists(ranker_data_dir), "ranker_data directory does not exist"
    num_folds = 5
    for i in tqdm(range(num_folds), desc="Gathering inputs from ranker data"):
        fold_dir = os.path.join(ranker_data_dir, f"fold_{i}")
        assert os.path.exists(fold_dir), f"fold_{i} does not exist, please unzip ranker_data/data.zip file"
        all_inputs.update(gather_unique_code_from_ranker_data(os.path.join(fold_dir, "train.jsonl")))
        all_inputs.update(gather_unique_code_from_ranker_data(os.path.join(fold_dir, "test.jsonl")))
        all_inputs.update(gather_unique_code_from_ranker_data(os.path.join(fold_dir, "valid.jsonl")))
    
    problem_folder = os.path.join(this_file_dir, 'problems/in_scope')
    problem_files = [os.path.join(problem_folder, f) for f in os.listdir(problem_folder) if f.endswith('.sl')]
    for pf in tqdm(problem_files, desc="Gathering inputs from in_scope problems"):
        all_inputs.add(open(pf).read())
    
    generated_solution_directory = os.path.join(this_file_dir, "llm-genrated-solutions")
    solution_folders = [
        os.path.join(generated_solution_directory, f, "details") 
        for f in os.listdir(generated_solution_directory) 
        if os.path.isdir(os.path.join(generated_solution_directory, f))
    ]
    for sf in tqdm(solution_folders, desc="Gathering inputs from LLM generated solutions"):
        for f in os.listdir(sf):
            if f.endswith(".json"):
                invariants = json.load(open(os.path.join(sf, f)))["generated_invariants"]
                for i in invariants:
                    all_inputs.add(i["inv"])
                    
    return list(all_inputs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', '-m', type=str, choices=['ada_002'],
        default='ada_002', help='Name of the OpenAI embedding Model Name'
    )
    parser.add_argument(
        '--use_azure', '-a', action='store_true', 
        help='Use Azure API. If Azure API is used, the following arguments are required: azure_config'
    )
    parser.add_argument(
        '--azure_config', '-c', type=str, 
        help='Azure Config file in json format. The json should contain the following keys: '
        'API_BASE: Base URL of the hosted model, '
        'API_VERSION: Version of the Azure OpenAI API, '
        'API_KEY: API key for the Azure OpenAI API'
    )
    parser.add_argument('--token_per_minute', '-t', type=int, default=10000, help='Token per minute limit')
    
    parser.add_argument(
        '--output_dir', '-o', type=str, 
        default=os.path.join(this_file_dir, 'embeddings'), help='Output directory'
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_json_name = os.path.join(output_dir, f"{args.model_name}.json")
    if os.path.exists(output_json_name):
        print(f"Loading already computed embeddings from {output_json_name}")
        already_computed = json.load(open(output_json_name))
    else:
        already_computed = {}
    
    embedder = Embedder(args.model_name, use_azure=args.use_azure, azure_config=args.azure_config, token_per_minute=args.token_per_minute)
    all_inputs = gather_all_inputs()
    
    all_input_set = set(all_inputs)
    already_computed_set = set(already_computed.keys())
    to_compute = list(all_input_set - already_computed_set)
    
    print(f"Total inputs: {len(all_input_set)}")
    print(f"Already computed: {len(already_computed_set)}")
    print(f"To compute: {len(to_compute)}")
    if len(to_compute) > 0:
        counter = TokenCounterResetter()
        counter.start()
        already_computed = embedder.embed(to_compute, already_computed)
        counter.kill()
        
        with open(os.path.join(output_dir, f"{args.model_name}.json"), 'w') as f:
            json.dump(already_computed, f)
    
    