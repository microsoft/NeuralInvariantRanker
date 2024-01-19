# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import multiprocessing
import openai
import time
from tqdm import tqdm
import json
import os

import sys
data_source = sys.argv[1]

def get_embedding(item):
    text, codex_model = item
    if text.strip() == '':
        t = "dummy"
    else:
        t = copy.copy(text)
    sleep_time = 1
    while True:
        try:
            return (text, openai.Embedding.create(
                input=[t], model=codex_model
            )['data'][0]['embedding'])
        except openai.error.RateLimitError as e:
            if 'Please reduce' in str(e):
                t = t[:int(.9 * len(t))]
            time.sleep(sleep_time)
        except openai.error.OpenAIError as e:
            if 'Please reduce' in str(e):
                t = t[:int(.9 * len(t))]
            time.sleep(sleep_time)
        except Exception as e:
            # print(f"Exception in get_codex_response_with_retries {e}")
            # print(type(e), e)
            if 'Please reduce' in str(e):
                t = t[:int(.9 * len(t))]
            time.sleep(sleep_time)
    return (text, None)


codes = set()


for file in tqdm(['train.jsonl', 'valid.jsonl', 'test.jsonl']):
    fn = f"{data_source}/{file}"
    with open(fn) as f:
        data = [json.loads(line.strip()) for line in f]
        for d in data:
            codes.add(d['code'])
            positives = d['positives']
            for p in positives:
                codes.add(p['code'])
            negatives = d['negatives']
            for n in negatives:
                codes.add(n['code'])

codes = list(codes)
print(len(codes))

models = {
    'ada': 'ada-code-search-code',
    'babbage': 'babbage-code-search-code',
    'curie': 'curie-similarity',
    'davinci': 'davinci-similarity',
    'ada_002': 'text-embedding-ada-002'
}

for model_name, model in models.items():
    output = f"{data_source}/embeddings_{model_name}.json"
    if os.path.exists(output):
        continue
    embeddings = {}
    items = [(code, model) for code in codes]
    pool = multiprocessing.Pool(20)
    results = pool.imap(get_embedding, items)
    bar = tqdm(
        results, 
        total=len(codes), 
        desc=f"Embedding {data_source} with {model_name}"
    )
    for (code, emb) in bar:
        bar.update()
        if emb is not None:
            embeddings[code] = emb
    with open(f"{data_source}/embeddings_{model_name}.json", 'w') as f:
        json.dump(embeddings, f)