# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import sys
from typing import Optional, List, Dict, Union
import openai
import time
import tenacity

from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../../.."))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    from src import util
    from src.ligen.loopinv import verify
    from src.config import Config
    from src.codex_models.model_base import InvGenModelBase
    from src.codex_models.utils import (
        filter_invariant_based_on_format,
        filter_invariant_based_on_tag,
        new_invs_and_update_cache
    )

logger = None
global_definition_map = {}
global_count_ssas = {}


class VanillaChat(InvGenModelBase):
    def __init__(
        self,
        config: Config,
        model_name: Optional[str] = 'gpt-3.5-turbo'
    ):
        super().__init__(config, model_name)
        global logger
        logger = util.get_logger(
            name=config.logger_name,
            include_time_in_output_file=config.include_time_in_logger
        )

    def count_tokens(self, s: str) -> int:
        return len(self.encoding.encode(s))

    def is_prompt_within_token_limit(
        self,
        prompt: List[Dict[str, str]]
    ) -> bool:
        token_count = self.count_tokens(str(prompt))
        return token_count <= (
            self.config.max_model_capacity_tokens -
            self.config.max_tokens
        )

    def add_to_prompt(self, existing_prompt, feedback):
        raise NotImplementedError(
            "This is vanilla chat, no feedback is needed.")

    def query_model_with_tenacity(self, prompt: str, n):
        parameters = {
            "messages": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "n": n
        }
        if self.config.use_azure:
            parameters["engine"] = self.deployment_name
        else:
            parameters["model"] = self.model_name
        response = None
        # TODO: Implement the openai call with the updated openai API.
        return response

    def query_with_retries(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_generations: Optional[int] = 1,
        auto_truncate: Optional[bool] = False,
        start_time: Optional[float] = None,
    ) -> List[Dict]:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        responses = []
        response_len = 0
        delay = int(120.0 / self.config.rate_limit_per_minute)
        logger.info(f"Delay: {delay}")
        while response_len < max_generations:
            if not self.config.parallel: 
                logger.info(f"Current Length: {response_len} {max_generations}")
            n = min(max_generations - response_len,
                    self.config.rate_limit_per_minute)
            retry_count = 0
            response = None
            while retry_count < self.config.num_retries:
                retry_count += 1
                try:
                    response = self.query_model_with_tenacity(prompt, n)
                    break
                except Exception as e:
                    logger.warn(f'Delay {delay}\t{type(e)}\t{e}')
                    time.sleep(delay)
            if response is None:
                continue
            responses.extend(response['choices'])
            response_len += len(response['choices'])
            if start_time is not None:
                time_spent = time.time() - start_time
                if (
                    not self.config.use_num_trial_as_limit and
                    time_spent >= self.config.max_allowed_time_per_example
                ):
                    return responses
        return responses

    def truncate(
        self,
        prompt,
    ):
        # TODO: implement this
        return prompt

    def generate_invariants(
        self,
        problem: str,
        inv_cache,
        c_problem_stmt: Optional[str] = None,
        instruction: Optional[str] = None,
        max_generations: Optional[int] = 1,
        auto_truncate: Optional[bool] = False,
        start_time: Optional[float] = None,
    ) -> List[str]:
        prompt = self.generate_prompt(
            problem, instruction=instruction, c_problem=c_problem_stmt
        )
        responses = self.query_with_retries(
            prompt=prompt,
            max_generations=max_generations,
            auto_truncate=auto_truncate,
            start_time=start_time
        )
        if not self.config.parallel:
            logger.info(f"Length of response {len(responses)}")
        invariants = []
        for m in responses:
            try:
                inv = m['message']['content']
                invariants.append(inv)
            except Exception as e:
                logger.warn(f'{type(e)}\t{e}')
        filtered_invariants = []
        for inv in invariants:
            tag_filtered_inv = filter_invariant_based_on_tag(
                inv, raise_error=False)
            fmt_filtered_inv = filter_invariant_based_on_format(
                inv, raise_error=False)
            filtered_invariants.extend(tag_filtered_inv)
            filtered_invariants.extend(fmt_filtered_inv)
        if not self.config.parallel:
            logger.info(f"Identified {len(filtered_invariants)} after filtering")
        if self.config.deduplicate:
            filtered_invariants = new_invs_and_update_cache(
                filtered_invariants, inv_cache, self.config.semantic_deduplicate
            )
        if not self.config.parallel:
            logger.info(
                f"Identified {len(filtered_invariants)} deduplication." +\
                f" Inv_cache size: {len(inv_cache)}")
        return filtered_invariants, prompt

    def solve_problem(
        self,
        problem_definition: str,
        problem_file_path: str,
        c_problem_definition: Optional[str] = None,
        problem_id: Optional[int] = -1,
        total_problems: Optional[int] = -1,
    ):
        global global_definition_map
        global_definition_map[problem_definition] = problem_file_path
        # declare a cache variable and pass it into and out of the generate_invariants
        # function
        cand_inv_cache = set()
        time_spent = 0
        verified = False
        trial_count = 0
        total_call_to_z3 = 0
        verified_inv = None
        start_time = time.time()
        generated_invariants = []
        bar = tqdm(total=self.config.num_trial_limit)
        while (
            not (self.config.stop_early and verified) and
            not self.stop_criteria_met(trial_count, start_time)
        ):
            # if not self.config.parallel:
            #     logger.info("Asking the model to generate invariants...")
            filtered_invariants, prompt = self.generate_invariants(
                problem=problem_definition,
                c_problem_stmt=c_problem_definition,
                max_generations=self.config.max_generations,
                auto_truncate=self.config.auto_truncate,
                inv_cache=cand_inv_cache,
                start_time=start_time
            )
            for inv in filtered_invariants:
                trial_count += 1
                local_verified, duration, feedback = verify(
                    inv, problem_file_path,
                    timeout=self.config.verification_timeout
                )
                generated_invariants.append({
                    'prompt': prompt,
                    'inv': inv,
                    'verified': local_verified,
                    'duration': duration,
                    'feedback': feedback,
                })
                total_call_to_z3 += feedback['call_to_z3']
                if local_verified and verified_inv is None:
                    verified = True
                    verified_inv = inv
                    if self.config.stop_early:
                        break
                if self.stop_criteria_met(trial_count, start_time):
                    break
                time_spent = time.time() - start_time
                bar.update()
        global global_count_ssas
        count_map = {}
        for k in global_count_ssas:
            c = len(global_count_ssas[k])
            if c > (0 if self.config.change_problem_to_redact_ssa else 1):
                count_map[k] = c
        # logger.info(json.dumps(
        #     count_map, indent=4
        # ))
        return {
            'generated_invariants': generated_invariants,
            'verified_inv': verified_inv,
            'verified': verified,
            'time_spent': time_spent,
            'trial_count': trial_count,
            'total_call_to_z3': total_call_to_z3,
            'ssa_violations': global_count_ssas
        }


if __name__ == '__main__':
    openai.api_key = os.environ['OPENAI_API_KEY']
    respose = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Here is a loop invariant synthesis problem in SyGus format.\n\n",
    )
    print(respose)
