import copy
import json
import os
import sys
import time

from typing import Any, Optional, List, Dict, Tuple, Union
import openai
import tiktoken

if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../../.."))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    from src import util
    from src.config import Config
    from src.util import OverLengthError
    from src.ligen.parse import (
        identify_ssa_and_nonssa_vars_in_invariant as variable_separator,
        change_problem_exclude_non_ssa
    )

logger = None


def spent_time(start_time: float) -> float:
    return time.time() - start_time


def get_model_encoding(model_name):
    if model_name.startswith('gpt-'):
        return 'cl100k_base'
    elif 'code-davinci' in model_name or 'text-davinci' in model_name:
        return 'p50k_base'
    else:
        return 'r50k_base'


class InvGenModelBase:
    def __init__(
        self,
        config: Config,
        model_name: Optional[str] = 'gpt-3.5-turbo'
    ):
        global logger
        logger = util.get_logger(
            name=config.logger_name,
            include_time_in_output_file=config.include_time_in_logger
        )
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = config.model_name
        self.encoding_name = get_model_encoding(self.model_name)
        self.encoding = tiktoken.get_encoding(self.encoding_name)
        self.config = config
        self.max_model_capacity_tokens = config.max_model_capacity_tokens
        if config.use_azure:
            assert config.azure_config_file is not None, "Must provide azure_config_file"
            azure_config = json.load(open(config.azure_config_file))
            logger.info("Using Azure OpenAI API")
            openai.api_type = "azure"
            openai.api_base = azure_config["AZURE_BASE_URL"]
            openai.api_version = azure_config["AZURE_API_VERSION"]
            openai.api_key = azure_config["AZURE_OPENAI_API_KEY"]
            self.deployment_name = azure_config["AZURE_DEPLOYMENT"]
        else:
            logger.info("Using OpenAI public API")
            if hasattr(config, "openai_api_key"):
                openai.api_key = config.openai_api_key
            else:
                key = os.getenv("OPENAI_KEY")
                if key is None:
                    raise ValueError(
                        "OPENAI_KEY is not set in the environment.")
                openai.api_key = key

    def stop_criteria_met(self, trial_count, start_time):
        return ((
            self.config.use_num_trial_as_limit and
            trial_count >= self.config.num_trial_limit
        ) or
            (
            not self.config.use_num_trial_as_limit and
            spent_time(start_time) >= self.config.max_allowed_time_per_example
        ))

    def solve_problem(
        self,
        problem_definition: str,
        problem_file_path: str,
        c_problem_definition: Optional[str] = None,
        problem_id: Optional[int] = -1,
        total_problems: Optional[int] = -1,
    ):
        raise NotImplementedError("Child class must implement this method.")

    def generate_prompt(
        self,
        problem: str,
        c_problem: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> List[Dict[str, str]]:
        if self.config.change_problem_to_redact_ssa:
            problem = change_problem_exclude_non_ssa(
                problem_definition=problem)
        lines = problem.split("\n")
        start_expr = None
        for l in lines:
            if l.strip().startswith("(synth-inv") and l.strip().endswith(")"):
                start_expr = l.strip()[len("(synth-inv"):-1].strip()
                start_expr = "(define-fun " + start_expr + " Bool ("
                break
        if start_expr is None:
            raise ValueError(f"Malformed problem in \n{problem}\n")
        
        msg = ""
        if c_problem is not None:
            msg += "Generate the loop invariant for the following C program with a loop:\n\n" +\
                f"{c_problem}\n\n"
        
        # add first two lines only when sygus format is enabled
        if not self.config.drop_sygus_formula:
            msg += "Here is a loop invariant synthesis problem in SyGus format.\n\n" +\
            f"{problem}\n\nSynthesize a necessary and sufficient invariant.\n\n"
        
        msg += f"Start the invariant with \"{start_expr}\" and end with \")\" \n\n" +\
            "Surround only the invariant with <code> and </code>. " +\
            "You don't need to explain the invariant, just synthesize it.\n"

        if self.config.redact_ssa_through_prompt:
            ssa, non_ssa = variable_separator(problem_definition=problem)
            msg += f"ONLY use the following variables in the invariant body: " +\
                f"{non_ssa}\n DO NOT use any of the following variable in the invariant body: " +\
                f"{ssa}\n\n"

        prompt = [{
            'role': 'user',
            'content': msg
        }]
        if not self.is_prompt_within_token_limit(prompt):
            raise OverLengthError(
                f"Prompt is too long: {self.count_tokens(msg)}"
            )
        if instruction is not None:
            new_prompt = copy.deepcopy(prompt)
            new_prompt[0]['cotent'] += instruction
            if self.is_prompt_within_token_limit(new_prompt):
                prompt = new_prompt
        system_msg = 'You are a Loop Invariant Synthesis GPT.' +\
            'Reply precisely with a necessary and sufficient invariant.'
        new_prompt = [
            {
                'role': 'system',
                'content': system_msg
            }
        ] + prompt
        if self.is_prompt_within_token_limit(new_prompt):
            prompt = new_prompt
        return prompt

    def query_with_retries(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_generations: Optional[int] = 1,
        auto_truncate: Optional[bool] = False,
        start_time: Optional[float] = None,
    ) -> List[Dict]:
        raise NotImplementedError("Child class must implement this method.")

    def truncate(
        self,
        prompt,
    ):
        raise NotImplementedError("Child class must implement this method.")

    def generate_invariants(
        self,
        problem: str,
        inv_cache,  # a set of candidate invariants
        instruction: Optional[str] = None,
        max_generations: Optional[int] = 1,
        auto_truncate: Optional[bool] = False,
        start_time: Optional[float] = None,
    ) -> List[str]:
        raise NotImplementedError("Child class must implement this method.")

    def generate_invariant_from_prompt(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_generations: Optional[int] = 1,
        auto_truncate: Optional[bool] = False,
    ) -> Tuple[List[str], Any]:
        raise NotImplementedError("Child class must implement this method.")

    def add_to_prompt(
        self,
        existing_prompt: List[Dict[str, str]],
        previous_reply: Optional[Union[str, Dict[str, str]]],
        feedback: util.FeedBack
    ):
        raise NotImplementedError("Child class must implement this method.")
