import os
import sys
from typing import Dict
import config
import util
import re
logger = None

if True:
    project_dir = os.path.abspath(os.path.join(__file__, "../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.invgen import InvariantGenerator
    from src.codex_models.chat import VanillaChat
    from src.codex_models.chat_with_feedback import ChatWithFeedback
    from src.codex_models.breadth_limited_chat import BreadthLimitedChatWithFeedback


def filter_examples_based_on_debug_need(
    problems_locations: Dict[str, str],
    config: config.Config
):
    problem_keys = sorted(list(problems_locations.keys()))
    if config.num_examples is not None:
        taken_problems = problem_keys[:config.num_examples]
    elif config.example_indices is not None:
        taken_problems = [
            problem_keys[i] for i in config.example_indices
        ]
    elif config.example_prefix is not None:
        taken_problems = []
        for prefix in config.example_prefix:
            taken_problems += [
                k for k in problem_keys if k.startswith(prefix)
            ]
        taken_problems = sorted(list(set(taken_problems)))
    elif config.example_name_regex is not None:
        taken_problems = [
            k for k in problem_keys if re.match(config.example_name_regex, k)
        ]
    else:
        taken_problems = problem_keys
    return {
        k: problems_locations[k] for k in taken_problems
    }


if __name__ == '__main__':
    args = config.parse_command_line_args()
    config = config.get_config(args)
    logger = util.get_logger(name=config.logger_name,
        include_time_in_output_file=config.include_time_in_logger
    )
    logger.info(config)
    project_dir = os.path.abspath(os.path.join(__file__, "../.."))
    # problems_directory = os.path.join(
        # project_dir, "data/problems/lig-solved")
    problems_directory = config.input_dir
    problems_locations = {
        f[:-3]: os.path.join(root, f) for (root, dirs, files) in
        os.walk(problems_directory)
        for f in files if f.strip().endswith('.sl')
    }
    problems_locations = filter_examples_based_on_debug_need(
        problems_locations, config
    )
    if config.model_name.startswith('gpt-'):
        if config.generation_type == 'depth':
            model = ChatWithFeedback(config=config, model_name=config.model_name)
        elif config.generation_type == 'breadth':
            model = VanillaChat(config=config, model_name=config.model_name)
        else:
            model = BreadthLimitedChatWithFeedback(
                config=config, model_name=config.model_name
            )
    else:
        raise ValueError(f"Unknown model name {config.model_name}")
    
    invgen = InvariantGenerator(model=model, output_dir=config.output_dir, _config=config)
    statistics = invgen.generate_invariants(
        problems=problems_locations, 
        skip_existing=config.do_not_redo_existing
    )
    total_success = sum([1 for s in statistics if s['verified'] == True])
    total_overlen = sum([1 for s in statistics if s['verified'] == 'OverLengthError'])
    avg_num_trials = (sum(
        [s['num_trials'] for s in statistics if s['verified'] == True]
    ) / float(total_success)) if total_success > 0 else "No Success"
    logger.info(f"Total Success : {total_success} / {len(statistics)}")
    logger.info(f"Total OverLengthError : {total_overlen} / {len(statistics)}")
    logger.info(f"Total Failed : {len(statistics) - total_success - total_overlen} / {len(statistics)}")
    logger.info(f"Average Number of Trials : {avg_num_trials}")
