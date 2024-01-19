import copy
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

if True:
    project_dir = os.path.abspath(os.path.join(__file__, "../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src import util
    from src.config import Config
    from src.util import OverLengthError
    from src.codex_models.model_base import InvGenModelBase
    

generation_model = None
config = None

def generate(
    problem_details: Dict[str, Any],
):
    global generation_model
    global config
    if generation_model is None:
        raise
    logger = util.get_logger(
        name=config.logger_name,
        include_time_in_output_file=config.include_time_in_logger
    )
    output_dir = problem_details['output_dir']
    pidx = problem_details['problem_id']
    p = problem_details['problem_name']
    problem_file_path = problem_details['problem_file_path']
    total_problems = problem_details['total_problems']
    config = problem_details['config']

    logger.enable_log = not config.disable_log
    os.makedirs(output_dir, exist_ok=True)
    detailed_result_directory = os.path.join(
        output_dir, "details")
    os.makedirs(detailed_result_directory, exist_ok=True)
    invariant_directory = os.path.join(output_dir, "invariants")
    os.makedirs(invariant_directory, exist_ok=True)
    if not config.parallel: logger.info(f"Processing problem {p}")
    detailed_result_path = os.path.join(
        detailed_result_directory, p + ".json"
    )
    problem_definition = open(problem_file_path).read()
    c_problem_definition = None
    if config.add_c_file_content:
        # read the json file using the json directory
        json_file = config.parsed_json_folder + "/" + p + ".sl.json"
        # read the C string from the json file looking for src_info and file_content fields
        json_data = json.load(open(json_file))
        # check if the json has the src_info and file_content fields
        if "src_info" in json_data and "file_content" in json_data["src_info"]:
            c_problem_definition = json_data["src_info"]["file_content"]

    time_spent, trial_count, total_call_to_z3 = 0, 0, 0
    verified, verified_inv = False, None
    generated_invariants = []
    try:
        results = generation_model.solve_problem(
            problem_definition=problem_definition,
            c_problem_definition=c_problem_definition,
            problem_file_path=problem_file_path,
            problem_id=pidx,
            total_problems=total_problems,
        )
        generated_invariants = results['generated_invariants']
        verified_inv = results['verified_inv']
        time_spent = results['time_spent']
        verified = results['verified']
        trial_count = results['trial_count']
        total_call_to_z3 = results['total_call_to_z3']
    except OverLengthError as e:
        logger.warn(f"OverLengthError: {e}")
        verified = "OverLengthError"
    stat = {
        'problem': p,
        'verified': verified,
        'inv': verified_inv,
        'total_time': time_spent,
        'num_trials': trial_count,
        'total_call_to_z3': total_call_to_z3,
    }
    if verified == True:
        liv = re.sub('[ \t\n]+', ' ', verified_inv)
        if logger.enable_log:
            logger.info(
                p,
                liv, 
                f"SUCCESS: trial={trial_count} and z3_call={total_call_to_z3}",
                sep='\n'
            )
        with open(os.path.join(invariant_directory, p + ".inv"), 'w') as f:
            f.write(verified_inv)
            f.close()
    else:
        if logger.enable_log:
            logger.info(f"Verified = {verified}")
            logger.info(
                p,
                f"FAILED trial={trial_count} and z3_call={total_call_to_z3}",
                sep='\n'
            )
    with open(detailed_result_path, 'w') as f:
        stat_to_write = copy.copy(stat)
        stat_to_write['generated_invariants'] = generated_invariants
        f.write(json.dumps(stat_to_write, indent=4))
        f.close()
    logger.info("#" * 100)
    return stat


class InvariantGenerator:
    def __init__(
        self,
        model,
        output_dir: str,
        _config: Config
    ):
        assert isinstance(model, InvGenModelBase)
        self.model = model
        self.output_dir = output_dir
        self.config = _config
        global generation_model
        generation_model = self.model
        global config
        config = self.config


    def generate_invariants(
        self,
        problems: Dict[str, str],
        skip_existing: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        statistics = []
        problem_names = sorted(problems.keys())
        detailed_result_directory = os.path.join(
            self.output_dir, "details"
        )
        problems_to_consider = [{
            'output_dir': self.output_dir,
            'config': self.config,
            'problem_id': pidx,
            'problem_name': p,
            'problem_file_path': problems[p],
            'total_problems': len(problem_names),
        } for pidx, p in enumerate(problem_names) if not (
            skip_existing and os.path.exists(
                os.path.join(
                    detailed_result_directory, p + ".json")
            )
        )]
        if self.config.parallel:
            pool = Pool(
                processes=max(cpu_count(), self.config.workers),
            )
            result_map = pool.imap(
                func=generate,
                iterable=problems_to_consider,
                chunksize=1,
            )
        else:
            result_map = map(generate, problems_to_consider)
        idx = 0
        bar = tqdm(result_map, total=len(problems_to_consider),
            desc=f"Processing problem {idx}/{len(problems_to_consider)}")
        for stat in bar:
            idx += 1
            bar.set_description(f"Processing problem {idx}/{len(problems_to_consider)}")
            # bar.update()
            statistics.append(stat)
        
        with open(os.path.join(self.output_dir, "statistics.json"), 'w') as f:
            f.write(json.dumps(statistics, indent=4))
            f.close()
        return statistics
