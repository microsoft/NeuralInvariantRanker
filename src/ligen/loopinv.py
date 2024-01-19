# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import os
import sys
import tempfile
import re
import time
import json


if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../../.."))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    from src import util
    # logger = util.get_logger()


def parse_verification_decision(inv_name, output, errors):
    output_lines = re.sub("[\n]+", "\n", output).split("\n")
    problem_properties = json.loads(output_lines[0])
    inv = problem_properties["inv"]
    prog_details = {'problem': problem_properties}
    error = re.sub("[\n]+", "\n", errors)
    if error.strip() != "":
        if "Bad/multiple S-exprs detected" in error:
            return False,  {
                'cause': "Exception",
                'details': ("Expected and generated invariant function names " +\
                        f"do not match. \n\nPlease use {inv['name']} " +\
                        "as generage invariant function name.") \
                                if inv["name"] != inv_name else\
                        ("Bad/multiple S-exprs detected, expecting " +\
                        "invariant as a single valid S-expr."),
                'call_to_z3': 0
            }, prog_details
        elif "reached EOF while in state Parsing_list" in error:
            return False, {
                'cause': "Parsing Error",
                'details': f"reached EOF while in state Parsing_list",
                'call_to_z3': 0
            }, prog_details
        else:
            error_lines = error.split("\n")
            taken_lines = []
            started, finished = False, False
            i = 0
            while not finished and i < len(error_lines):
                line = error_lines[i].strip()
                if "Uncaught exception:" in line: started = True
                elif "Raised at file" in line: 
                    finished = True 
                    break
                elif started: 
                    if line != "": taken_lines.append(line)
                i += 1
            if len(taken_lines) == 0:
                taken_lines = [l for l in error_lines if l.strip() != ""]
            return False, {
                'cause': "Parsing Error",
                'details': "\n".join(taken_lines),
                'call_to_z3': 0
            }, prog_details
    elif "PASS" in output:
        return True, {
            'cause': 'Satisfied',
            'details': f"Verifier found no counter example for {inv_name}",
            'call_to_z3': 4
        }, prog_details
    else:
        parsed_model_results = []
        try:
            output_lines = output_lines[1:]
            for line in output_lines:
                line = line.strip()
                if line == "####":
                    break
                if not line.startswith("{"):
                    continue
                model_result = json.loads(line)
                parsed_model_results.append(model_result)
            faulty_model = parsed_model_results[-1]["model"]
            faulty_model = re.sub("[ \t\n]+", " ", faulty_model)
            faulty_call = parsed_model_results[-2]["condition"]
            faulty_call = re.sub("[ \t\n]+", " ", faulty_call)
            prog_details["evaluated_calls"] = parsed_model_results[:-1]
            prog_details["faulty_model"] = parsed_model_results[-1]
            return False, {
                'cause': 'Un-satisfied',
                'details': f"""The verifier found a counter example """ +\
                        f"""that invalidates the following assertion \n""" +\
                        f"""{faulty_call}\n\n""" +\
                        f"""The counter examples is: \n""" +\
                        f"""{faulty_model}\n""",
                'call_to_z3': len(parsed_model_results) - 1
            }, prog_details
        except:
            False, {
                'cause': 'Un-satisfied',
                'details': 'The invariant fails to satisfy some condition.',
                'call_to_z3': len(parsed_model_results) - 1
            }, prog_details
    return False, {
        'cause': 'Unknown',
        'details': 'The verifier failed to verify the invariant.',
        'call_to_z3': 4
    }, prog_details

def verify(
    inv,
    problem_path,
    z3_path=os.path.join(PROJECT_PATH, "LoopInvGen/_dep/z3"),
    script_path=os.path.join(
        PROJECT_PATH, "LoopInvGen/_build/install/default/bin/lig-verify"
    ),
    timeout=60
):
    logger = util.get_logger()
    assert os.path.exists(problem_path), \
        f'Invalid problem definition path {problem_path}'
    assert os.path.exists(z3_path), \
        f'z3 does not exist in {z3_path}'
    assert os.path.exists(script_path), \
        f'verification script was not found in {script_path}'
    start_time = time.time()

    problem_type = util.get_problem_type(problem_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        inv_file = os.path.join(tmpdir, 'test.inv')
        command = f'{script_path} -z {z3_path} -s {problem_path} {inv_file}'
        fp = open(inv_file, 'w')
        fp.write(inv + "\n")
        fp.close()
        try:
            util.check_z3_parsable(inv, problem_type=problem_type)
        except Exception as e:
            return False, 0, {
                'cause': 'Parsing Error',
                'details': f'The invariant cannot be parsed by z3 with error {e}',
                'call_to_z3': 0
            }
        # logger.info(command)
        p = subprocess.Popen(
            command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            end_time = time.time()
            duration = round((end_time - start_time)* 1000, 4)
            return False, duration, {
                "cause": "TimeoutExpired",
                "details": f"Invariant check timeout after {timeout} seconds",
                'call_to_z3': 4
            }
        end_time = time.time()
        duration = round((end_time - start_time)* 1000, 4)
        inv_parts = inv.replace("(",  " ( ").replace(")", " ) ").split()
        if len(inv_parts) >=2 and (inv_parts[0] == "(" and inv_parts[1] == "define-fun"):
            inv_name = inv_parts[2]
        else:
            inv_name = "inf-f"
        output = p.stdout.read().decode()
        errors = p.stderr.read().decode()
        # logger.info(f"Output: {output}, Errors: {errors}, Duration: {duration}")
        decision, feedback, problem_details = parse_verification_decision(
            inv_name, output, errors)
        p.terminate()
        return decision, duration, feedback

