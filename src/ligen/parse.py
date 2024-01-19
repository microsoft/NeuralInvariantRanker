import shutil
import subprocess
import os
import sys
import json
from typing import Optional
from tqdm import tqdm
import re


if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../../.."))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    from src import util
    # logger = util.get_logger()


def format_result(result):
    functions = result["functions"]
    assert isinstance(functions, list)
    for f in functions:
        if f['name'] == "END_OF_FUNC":
            functions.remove(f)
    result['functions'] = functions
    variables = result["variables"]
    assert isinstance(variables, list)
    for v in variables:
        if v['name'] == "END_OF_VAR":
            variables.remove(v)
    result['variables'] = variables
    synth_variables = result["synth_variables"]
    assert isinstance(synth_variables, list)
    for v in synth_variables:
        if v['name'] == "END_OF_VAR":
            synth_variables.remove(v)
    result['synth_variables'] = synth_variables
    branches = result["branches"]
    if "END_OF_BRANCH" in branches:
        branches.remove("END_OF_BRANCH")
    result["branches"] = branches
    return generate_function_body(result)

def generate_function_body(result):
    variables = result["variables"]
    variable_map = {}
    for v in variables:
        variable_map[v['name']] = v['type']

    functions = result["functions"]
    for f in functions:
        name = f['name']
        args = f['args']
        for v in args.keys():
            if v not in variable_map:
                variable_map[v] = args[v]
        args_string = " ".join([f"({v} {args[v]})" for v in args.keys()])
        return_type = f['return_type']
        body = f['body']
        full_function = f"(define-fun {name} ({args_string}) {return_type} \n\t{body})"
        f['full_function'] = full_function
    
    variables = []
    for v in variable_map:
        variables.append({
            'name': v,
            'type': variable_map[v]
        })
    result["variables"] = variables
    return result

def parse(
    problem_path: Optional[str] = None,
    problem_definition: Optional[str] = None,
    script_path=os.path.join(
        PROJECT_PATH, "LoopInvGen/_build/install/default/bin/lig-parse"
    ),
):
    assert problem_path is not None or problem_definition is not None, \
        'Either problem_path or problem_definition must be provided'
    if problem_path is None:
        problem_dir = os.path.join(PROJECT_PATH, "tmp/")
        if os.path.exists(problem_dir):
            shutil.rmtree(problem_dir)
        os.makedirs(problem_dir, exist_ok=True)
        problem_path = os.path.join(problem_dir, "tmp.sl")
        with open(problem_path, "w") as f:
            f.write(problem_definition)
            f.close()
    assert os.path.exists(script_path), \
        f'verification script was not found in {script_path}'
    command = f'{script_path} -s {problem_path}'

    # logger.info(command)
    p = subprocess.Popen(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.terminate()
        raise RuntimeError(f"TimeoutExpired {problem_path}\n{command}")
    output = p.stdout.read().decode()
    errors = p.stderr.read().decode()
    if errors is not None and errors.strip() != "":
        raise RuntimeError(f"Errors \n{errors} \nin \n{problem_path}\n{command}")
    p.terminate()
    result = json.loads(output)
    result = format_result(result)
    return result

def change_problem_exclude_non_ssa(
    problem_path: Optional[str] = None,
    problem_definition: Optional[str] = None,
    script_path=os.path.join(
        PROJECT_PATH, "LoopInvGen/_build/install/default/bin/lig-parse"
    )
) -> str:
    assert problem_path is not None or problem_definition is not None, \
        'Either problem_path or problem_definition must be provided'
    if problem_definition is None:
        f = open(problem_path, "r")
        problem_definition = f.read()
        f.close()
    parsed_problem = parse(
        problem_definition=problem_definition,
        script_path=script_path
    )
    ssa, _ = identify_ssa_and_nonssa_vars_in_invariant(
        problem_definition=problem_definition,
        script_path=script_path,
        parsed_problem=parsed_problem
    )

    invariant = parsed_problem['inv']
    inv_args = invariant['args']
    taken_args = []
    for arg_name in inv_args:
        if arg_name not in ssa:
            taken_args.append(f"({arg_name} {inv_args[arg_name]})")
    problem_lines = problem_definition.split("\n")
    for i in range(len(problem_lines)):
        if 'synth-inv' in problem_lines[i]:
            synth_inv_line = f'(synth-inv {invariant["name"]} ({" ".join(taken_args)})'
            problem_lines[i] = synth_inv_line
            break
    return "\n".join(problem_lines)


def identify_ssa_and_nonssa_vars_in_invariant(
    problem_path: Optional[str] = None,
    problem_definition: Optional[str] = None,
    script_path=os.path.join(
        PROJECT_PATH, "LoopInvGen/_build/install/default/bin/lig-parse"
    ),
    parsed_problem: Optional[dict] = None
):
    if parsed_problem is None:
        parsed_problem = parse(
            problem_path=problem_path,
            problem_definition=problem_definition,
            script_path=script_path
        )
    synth_variables = parsed_problem['synth_variables']
    ssa_variables, all_variables = [], []
    for v in synth_variables:
        vname = v['name']
        all_variables.append(vname)
        if "_" in vname:
            parts = vname.split("_")
            num = str(parts[-1])
            if num.isnumeric():
                ssa_variables.append(
                    (vname, "_".join(parts[:-1]))
                )
    ssa_variable_list = []
    for vname, v_non_ssa in ssa_variables:
        if v_non_ssa in all_variables:
            ssa_variable_list.append(vname)
    non_ssa_variable = list(set(all_variables) - set(ssa_variable_list))
    return list(ssa_variable_list), non_ssa_variable


if __name__ == '__main__':
    for problem_set_name in ["svcomp_in_scope"]:
        print("#######################")
        print("Now processing : ", problem_set_name)
        problem_dir = os.path.join(
            PROJECT_PATH, "data/problems", problem_set_name  
        )
        problems = os.listdir(problem_dir)
        output_dir = os.path.join(
            PROJECT_PATH, "data/problems", problem_set_name + "-parsed"  
        )
        os.makedirs(output_dir, exist_ok=True)
        for p in tqdm(problems):
            if not p.endswith(".sl"):
                continue
            ppath = os.path.join(problem_dir, p)
            try:
                parsed_result = parse(ppath)
                with open(os.path.join(output_dir, p + ".json"), "w") as fp:
                    json.dump(parsed_result, fp, indent=4)
            except RuntimeError as e:
                print("=========================")
                print(e)
            
    #     print("#######################\n\n\n")


