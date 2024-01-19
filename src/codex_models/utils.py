# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import tiktoken
import sys
import z3


if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../../.."))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    from src.util import check_z3_parsable
    from src.util import get_logger


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding =  tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_dictionary_prompt(prompt, model_name, text_key='content'):
    lengths = [num_tokens_from_string(m[text_key], model_name) for m in prompt]
    return sum(lengths)


def filter_invariant_based_on_format(
        text, format='```',
        raise_error=False
):
    codes = []
    tag = f"{format}"
    if tag not in text:
        return []
    text = text.replace(tag, f" {tag} ")
    tokens = text.split(" ")
    tidx = 0
    current_code = []
    code_started = False
    while tidx < len(tokens):
        if not code_started and tokens[tidx].strip() == tag:
            current_code = []
            code_started = True
        elif code_started and tokens[tidx].strip() == tag:
            inv_code = " ".join(current_code)
            if raise_error:
                check_z3_parsable(inv_code)
                codes.append(inv_code)
            else:
                try:
                    check_z3_parsable(inv_code)
                    codes.append(inv_code)
                except:
                    pass
            code_started = False
        else:
            if code_started:
                current_code.append(tokens[tidx])
        tidx += 1
    return [c for c in codes if 'define-fun' in c]


def filter_invariant_based_on_tag(
    text, tag='code',
    raise_error=False,
    check_whether_z3_parsable=True
):
    codes = []
    otag, ctag = f"<{tag}>", f"</{tag}>"
    if otag not in text or ctag not in text:
        return []
    text = text.replace(otag, f" {otag} ").replace(ctag, f" {ctag} ")
    tokens = [t for t in text.split(" ") if t.strip() != ""]
    tidx = 0
    current_code = []
    code_started = False
    while tidx < len(tokens):
        if tokens[tidx].strip() == otag:
            current_code = []
            code_started = True
        elif tokens[tidx].strip() == ctag:
            inv_code = " ".join(current_code)
            if raise_error:
                if check_whether_z3_parsable:
                    check_z3_parsable(inv_code)
                codes.append(inv_code)
            else:
                try:
                    if check_whether_z3_parsable:
                        check_z3_parsable(inv_code)
                    codes.append(inv_code)
                except:
                    pass
            code_started = False
        else:
            if code_started:
                current_code.append(tokens[tidx])
        tidx += 1
    return [c for c in codes if 'define-fun' in c]

"""Update invariant cache and return new invariants that are not in the cache yet"""
def new_invs_and_update_cache(invariants, inv_cache, semantic_dedup=False, logger=None):
    # dedpulicate invariants
    invariants = list(set(invariants))
    taken_invariants = []
    # only retain invariants that are not in the cache
    for inv in invariants:
        i = re.sub("[ \n\t]+", " ", inv)
        if i not in inv_cache:
            if semantic_dedup:
                # check if the invariant is semantically equivalent to any of the invariants in the cache
                # if so, then do not add it to the cache
                is_semantically_equivalent = False
                for c in inv_cache:
                    if check_equivalence_inv_pair(c, inv, logger=logger):
                        is_semantically_equivalent = True
                        break
                if is_semantically_equivalent:
                    continue
            taken_invariants.append(inv)
            inv_cache.add(i)
    return taken_invariants

def truncate(text):
    return text[:int(.9 * len(text))]

from z3 import *


def remove_comments(s):
    # remove comments
    lines = s.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith(";"):
            continue
        elif ";" in line:
            line = line[:line.index(";")]
        new_lines.append(line)
    return "\n".join(new_lines)

# Define a function that takes two SMT2 functions as strings and checks their equivalence using Z3
def check_equivalence_inv_pair(_sf1, _sf2, logger=None):
    sf1 = remove_comments(_sf1)
    sf2 = remove_comments(_sf2)
    s = Solver()
    s.set("timeout", 30)
    import re
    # Define the pattern to match (assumes no Bool arguments for now!)
    # match define-fun <inv-name> (<args>) Bool (<body>)
    sf1 = re.sub("[ \n\t]+", " ", sf1).strip()
    sf2 = re.sub("[ \n\t]+", " ", sf2).strip()
    pattern = r"\(\s*define-fun\s*([-a-zA-Z0-9._]+)\s*\((.+?)\)\s*Bool\s*\((.+)\)\s*\)"
    f1 = re.search(pattern, sf1)
    f2 = re.search(pattern, sf2)
    if (f1 is None or f2 is None):
        return None

    # assert that there are no Bool arguments in both f1 and f2
    if (("Bool" in f1.groups()[1]) or ("Bool" in f2.groups()[1])):
        return None

    str = f"""(assert 
    (not
        (forall 
            ({f1.groups()[1]})  
            (= 
                ({f1.groups()[2]}) 
                ({f2.groups()[2]})
            )
        )
    ))"""
    try:
        eqstr = parse_smt2_string(str)
        s.add(eqstr)
        result = s.check()
        if result == unsat:
            return True
        elif result == sat:
            return False
        else:
            return None
    except z3types.Z3Exception as e:
        print("=" * 80)
        print(_sf1)
        print(_sf2)
        print("=" * 80)
        # exit()
        return None


if __name__ == '__main__':
    msg = """
    <code>(define-fun inv_fun ((a (Array Int Int)) (n Int) (i Int)) Bool (and (>= i 0) (<= i n) (forall ((j Int)) (=> (and (>= j 0) (<= j (* 2 n))) (>= (select a j) 0))) (forall ((j Int)) (=> (and (>= j 0) (< j i)) (not (= (select a j) 0))))))</code>
    """
    # logger.info(invs)
    # examples for checking equivalence of two invariants
    f1 = "(define-fun inv-f ((c Int) (n Int) (tmp Int) (c_0 Int) (c_1 Int) (c_2 Int) (c_3 Int) (c_4 Int) (c_5 Int) (n_0 Int)) Bool (or (and (= c c_2) (= n n_0) (<= tmp 0) (<= c_0 0) (<= c_1 0) (<= c_3 0) (<= c_4 0) (<= c_5 0)) (and (not (= c c_2)) (not (= c n_0)) (not (<= c n_0))) (and (= c c_2) (not (= c n_0)) (>= tmp 1) (>= c_1 1) (>= c_3 1) (>= c_4 1) (>= c_5 1)) ) )"
    f2 = "(define-fun inv-f ((c Int) (n Int) (tmp Int) (c_0 Int) (c_1 Int) (c_2 Int) (c_3 Int) (c_4 Int) (c_5 Int) (n_0 Int)) Bool (and (<= 0 c_1) (<= 0 n) (<= 0 tmp) (<= 0 c_0) (<= 0 c_2) (<= 0 c_3) (<= 0 c_4) (<= 0 c_5) (<= 0 n_0) (<= (+ c_2 1) c_3) (<= (+ c_2 1) c_4) (<= c_4 c_5) (<= c_4 n_0) (<= 1 c_5) (<= c_1 c_2) (<= c_2 n_0) (or (= c_2 n_0) (<= c_2 n_0)) (or (not (= c_2 n_0)) (= tmp (+ c_2 1))) (or (not (= c_2 n_0)) (= tmp c_2)) (or (= c_2 n_0) (= c_5 1)) (or (not (= c_2 n_0)) (= c_4 c_2))))"
    print(check_equivalence_inv_pair(f1, f2))
    # f3 = "(define-fun inv-f.1 ((x Int) (y.1 Int)) Bool (- (- 0 x) y.1))"
    # check_equivalence_inv_pair(f1, f3)
    # f4 = "(define-fun inv-f.1 ((x Bool) (y.1 Int)) Bool (- (- 0 x) y.1))"
    # check_equivalence_inv_pair(f1, f4)
