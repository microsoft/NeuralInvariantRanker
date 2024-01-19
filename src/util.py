# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import os
import sys
from datetime import datetime
from typing import Dict
import z3

if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    LOG_DIRECTORY = os.path.join(PROJECT_PATH, "logs")
    os.makedirs(LOG_DIRECTORY, exist_ok=True)


def get_problem_type(problem_path):
    problems_lines = open(problem_path).readlines()
    for line in problems_lines:
        if 'set-logic' in line:
            return line.strip()
    return ''


class OverLengthError(Exception):
    def __init__(self, msg):
        self.msg = msg


class FeedBack:
    def __init__(
        self,
        feedback: Dict,
    ):
        self.error_cause = feedback['cause']
        self.error_details = feedback['details']
    
    def patch_feedback(self, details):
        if len(details) > 0:
            details = f"The error details are: \n{details}"
        return f"The invariant failed with: {self.error_cause}.\n\n" + \
            f"{details}\n" + \
            f"Please fix the error and generate a new necessary and sufficient invariant\n" + \
            f"Only generate the invariant, do not generate anything else." + \
            f"Also make sure that there is no parsing error in the invariant."
    
    def __str__(self) -> str:
        return self.get_feedback()
    
    def get_feedback(
            self, level: int = 0
        ):
        detail_lines = self.error_details.split("\n\n")
        if level == 0:
            details = self.error_details
        else:        
            details = "\n\n".join(detail_lines[:-level]) \
                if len(detail_lines) > level else ""
        return self.patch_feedback(details)
    

def get_logger(
    name=None, 
    disable_stdout_globally=False,
    include_time_in_output_file=True
):
    if Logger.logger is None:
        Logger.logger = Logger(name, include_time_in_output_file)
        Logger.logger.stdout = not disable_stdout_globally
    return Logger.logger


def check_z3_parsable(text, problem_type=''):
    solver = z3.Solver()
    solver.from_string(problem_type + "\n" + text)


class Logger:
    logger = None

    def __init__(self, name=None, include_time_in_output_file=True):
        frame = inspect.stack()[1]
        if os.path.abspath(frame.filename) != os.path.abspath(__file__):
            raise RuntimeError(
                "Creating Logger object is prohibited.\n" +\
                "Use src.util.get_logger() instead!"
            )
        now = datetime.now()
        name_parts = []
        if name is not None:
            name_parts.append(name)
        if include_time_in_output_file:
            name_parts.append(f"{now.strftime('%Y-%m-%d-%H:%M:%S')}")
        if len(name_parts) == 0:
            name_parts.append('log-file')
        log_file_name = f"{'-'.join(name_parts)}.log"
        self.log_file = os.path.join(LOG_DIRECTORY, log_file_name)
        if name is not None and not include_time_in_output_file:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
        self.stdout = True
        self.enable_log = True

    def msg(self, *items, msg_type='INFO', sep='', stdout=True):
        if not self.enable_log:
            return
        fp = open(self.log_file, 'a')
        frame = inspect.stack()[2]
        filename = frame.filename
        if filename.startswith(PROJECT_PATH):
            filename = filename[(len(PROJECT_PATH) + 1):]
        t = datetime.now().strftime('%H:%M:%S')
        lineno = frame.lineno
        if sep == '\n':
            for i in items:
                if stdout:
                    print(
                        f'{msg_type} {t} File "{filename}", line {lineno} : ', 
                        end='\t', flush=True
                    )
                    print(i, flush=True)
                print(
                    f'{msg_type} {t} File "{filename}", line {lineno} : ', 
                    end='\t', file=fp
                )
                print(i, file=fp)
        else:
            if stdout:
                print(
                    f'{msg_type} {t} File "{filename}", line {lineno} : ', 
                    end='\t', flush=True
                )
                print(*items, sep=sep, flush=True)
            print(
                f'{msg_type} {t} File "{filename}", line {lineno} : ', 
                end='\t', file=fp
            )
            print(*items, sep=sep, file=fp)
        fp.close()

    def info(self, *items, sep=' ', stdout=False):
        self.msg(*items, msg_type="[INFO] ", sep=sep, stdout=stdout or self.stdout)

    def debug(self, *items, sep=' ', stdout=False):
        self.msg(*items, msg_type="[DEBUG]", sep=sep, stdout=stdout or self.stdout)

    def warn(self, *items, sep=' ', stdout=False):
        self.msg(*items, msg_type="[WARN] ", sep=sep, stdout=stdout or self.stdout)

    def error(self, *items, sep=' ', stdout=False):
        self.msg(*items, msg_type="[ERROR]", sep=sep, stdout=stdout or self.stdout)


# if __name__ == '__main__':
#     logger1 = get_logger()
#     logger2 = get_logger()
#     logger1.info("Hello World")
#     logger2.info("This is good")
#     logger1.error("Hello World")
#     import time
#     time.sleep(15)
#     logger2.warn("This is good")
#     logger1.debug("Hello World")
#     logger2.error("This is good")
