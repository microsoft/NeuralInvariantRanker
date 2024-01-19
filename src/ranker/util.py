from datetime import datetime
import inspect
import os


def get_logger(name=''):
    return Logger()


class Logger:
    def __init__(self) -> None:
        file_path = os.path.abspath(__file__)
        self.project_dir = os.path.abspath(os.path.join(file_path, "../../.."))
        pass

    def msg(self, *items, initial="[INFO] ", sep=' '):
        frame = inspect.stack()[2]
        filename = os.path.realpath(frame.filename)
        filename = filename[(len(self.project_dir)+1):]
        lineno = frame.lineno
        t = datetime.now().strftime('%H:%M:%S')
        if sep == '\n':
            for i in items:
                print(f'{initial} {t} File "{filename}", line {lineno} : ', end='\t', flush=True)
                print(i, flush=True)
        else:
            print(f'{initial} {t} File "{filename}", line {lineno} : ', end='\t', flush=True)
            print(*items, sep=sep, flush=True)

    def info(self, *items, sep=' '):
        self.msg(*items, initial="[INFO] ", sep=sep)

    def debug(self, *items, sep=' '):
        self.msg(*items, initial="[DEBUG]", sep=sep)

    def warn(self, *items, sep=' '):
        self.msg(*items, initial="[WARN] ", sep=sep)
    
    def warning(self, *items, sep=' '):
        self.msg(*items, initial="[WARN] ", sep=sep)

    def error(self, *items, sep=' '):
        self.msg(*items, initial="[ERROR]", sep=sep)
