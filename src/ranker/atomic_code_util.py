# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import signal
import os
import sys

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.ranker import util
    logger = util.get_logger()


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logger.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)