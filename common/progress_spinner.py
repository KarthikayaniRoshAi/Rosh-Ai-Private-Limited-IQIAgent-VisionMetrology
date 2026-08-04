"""
------------------------------------------------------------
IQI Progress Spinner
------------------------------------------------------------

Displays an animated spinner while a long running task is
executing.

Example

spinner = ProgressSpinner(
    "Engineering Knowledge Extraction"
)

spinner.start()

...
Long running task
...

spinner.stop()
"""

import itertools
import sys
import threading
import time


class ProgressSpinner:

    def __init__(self, message: str):

        self.message = message
        self._running = False
        self._thread = None

    def start(self):

        self._running = True

        self._thread = threading.Thread(
            target=self._animate,
            daemon=True
        )

        self._thread.start()

    def stop(self):

        self._running = False

        if self._thread:
            self._thread.join()

        # Clear current line
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()

    def _animate(self):

        spinner = itertools.cycle(
            ["|", "/", "-", "\\"]
        )

        start_time = time.time()

        while self._running:

            elapsed = int(time.time() - start_time)

            mins = elapsed // 60
            secs = elapsed % 60

            sys.stdout.write(
                f"\r{self.message}... "
                f"{next(spinner)} "
                f"{mins:02d}:{secs:02d}"
            )

            sys.stdout.flush()

            time.sleep(0.2)