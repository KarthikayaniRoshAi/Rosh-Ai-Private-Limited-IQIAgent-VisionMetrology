import time


class Progress:

    def __init__(self):

        self.stage = ""
        self.start_time = None

    def banner(self, title: str):

        print("\n" + "=" * 70)
        print(f"{title:^70}")
        print("=" * 70)

    def drawing(self,
                index: int,
                total: int,
                drawing_name: str):

        print(f"\nDrawing {index}/{total}")
        print("-" * 70)
        print(f"File : {drawing_name}")
        print("-" * 70)

    def start(self, stage: str):

        self.stage = stage
        self.start_time = time.perf_counter()

        print(f"{stage}...", end="", flush=True)

    def complete(self, message: str = ""):

        elapsed = time.perf_counter() - self.start_time

        print(
            f"\r{self.stage:<45}"
            f"{elapsed:6.2f} sec"
            f"  {message}"
        )

    def failed(self, message: str):

        elapsed = time.perf_counter() - self.start_time

        print(
            f"\r{self.stage:<45}"
            f"{elapsed:6.2f} sec"
        )

        print(f"    {message}")

    def info(self, message: str):

        print(f"    {message}")

    def separator(self):

        print("-" * 70)

    def summary(self,
                total_time: float):

        print("=" * 70)
        print(
            f"Completed Successfully "
            f"({total_time:.2f} sec)"
        )
        print("=" * 70)