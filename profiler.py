import cProfile
import io
import pstats
import random

import torchvision  # pylint: disable=unused-import  # noqa
from tqdm import trange  # type: ignore[import]  # pylint: disable=unused-import  # noqa

from torchview import draw_graph  # pylint: disable=unused-import  # noqa


def profile() -> None:
    """
    Prints top N methods, sorted by time.
    Equivalent to:
        python -m cProfile -o data/profile.txt main.py -n 100
    Options:
        time, cumulative, line, name, nfl, calls
    -----------
    ncalls - for the number of calls.
    time/tottime - for the total time spent in the given function
    (and excluding time made in calls to sub-functions)
    cumulative/cumtime - is the cumulative time spent in this and all subfunctions
    (from invocation till exit). This figure is accurate even for recursive functions.
    """
    random.seed(0)
    command = (
        "for _ in trange(10): "
        "draw_graph(torchvision.models.resnet152(), input_size=(1, 3, 224, 224))"
    )
    profile_file = "profile.txt"
    sort = "time"

    cProfile.run(command, filename=profile_file, sort=sort)
    s = io.StringIO()
    stats = pstats.Stats(profile_file, stream=s)
    stats.sort_stats(sort).print_stats(100)

    with open('profile-readable.txt', 'w+') as f:
        f.write(s.getvalue())


if __name__ == "__main__":
    profile()
