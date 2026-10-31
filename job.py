#!/usr/bin/env python

import json
import signal
import sys
import traceback

from mtpy import jobs, mtpy
from mtpy.core.utils.strings import to_camel
from mtpy.core.worker import Job, JobInterrupted


class Launcher(Job):
    """Minimal job used to log and alert when the requested job cannot even be built."""


def interrupt(signum, frame):
    """Signal handler that turns SIGTERM into a `JobInterrupted` so the job can alert and exit."""
    raise JobInterrupted('SIGTERM')


def main(name, params, namespace=None):
    """
    Run a job through `dispatch` and return the process exit code.

    Parameters
    ----------
    name : str
        Job name in snake_case.
    params : str or None
        JSON object with the keyword arguments of the job.
    namespace : object, optional
        Where to look the job class up; defaults to `mtpy.jobs`.

    Returns
    -------
    int
        0 on success, 1 when the job was interrupted or failed.
    """
    mtpy.run()
    signal.signal(signal.SIGTERM, interrupt)

    job = None

    try:
        job_class = getattr(namespace if namespace is not None else jobs, to_camel(name))
        kwargs = json.loads(params) if params else {}
        job = job_class()
        job.dispatch(**kwargs)
    except JobInterrupted as e:
        print('{}: interrupted ({})'.format(name, e), file=sys.stderr)
        return 1
    except Exception as e:
        if job is None:
            # `dispatch` never ran, so nobody alerted: do it here (the alert may be the only signal of a detached task)
            launcher = Launcher()
            launcher.log('error', '{} job failed'.format(name))
            launcher.alert('{} job failed: {}: {}'.format(name, type(e).__name__, e))
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':

    name = sys.argv[1]
    params = sys.argv[2] if len(sys.argv) > 2 else None

    sys.exit(main(name, params))
