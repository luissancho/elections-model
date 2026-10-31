#!/usr/bin/env python

import sys

from mtpy import mtpy, pipelines
from mtpy.core.utils.strings import to_camel


def collect(lines, pipeline):
    """Append the `line` of a pipeline (string or list of strings) to `lines`."""
    line = pipeline.line
    if isinstance(line, str):
        lines.append(line)
    elif line:
        lines.extend(line)


def failed_in(result):
    """Tell whether a chunk result (a dict or a list of dicts) is a failure."""
    results = result if isinstance(result, list) else [result]
    return any(isinstance(r, dict) and r.get('status') == 'failed' for r in results)


def main(name, action, chunks, namespace=None):
    """
    Run a pipeline chunk by chunk, finish it and print its summary lines.

    Parameters
    ----------
    name : str
        Pipeline name, `group.pipeline` or `pipeline`, in snake_case.
    action : str or None
        Pipeline action to run.
    chunks : list of str
        Chunks to run one by one; empty runs the pipeline once.
    namespace : object, optional
        Where to look the pipeline class up; defaults to `mtpy.pipelines`.

    Returns
    -------
    int
        0 when every chunk and the closing went well; 1 when a chunk result failed or `finish` returned a failure.
    """
    mtpy.run()

    if '.' in name:
        module, name = name.split('.')
        name = '{}.{}'.format(module, to_camel(name))
    else:
        name = to_camel(name)

    pipeline = getattr(namespace if namespace is not None else pipelines, name)()
    lines = []
    failed = False

    if len(chunks) > 0:
        for index, chunk in enumerate(chunks, 1):
            pipeline.run(action, chunk=chunk, position=(index, len(chunks)))
            collect(lines, pipeline)
            failed = failed or failed_in(pipeline.result)
    else:
        pipeline.run(action, position=(1, 1))
        collect(lines, pipeline)
        failed = failed or failed_in(pipeline.result)

    pipeline.line = None
    failed = bool(pipeline.finish()) or failed
    collect(lines, pipeline)

    for line in lines:
        print(line)

    return 1 if failed else 0


if __name__ == '__main__':

    name = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else None
    chunks = sys.argv[3:] if len(sys.argv) > 3 else []

    sys.exit(main(name, action, chunks))
