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
        0 when it ends.
    """
    mtpy.run()

    if '.' in name:
        module, name = name.split('.')
        name = '{}.{}'.format(module, to_camel(name))
    else:
        name = to_camel(name)

    pipeline = getattr(namespace if namespace is not None else pipelines, name)()
    lines = []

    if len(chunks) > 0:
        for index, chunk in enumerate(chunks, 1):
            pipeline.run(action, chunk=chunk, position=(index, len(chunks)))
            collect(lines, pipeline)
    else:
        pipeline.run(action, position=(1, 1))
        collect(lines, pipeline)

    pipeline.line = None
    pipeline.finish()
    collect(lines, pipeline)

    for line in lines:
        print(line)

    return 0


if __name__ == '__main__':

    name = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else None
    chunks = sys.argv[3:] if len(sys.argv) > 3 else []

    sys.exit(main(name, action, chunks))
