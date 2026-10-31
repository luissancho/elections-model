from ..core.worker import SequenceJob


class Pipeline(SequenceJob):
    """
    Launch a single pipeline as a sequence of one.

    Run it with ``python job.py pipeline '{"select": "group.pipeline", "chunks": "a,b"}'``;
    ``action`` and the rest of the parameters are passed on to the pipeline.
    """

    def run(self, select=None, action=None, chunks=None, **kwargs):
        if not select or '.' not in select:
            raise ValueError('pipeline: "select" must be "group.pipeline"')

        group, name = select.split('.', 1)
        self.sequence = {group: [name]}

        return super().run(action=action, chunks=chunks, **kwargs)
