class DaskToArray(object):
    def __init__(self, use_multithread=False):
        self._scheduler = "threads" if use_multithread else "synchronous"

    def __call__(self, pic):
        return pic.compute(scheduler=self._scheduler)
