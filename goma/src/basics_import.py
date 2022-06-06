from firedrake import *


class GenericImport(object):
    def __init__(self, x) -> None:
        super().__init__()
        self.x = x

    def test(self):
        return self.x
