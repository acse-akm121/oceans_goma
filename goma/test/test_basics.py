from goma.src.basics_import import GenericImport


def test_basics_import():
    x = 1
    obj = GenericImport(x)
    assert obj.test() == x
