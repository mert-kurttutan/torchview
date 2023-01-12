from torchview.utils import OrderedSet


def test_discard_remove() -> None:

    oset_1: OrderedSet[int] = OrderedSet(range(10))

    oset_1.discard(7)
    assert 7 not in oset_1

    oset_1.discard(3)
    assert 3 not in oset_1
