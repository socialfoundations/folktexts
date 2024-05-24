"""Dummy test.
"""


def test_version_in_alpha():
    """Checks whether we're still in alpha."""
    from folktexts import __version__
    assert __version__.startswith("0.")
