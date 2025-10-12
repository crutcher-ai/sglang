import pytest

from tools.infer_client import _think_split_qwen


@pytest.mark.parametrize(
    "text,exp_r,exp_c",
    [
        ("</think> Answer.", "</think>", "Answer."),
        ("<think>abc</think> Final.", "<think>abc</think>", "Final."),
        ("No think here.", None, "No think here."),
        ("<think>multi</think>\n\nFinal line.", "<think>multi</think>", "Final line."),
        ("<think>first</think> Middle <think>second</think> End.", "<think>first</think> Middle <think>second</think>", "End."),
    ],
)
def test_qwen_think_split(text, exp_r, exp_c):
    r, c = _think_split_qwen(text)
    if exp_r is None:
        assert r is None
    else:
        assert r.strip() == exp_r
    assert c.strip() == exp_c

