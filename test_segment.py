from segment import hex_to_rgb, rank_subjects


def test_rank_subjects():
    w, h = 1000, 1000
    center = [400, 400, 600, 600]
    corner = [0, 0, 200, 200]
    tiny = [490, 490, 500, 500]
    assert rank_subjects([corner, tiny, center], [0.9, 0.9, 0.9], w, h) == [2, 0, 1]
    assert rank_subjects([center, center], [0.5, 0.9], w, h)[0] == 1
    assert rank_subjects([], [], w, h) == []


def test_hex_to_rgb():
    assert hex_to_rgb("#009000") == (0, 144, 0)
    assert hex_to_rgb("ffFFff") == (255, 255, 255)


if __name__ == "__main__":
    test_rank_subjects()
    test_hex_to_rgb()
    print("ok")
