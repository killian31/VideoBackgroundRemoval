import numpy as np

from segment import box_distance, box_iou, closest, hex_to_rgb, mask_box, rank_subjects


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


def test_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    near, far, big = [100, 100, 120, 120], [500, 500, 520, 520], [95, 95, 300, 300]
    subjects = [{"box": b} for b in (far, big, near)]
    assert closest(subjects, [102, 98, 122, 118])["box"] == near
    assert box_distance(near, near) == 0
    assert box_distance(far, near) > 1 > box_distance([105, 100, 125, 120], near)
    alpha = np.zeros((50, 80), np.uint8)
    assert mask_box(alpha) is None
    alpha[10:20, 30:45] = 255
    assert mask_box(alpha) == [30, 10, 45, 20]


if __name__ == "__main__":
    test_boxes()
    test_rank_subjects()
    test_hex_to_rgb()
    print("ok")
