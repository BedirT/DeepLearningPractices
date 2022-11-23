from tools import intersection_over_union, iou_simple


def test_iou():
    pred_box = torch.tensor([[1., 1., 2., 2.], [2., 2., 2., 2.]])
    target_box = torch.tensor([[2., 2., 2., 2.], [2., 2., 2., 2.]])

    res = intersection_over_union(pred_box, target_box)
    res_simple = iou_simple(pred_box, target_box)

    true_res = torch.tensor([1/7, 1.0])

    assert torch.all(res == true_res)
    assert torch.all(res_simple == true_res)

    print("All tests passed!")
