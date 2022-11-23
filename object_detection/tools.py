import torch

def intersection_over_union(
    pred_box: torch.Tensor, target_box: torch.Tensor, box_format: str = "midpoint"
) -> torch.Tensor:
    """
    Calculates intersection over union (IoU) for two sets of boxes
    
    Args:
        pred_box: (tensor) Bounds for the predicted boxes, sized [N,4]
        target_box: (tensor) Bounds for the target boxes, sized [N,4]
        box_format: (str) midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2)
    """
    if box_format == "midpoint":
        # Convert midpoint to corners
        pred_box = torch.cat(
            (pred_box[..., :2] - pred_box[..., 2:] / 2, 
             pred_box[..., :2] + pred_box[..., 2:] / 2), dim=1)
        target_box = torch.cat(
            (target_box[..., :2] - target_box[..., 2:] / 2, 
             target_box[..., :2] + target_box[..., 2:] / 2), dim=1)

    # Get the coordinates of bounding boxes
    x1 = torch.max(pred_box[..., 0], target_box[..., 0])
    y1 = torch.max(pred_box[..., 1], target_box[..., 1])
    x2 = torch.min(pred_box[..., 2], target_box[..., 2])
    y2 = torch.min(pred_box[..., 3], target_box[..., 3])

    # Intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union Area
    box1_area = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])
    box2_area = (target_box[..., 2] - target_box[..., 0]) * (target_box[..., 3] - target_box[..., 1])
    union = box1_area + box2_area - intersection

    return intersection / union  # iou

def iou_simple(
    pred_box: torch.Tensor, target_box: torch.Tensor, box_format: str = "midpoint"):
    """
    Same as above, but not using vectorization
    """
    if box_format == "midpoint":
        # Convert midpoint to corners
        for i in range(len(pred_box)):
            w, h = pred_box[i, 2], pred_box[i, 3]

            pred_box[i, 0] = pred_box[i, 0] - w / 2
            pred_box[i, 1] = pred_box[i, 1] - h / 2
            pred_box[i, 2] = pred_box[i, 0] + w
            pred_box[i, 3] = pred_box[i, 1] + h

        for i in range(len(target_box)):
            w, h = target_box[i, 2], target_box[i, 3]

            target_box[i, 0] = target_box[i, 0] - w / 2
            target_box[i, 1] = target_box[i, 1] - h / 2
            target_box[i, 2] = target_box[i, 0] + w
            target_box[i, 3] = target_box[i, 1] + h

    ious = []
    for i in range(len(pred_box)):
        x1 = max(pred_box[i, 0], target_box[i, 0])
        y1 = max(pred_box[i, 1], target_box[i, 1])
        x2 = min(pred_box[i, 2], target_box[i, 2])
        y2 = min(pred_box[i, 3], target_box[i, 3])

        x_diff = x2 - x1 if x2 - x1 > 0 else 0
        y_diff = y2 - y1 if y2 - y1 > 0 else 0
        intersection = x_diff * y_diff

        box1_x_diff = (pred_box[i, 2] - pred_box[i, 0])
        box1_y_diff = (pred_box[i, 3] - pred_box[i, 1])
        box1_area = box1_x_diff * box1_y_diff

        box2_x_diff = (target_box[i, 2] - target_box[i, 0])
        box2_y_diff = (target_box[i, 3] - target_box[i, 1])
        box2_area = box2_x_diff * box2_y_diff
        
        union = box1_area + box2_area - intersection

        ious.append(intersection / union)

    return torch.tensor(ious)

def non_max_suppression(
    bboxes: list, iou_threshold: float, threshold: float, box_format: str = "corners"
) -> List:
    """
    Does Non Max Suppression given bboxes
    
    Args:
        bboxes: (torch.tensor) All bboxes with their class probabilities,
            shape: [N, 6] (class_pred, prob, x1, y1, x2, y2) or
                   [N, 6] (class_pred, prob, x, y, w, h)
        iou_threshold: (float) threshold where predicted bboxes is correct
        threshold: (float) threshold to remove predicted bboxes
        box_format: (str) "midpoint" or "corners" used to specify bboxes
    """
    assert type(bboxes) == list

    # Converting midpoint to corners
    if box_format == "midpoint":
        for box in bboxes:
            w, h = box[4], box[5]
            box[2] = box[2] - w / 2
            box[3] = box[3] - h / 2
            box[4] = box[2] + w
            box[5] = box[3] + h

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        coords = chosen_box[2:]
        bboxes = [
            box for box in bboxes if box[0] != chosen_box[0] or
            intersection_over_union(
                torch.tensor(coords),
                torch.tensor(box[2:]),
                box_format=box_format) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
    pred_boxes: list, target_boxes: list, iou_threshold: float = 0.5, 
    box_format: str = "midpoint", num_classes: int = 20
) -> float:
    """
    Calculates mean average precision
    
    Args:
        pred_boxes: (list) list of lists containing all bboxes with each bboxes 
            specified as [img_idx, class_pred, prob_score, x1, y1, x2, y2] or
                         [img_idx, class_pred, prob_score, x, y, w, h]
        target_boxes: (list) similar to pred_boxes except all the correct ones
        iou_threshold: (float) threshold where predicted bboxes is correct
        box_format: (str) "midpoint" or "corners" used to specify bboxes
        num_classes: (int) number of classes
    """
    # Starting by defining a list for all AP for each class
    average_precisions = []

    for class_ in range(num_classes):
        # Getting all the detections with the particular class
        c_predicted = [box for box in pred_boxes if box[1] == class_]
        # Getting all the ground truth boxes with the particular class
        c_target = [box for box in target_boxes if box[1] == class_]

        # Counting the number of bboxes
        n_predicted = len(c_predicted)
        n_target = len(c_target)

        # If there are no predictions or no targets then AP is 0
        if n_predicted == 0 or n_target == 0:
            average_precisions.append(0)
            continue

        # Sorting the predictions by the probability score
        c_predicted = sorted(c_predicted, key=lambda x: x[2], reverse=True)
        
        # Defining a list to keep track of which target bboxes have
        # already been matched to a prediction
        target_boxes_already_matched = []

        # Defining a list to keep track of the precision at each detection
        # (i.e. for each prediction)
        precisions = []

        # Iterating through all the predicted bboxes
        for prediction in c_predicted:
            # Getting the image index
            img_idx = prediction[0]

            # Getting the target boxes which correspond to the same image
            # as the prediction
            target_boxes_with_same_img_idx = [
                box for box in c_target if box[0] == img_idx
            ]

            # If there are no target boxes in the image then the prediction
            # is automatically a false positive
            if len(target_boxes_with_same_img_idx) == 0:
                precisions.append(0)
                continue

            # Iterating through all the target bboxes in the image
            for target in target_boxes_with_same_img_idx:
                # If the target bbox has already been matched to a prediction 
                # then we skip
                if target in target_boxes_already_matched:
                    continue

                # If the IOU between the target and the prediction is above
                # the threshold then the prediction is a true positive
                if intersection_over_union(
                    torch.tensor(prediction[3:]),
                    torch.tensor(target[3:]),
                    box_format=box_format
                ) > iou_threshold:
                    target_boxes_already_matched.append(target)
                    precisions.append(1)
                else:
                    precisions.append(0)

        # If all the predictions are false positives then precision is 0
        if sum(precisions) == 0:
            average_precisions.append(0)
            continue

        # Calculating the precision and recall at each detection
        precisions = [sum(precisions[:i+1]) / (i+1) for i in range(n_predicted)]
        recalls = [sum(precisions[:i+1]) / (n_target + epsilon) 
                   for i in range(n_predicted)]

        # Adding an extra precision and recall value of 0 and 1 respectively
        # to make the graph go from 0 to 1
        precisions.insert(0, 0)
        recalls.insert(0, 0)

        # Calculating the average precision using the precision-recall curve 
        # using the trapezoidal rule in pytorch
        average_precisions.append(
            torch.trapz(torch.tensor(precisions), torch.tensor(recalls)))

    return sum(average_precisions) / len(average_precisions)


def map_driver(pred_boxes: list, target_boxes: list, starting_iou=0.5, 
               increment=0.05, ending_iou=0.9, num_classes=20, 
               box_format="midpoint") -> float:
    """
    Calculates the mean average precision for a range of IOU thresholds
    
    Args:
        pred_boxes: (list) list of lists containing all bboxes with each bboxes
            specified as [img_idx, class_pred, prob_score, x1, y1, x2, y2] or
                         [img_idx, class_pred, prob_score, x, y, w, h]
        target_boxes: (list) same as the bbox list but contains the correct
            bboxes
        starting_iou: (float) starting IOU threshold
        increment: (float) increment to increase the IOU threshold by
        ending_iou: (float) ending IOU threshold
        num_classes: (int) number of classes
        box_format: (str) "midpoint" or "corners" used to specify bboxes
    """
    mean_average_precisions = []

    for iou_threshold in np.arange(starting_iou, ending_iou, increment):
        mean_average_precisions.append(
            mean_average_precision(
                pred_boxes, target_boxes, iou_threshold, box_format, num_classes
            )
        )

    return mean_average_precisions
