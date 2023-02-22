

class CrossEntropy:
    def calculate(pred, gt):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt)
        return loss