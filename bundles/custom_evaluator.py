from monai.engines import SupervisedEvaluator
import torch

class ClearCacheEvaluator(SupervisedEvaluator):
    def _iteration(self, engine, batchdata):
        outputs = super()._iteration(engine, batchdata)
        torch.cuda.empty_cache()
        del outputs
        del batchdata
        

