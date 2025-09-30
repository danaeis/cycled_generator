from monai.engines import SupervisedEvaluator
import torch
import gc

class ClearCacheEvaluator(SupervisedEvaluator):
    def _iteration(self, engine, batchdata):
        outputs = None
        try:
            with torch.no_grad():
                outputs = super()._iteration(engine, batchdata)
            return outputs
        finally:
            # Clean up batch data
            del batchdata
            
            # Clean up outputs if they exist
            if outputs is not None:
                del outputs
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()