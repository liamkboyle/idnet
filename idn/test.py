from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
from .utils.validation import Validator
from .utils.trainer import Trainer
from .utils.helper_functions import move_batch_to_cuda
from .utils.plotting import plot_of_arrows
import torch
from tqdm import tqdm
import cv2
import numpy as np

def run_inference(trainer: Trainer):
    TestDSEC = Validator.get_test_type("dsec")
    test_cfg = compose(config_name="validation/dsec_test",
                       overrides=[]).validation
    class CustomTestDSEC(TestDSEC):
        def __init__(self, test_type):
            super().__init__(test_type)

        @torch.no_grad()
        def execute_single_test(self, model_eval, save_all=False):
            with self.evaluate_model(model_eval) as model:
                model_forward_fn = self.configure_model_forward_fn(model)
                forward_pass_fn = self.configure_forward_pass_fn(
                    model_forward_fn)
                with self.logger.log_test(model) as log_path:
                    if not isinstance(self.data_loader, list):
                        self.data_loader = [self.data_loader]

                    first_seq = self.data_loader[0]
                    print(f"Sequence name: {first_seq.dataset.seq_name}")

                    for idx, batch in enumerate(tqdm(first_seq, position=1)):
                        if isinstance(batch, list):
                            assert 'save_submission' in batch[-1]
                        else:
                            assert 'save_submission' in batch
                            # for non-recurrent loading, we skip samples
                            # not for eval
                            if not batch['save_submission'].cpu().item() and not save_all:
                                continue

                        batch = move_batch_to_cuda(
                            batch, self.device)
                        
                        out = forward_pass_fn(batch)
                        final_predicition = out["final_prediction"]
                        flow_history = out["flow_history"]

                        flow = final_predicition[0].cpu().numpy().transpose(1, 2, 0)
                        flow_hist = flow_history[0, 0].cpu().numpy().transpose(1, 2, 0)

                        black_img = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
                        flow_viz = plot_of_arrows(black_img, flow, fraction=0.03)
                        flow_hist_viz = plot_of_arrows(black_img, flow_hist, fraction=0.03)

                        cv2.imshow('flow_image', flow_viz)
                        cv2.imshow('flow_hist_image', flow_hist_viz)
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return
                    
                    first_batch = next(iter(first_seq))
                    first_batch = move_batch_to_cuda(first_batch, self.device)
                    # start_time = time.perf_counter()

                    out = forward_pass_fn(first_batch)#.to("cpu")
                    final_predicition = out["final_prediction"]
                    # flow_trajectory = out["flow_trajectory"]
                    # flow_next_trajectory = out["flow_next_trajectory"]

                    flow = final_predicition[0].cpu().numpy().transpose(1, 2, 0)

                    black_img = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
                    flow_viz = plot_of_arrows(black_img, flow, fraction=0.01)

                    cv2.imshow('flow_image', flow_viz)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                    
                    # cv2.imwrite("/root/ajnaboiz/idanet/flow_idnet.png", flow_viz)                  


                
    cust_test = CustomTestDSEC(test_cfg)
    cust_test.execute_single_test(trainer.model, save_all=True)


@hydra.main(config_path="config", config_name="id_eval")

def main(config):
    trainer = Trainer(config)
    run_inference(trainer)


if __name__ == '__main__':
    main()
