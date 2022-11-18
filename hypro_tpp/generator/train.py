import time
import torch
from torch import optim
from hypro_tpp.model_runner import ModelRunner


class Trainer(ModelRunner):

    def __init__(self, config):
        super(Trainer, self).__init__(config)

        # load data
        [self.train_loader, self.dev_loader, self.test_loader] \
            = self.get_dataloader()

        self.check_data_loaders()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.train_config['lr']
        )
        self.optimizer.zero_grad()  # init clear

    def check_data_loaders(self):
        for data_loader in [self.train_loader, self.dev_loader, self.test_loader]:
            # valid checks
            assert self.model_config['add_bos'] == data_loader.dataset.add_bos
            assert self.model_config['num_event_types_pad'] == data_loader.dataset.num_types  # include pad, bos and eos
            assert self.model_config['num_event_types_no_pad'] == data_loader.dataset.event_num
            assert self.model_config['event_pad_index'] == data_loader.dataset.pad_index

        return

    def run(self):
        print("start hypro_tpp...")

        for _epoch in range(self.train_config['max_epoch']):
            tic = time.time()
            log_lik, acc, (event_ll, non_event_ll), num_tokens, num_events, _, _, _, _ \
                = self.run_one_epoch(self.model, self.train_loader, "train", self.optimizer)
            time_train = (time.time() - tic)
            message = f"[ Epoch {_epoch} (train) ]: time to train one epoch is {time_train}, train log-like is {log_lik / num_tokens}, num_tokens: {num_tokens}, num_events: {num_events}\n" \
                      f", event_ll is {event_ll / num_tokens: .4f}, non_event_ll is {non_event_ll / num_tokens: .4f}, acc is {acc : .4f} "
            self.log.checkpoint(message)
            print(message)
            with torch.no_grad():
                tic = time.time()
                log_lik, acc, (event_ll, non_event_ll), num_tokens, num_events, _, _, _, _ \
                    = self.run_one_epoch(self.model, self.dev_loader, "eval")
                time_valid = (time.time() - tic)
                message = f"[ Epoch {_epoch} (valid) ]: time to validate is {time_valid}, valid log-like is {log_lik / num_tokens}, valid acc is {acc : .4f}, " \
                          f"valid_event_ll is {event_ll / num_tokens: .4f}, valid_non_event_ll is {non_event_ll / num_tokens: .4f}"
                self.log.checkpoint(message)
                print(message)
                updated = self.log.updateBest("loglik", log_lik / num_tokens, _epoch)
                message = "current best loglik is {:.4f} (updated at epoch-{})".format(
                    self.log.current_best['loglik'], self.log.episode_best)
                if updated:
                    message += f", best updated at this epoch"
                    torch.save(self.model.state_dict(), self.model_config['saved_model_dir'])
                self.log.checkpoint(message)
                print(message)
                tic = time.time()
                log_lik, acc, (event_ll, non_event_ll), num_tokens, num_events, _, _, _, _ \
                    = self.run_one_epoch(self.model, self.test_loader, "eval")
                time_valid = (time.time() - tic)
                message = f"[ Epoch {_epoch} (test) ]: time to validate is {time_valid}, valid log-like is {log_lik / num_tokens}, valid acc is {acc : .4f}, " \
                          f"valid_event_ll is {event_ll / num_tokens: .4f}, valid_non_event_ll is {non_event_ll / num_tokens: .4f}"
                self.log.checkpoint(message)
                print(message)