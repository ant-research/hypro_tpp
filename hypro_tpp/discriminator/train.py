import time
from tqdm import tqdm
import torch
from torch import optim
from hypro_tpp.model_runner import ModelRunner
from hypro_tpp.preprocess import TPPNoiseDataset


class Trainer(ModelRunner):

    def __init__(self, config):
        super(Trainer, self).__init__(config)
        # load data
        [self.train_loader, self.dev_loader, self.test_loader] \
            = self.get_dataloader(dataset_fn=TPPNoiseDataset)

        if self.model_config.get('pretrained_model_dir'):
            print('Discriminator: load pretrained model from', self.model_config.get('pretrained_model_dir'))
            self.model.load_state_dict(torch.load(self.model_config['pretrained_model_dir']), strict=False)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.train_config['lr']
        )
        self.optimizer.zero_grad()  # init clear

        self.num_samples_per_seq = self.data_config.get('num_samples_per_seq', None)
        self.sample_len = self.model_config['sample_len']
        self.batch_size = self.config['data']['batch_size']

    def run(self):
        print("start hypro_tpp...")

        for _epoch in range(self.train_config['max_epoch']):
            tic = time.time()
            loss, real_acc, fake_acc, num_seqs = self.run_one_epoch(self.model,
                                                                    self.train_loader,
                                                                    "train",
                                                                    self.optimizer)
            time_train = (time.time() - tic)
            message = f"[ Epoch {_epoch} (train) ]: num_seqs: {num_seqs}\n " \
                      f"train loss is {loss: .6f}, real acc is {real_acc: .6f}, fake acc is {fake_acc: .6f} "
            self.log.checkpoint(message)
            print(message)
            with torch.no_grad():
                tic = time.time()
                val_loss, val_real_acc, val_fake_acc, val_seqs = self.run_one_epoch(self.model,
                                                                                    self.dev_loader,
                                                                                    "eval")

                time_valid = (time.time() - tic)
                message = f"[ Epoch {_epoch} (valid) ]: num_seqs: {val_seqs}, " \
                          f"valid loss is {val_loss : .6f}, real acc is {val_real_acc: .5f}, fake acc is {val_fake_acc: .5f}"
                self.log.checkpoint(message)
                print(message)
                updated = self.log.updateBest("loss", val_loss, _epoch)
                message = "current best loss is {:.6f} (updated at epoch-{})".format(
                    self.log.current_best['loss'], self.log.episode_best)
                if updated:
                    message += f", best updated at this epoch"
                    torch.save(self.model.state_dict(), self.model_config['saved_model_dir'])
                self.log.checkpoint(message)
                print(message)
                tic = time.time()
                test_loss, test_real_acc, test_fake_acc, test_seqs = self.run_one_epoch(self.model,
                                                                                        self.test_loader,
                                                                                        "eval")
                time_test = (time.time() - tic)
                message = f"[ Epoch {_epoch} (test) ]: num_seqs: {test_seqs}, " \
                          f"test loss is {test_loss : .6f}, real acc is {test_real_acc: .5f}, fake acc is {test_fake_acc: .5f}"
                self.log.checkpoint(message)
                print(message)

    def run_one_epoch(self, model, data_loader, mode, optimizer=None):
        loss_name = self.model_config.get('loss').lower()
        loss_func = eval(f'self.compute_{loss_name}_loss')

        msg = f'Loss function: {loss_name}'
        self.log.checkpoint(msg)
        print(msg)

        assert mode in {"train", "eval"}
        if mode == "eval":
            model = model.eval()
        else:
            assert optimizer is not None
        total_seqs = 0
        total_loss = 0.0
        total_real_correct = 0
        total_fake_correct = 0
        total_real_seq = 0
        total_fake_seq = 0
        for batch in tqdm(data_loader, mininterval=2, desc=f'   - ({mode}) -    ', leave=False):
            new_batch = [x.to(self.device) for x in batch]
            num_seqs = batch[0].size()[0] * 2  # real+fake(averaged to one seq)

            _loss, real_correct, fake_correct, real_num_seqs, fake_num_seqs = loss_func(
                model,
                new_batch
            )
            total_loss += _loss.item()
            total_seqs += num_seqs
            total_real_correct += real_correct
            total_fake_correct += fake_correct
            total_real_seq += real_num_seqs
            total_fake_seq += fake_num_seqs
            if mode == "train":
                _loss.div(num_seqs).backward()
                optimizer.step()
                optimizer.zero_grad()

        return total_loss / total_seqs, total_real_correct / total_real_seq, total_fake_correct / total_fake_seq, total_seqs

    def compute_bce_loss(self, model, batch):
        real_logit = self._get_real_logit(model, batch)
        fake_logits, fake_distances = self._get_fake_logits_and_distances(model, batch)

        # Real loss
        real_loss = torch.log(torch.sigmoid(-real_logit))

        # Fake loss
        fake_loss = torch.log(torch.sigmoid(fake_logits)).sum(dim=-1)

        loss = - (real_loss + fake_loss).mean()
        # metric calculation
        (num_real_correct, num_correct_seqs,
         num_fake_correct, num_fake_seqs) = self._top1_metric(real_logit, fake_logits, fake_distances)

        return loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs

    def compute_hinge_loss(self, model, batch, is_detach=False):
        real_logit = self._get_real_logit(model, batch)
        fake_logits, fake_distances = self._get_fake_logits_and_distances(model, batch)

        # shape -> [batch_size, 1]
        real_energy = real_logit
        # shape -> [batch_size, num_noise_samples]
        fake_energies = fake_logits[..., 0]

        if is_detach:
            # stop gradient to real sample
            real_energy = real_energy.detach()

        # normalize
        real_energy = torch.sigmoid(real_energy)
        fake_energies = torch.sigmoid(fake_energies)

        beta = 1.0
        # shape -> [batch_size, num_noise_samples]
        energy_gap = real_energy - fake_energies + fake_distances * beta

        energy_gap = torch.relu(energy_gap)

        loss = energy_gap.sum(dim=-1)
        loss = loss.mean()

        # metric calculation
        (num_real_correct, num_correct_seqs,
         num_fake_correct, num_fake_seqs) = self._top1_metric(real_logit, fake_logits, fake_distances)

        return loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs

    def compute_mnce_loss(self, model, batch):
        real_logit = self._get_real_logit(model, batch)
        fake_logits, fake_distances = self._get_fake_logits_and_distances(model, batch)

        real_energy = real_logit[..., 0]
        fake_energy = fake_logits[..., 0]

        # normalize
        real_energy = torch.sigmoid(real_energy)
        fake_energy = torch.sigmoid(fake_energy)

        whole_energy = torch.cat([real_energy[..., None], fake_energy], dim=-1)

        # Real loss
        real_loss = real_energy

        fake_loss = torch.log(torch.exp(-whole_energy).sum(dim=-1))

        loss = real_loss + fake_loss
        loss = loss.mean()
        # metric calculation
        (num_real_correct, num_correct_seqs,
         num_fake_correct, num_fake_seqs) = self._top1_metric(real_logit, fake_logits, fake_distances)

        return loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs

    def compute_bce_hinge_loss(self, model, batch):
        main_loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs = self.compute_bce_loss(model,
                                                                                                               batch)
        hinge_loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs = self.compute_hinge_loss(model,
                                                                                                                  batch)

        loss = main_loss + hinge_loss
        return loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs

    def compute_mnce_hinge_loss(self, model, batch):
        main_loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs = self.compute_mnce_loss(
            model,
            batch
        )
        hinge_loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs = self.compute_hinge_loss(
            model,
            batch,
            is_detach=True
        )

        loss = main_loss + hinge_loss * 0.1
        return loss, num_real_correct, num_fake_correct, num_correct_seqs, num_fake_seqs

    def _get_real_logit(self, model, batch):
        """

        Args:
            model:
            batch:

        Returns:
            real_logit: tensor with shape [batch_size, 2]
        """
        (time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask,
         noise_time_seq, noise_delta_seq, noise_event_seq, noise_distance) = batch

        real_logit = model.forward(event_seq, time_seq, batch_non_pad_mask, attention_mask)
        # retrieve last predicted lo我gits and output the score
        # shape -> [batch_size, hidden_size]
        real_logit = model.get_logits_at_last_step(real_logit, batch_non_pad_mask)

        # [batch_size, 1]
        real_logit = model.predict_as_discriminator(real_logit)

        return real_logit

    def _get_fake_logits_and_distances(self, model, batch):
        """

        Args:
            model:
            batch:

        Returns:
            fake_logits: tensor with shape [batch_size, num_samples, 2]
            fake_distances: tensor with shape [batch_size, num_samples]

        """
        # [batch_size, max_len]
        # [batch_size, num_samples, max_len]
        (time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask,
         noise_time_seq, noise_delta_seq, noise_event_seq, noise_distance) = batch

        # Fake loss
        fake_logits_list = []
        num_samples_per_seq = self.num_samples_per_seq or noise_time_seq.size()[1]

        for i in range(num_samples_per_seq):
            # [batch_size, seq_len, hidden_size]
            fake_logits_ = model.forward(noise_event_seq[:, i, :],
                                         noise_time_seq[:, i, :],
                                         batch_non_pad_mask,
                                         attention_mask)
            # [batch_size, hidden_size]
            fake_logits_ = model.get_logits_at_last_step(fake_logits_, batch_non_pad_mask)
            # [batch_size, 1]
            fake_logits_ = model.predict_as_discriminator(fake_logits_)

            fake_logits_list.append(fake_logits_)
        # shape -> [batch_size, num_samples, 1]
        fake_logits_list = torch.stack(fake_logits_list, dim=1)
        # shape -> [batch_size, num_samples]
        fake_distances = noise_distance[:, :num_samples_per_seq]

        return fake_logits_list, fake_distances

    def _top1_metric(self, real_logit, fake_logits, fake_distances):
        """ Top 1 accuracy metric.

        Args:
            real_logit: tensor with shape [batch_size, 2]
            fake_logits: tensor with shape [batch_size, num_samples, 2]
            fake_distances: tensor with shape [batch_size, num_samples]
                The Distance for each fake sample.

        Returns:
            num_real_correct: int
            num_real_seqs: int
            num_fake_correct: int
            num_fake_seqs: int
        """
        batch_size = real_logit.size()[0]
        # shape -> [batch_size]
        real_score = - real_logit[..., 0]
        # shape -> [batch_size, num_samples]
        fake_scores = - fake_logits[..., 0]

        # calc real correct number
        max_fake_score, _ = fake_scores.max(dim=-1)
        num_real_correct = (real_score > max_fake_score).sum().item()

        # calc fake correct number
        label_best_fake_idx = fake_distances.min(dim=-1)[1]
        best_fake_score = fake_scores[torch.arange(batch_size), label_best_fake_idx]
        num_fake_correct = (best_fake_score >= max_fake_score).sum().item()

        num_real_seqs = batch_size
        num_fake_seqs = batch_size

        return num_real_correct, num_real_seqs, num_fake_correct, num_fake_seqs
