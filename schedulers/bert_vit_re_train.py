import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report as re_cls_report
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd

from utilities.metrics import eval_result


class BertVitReTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None,
                 model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.re_dict = re_dict

        self.model = model
        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.final_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_dev_epoch = None
        self.best_test_epoch = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.pbar = None
        self.re_optimizer = None
        self.re_scheduler = None
        self.before_train()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        if self.args.do_test:
            self.logger.info("***** Start testing without training *****")
            self.test(0)
            return

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            re_avg_loss = 0, 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (re_loss, re_logits), labels, _ = self._step(re_batch,
                                                                 mode="train",
                                                                 task='re',
                                                                 epoch=epoch)
                    re_avg_loss += re_loss.detach().cpu().item()
                    re_loss.backward()
                    self.re_optimizer.step()
                    self.re_optimizer.zero_grad()
                    self.re_scheduler.step()

                    if self.step % self.refresh_step == 0:
                        re_avg_loss = float(re_avg_loss) / self.refresh_step
                        print_output = "RE loss:{:<6.5f}".format(re_avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        re_avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
                    self.test(epoch)

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, "
                             "best dev f1 is {}".format(self.best_dev_epoch,
                                                        self.best_dev_metrics['micro_f1'],
                                                        ))
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 is {}".format(self.best_test_epoch,
                                            self.best_test_metrics['micro_f1'],
                                            ))
            self.logger.info(
                "Get final test performance according to validation results at epoch {}, "
                "final f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_dev_epoch,
                    self.final_test_metrics['micro_f1'],
                    self.final_test_metrics['micro_r'],
                    self.final_test_metrics['micro_p'],
                    self.final_test_metrics['acc']))
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_test_epoch,
                    self.best_test_metrics['micro_f1'],
                    self.best_test_metrics['micro_r'],
                    self.best_test_metrics['micro_p'],
                    self.best_test_metrics['acc']))

    def evaluate(self, epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                step = 0
                re_true_labels, re_pred_labels = [], []
                total_loss = 0
                hits = torch.zeros([len(self.re_dict), len(self.re_dict) + 1], device=self.args.device)
                for batch in self.dev_data:
                    step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels, _ = self._step(re_batch,
                                                           mode="dev",
                                                           task='re',
                                                           epoch=epoch,)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    re_preds = logits.argmax(-1)
                    re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    re_pred_labels.extend(re_preds.view(-1).detach().cpu().tolist())
                    re_pred_ranks = 1 + torch.argsort(torch.argsort(logits, dim=1, descending=True), dim=1,
                                                      descending=False)[
                        torch.arange(labels.shape[0], device=self.args.device), labels]
                    re_pred_ranks = re_pred_ranks.float()
                    for rel_id in range(len(self.re_dict)):
                        ranks = re_pred_ranks[labels == rel_id]
                        for k in range(len(self.re_dict)):
                            hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + hits[rel_id, k + 1]
                    pbar.update()
                # evaluate done
                pbar.close()
                re_cls_result = re_cls_report(y_true=re_true_labels, y_pred=re_pred_labels,
                                              labels=list(self.re_dict.values())[1:],
                                              target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", re_cls_result)
                result = eval_result(re_true_labels, re_pred_labels, self.re_dict, self.logger)

                self.logger.info(
                    "Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}." \
                        .format(epoch, self.args.num_epochs, self.best_dev_metrics['micro_f1'],
                                self.best_dev_epoch,
                                result['micro_f1'], ))
                if result['micro_f1'] >= self.best_dev_metrics['micro_f1']:  # this epoch get best performance
                    self.logger.info("Get better dev performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metrics['micro_f1'] = result['micro_f1']  # update best metric
                    self.best_dev_metrics['micro_r'] = result['micro_r']
                    self.best_dev_metrics['micro_p'] = result['micro_p']
                    self.best_dev_metrics['acc'] = result['acc']
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path)
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch=0):
        self.model.eval()
        self.logger.info(f"\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        re_true_labels, re_pred_labels, sample_word_lists, sample_image_ids = [], [], [], []
        re_pred_logits = []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                hits = torch.zeros([len(self.re_dict), len(self.re_dict) + 1], device=self.args.device)
                for batch in self.test_data:
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                                batch)  # to cpu/cuda device
                    if self.args.write_path is not None and self.args.do_test:
                        (loss, logits), labels, extend_word_lists, imgids = self._step(re_batch,
                                                                                       mode="test",
                                                                                       task='re',
                                                                                       epoch=epoch,)
                    else:
                        (loss, logits), labels, _ = self._step(re_batch,
                                                               mode="test",
                                                               task='re',
                                                               epoch=epoch,)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    re_pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    re_pred_logits.extend(logits.detach().cpu().tolist())
                    re_pred_ranks = 1 + torch.argsort(torch.argsort(logits, dim=1, descending=True), dim=1,
                                                      descending=False)[
                        torch.arange(labels.shape[0], device=self.args.device), labels]
                    re_pred_ranks = re_pred_ranks.float()
                    for rel_id in range(len(self.re_dict)):
                        ranks = re_pred_ranks[labels == rel_id]
                        for k in range(len(self.re_dict)):
                            hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + hits[rel_id, k + 1]
                    if self.args.write_path is not None and self.args.do_test:
                        sample_word_lists.extend([*extend_word_lists])
                        sample_image_ids.extend([*imgids])
                    pbar.update()
                # evaluate done
                pbar.close()
                if self.args.write_path is not None and self.args.do_test:
                    # dictionary of lists
                    write_file_dict = {'sample_word_lists': sample_word_lists, 'sample_image_ids': sample_image_ids,
                                       'true_labels': re_true_labels, 'pred_labels': re_pred_labels,
                                       'pred_logits': re_pred_logits}
                    df = pd.DataFrame(write_file_dict)
                    # saving the dataframe
                    df.to_csv(self.args.write_path + '_' + 'test.csv')
                sk_result = re_cls_report(y_true=re_true_labels, y_pred=re_pred_labels,
                                          labels=list(self.re_dict.values()),
                                          target_names=list(self.re_dict.keys()), digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(re_true_labels, re_pred_labels, self.re_dict, self.logger)
                ############
                self.logger.info(
                    "Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, " \
                        .format(epoch, self.args.num_epochs,
                                self.best_test_metrics['micro_f1'],
                                self.best_test_epoch,
                                result['micro_f1'], ))

                if epoch == self.best_dev_epoch:
                    if result['micro_f1'] > self.final_test_metrics['micro_f1']:
                        self.final_test_metrics['micro_f1'] = result['micro_f1']  # update best metric
                        self.final_test_metrics['micro_r'] = result['micro_r']
                        self.final_test_metrics['micro_p'] = result['micro_p']
                        self.final_test_metrics['acc'] = result['acc']

                if result['micro_f1'] >= self.best_test_metrics['micro_f1']:  # this epoch get best performance
                    self.logger.info("Get better test performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metrics['micro_f1'] = result['micro_f1']  # update best metric
                    self.best_test_metrics['micro_r'] = result['micro_r']
                    self.best_test_metrics['micro_p'] = result['micro_p']
                    self.best_test_metrics['acc'] = result['acc']

        self.model.train()

    def _step(self, batch, mode="train", task='re', epoch=0):
        if self.args.write_path is not None and mode == 'test' and self.args.do_test:
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, extend_word_lists, imgids = batch
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=aux_imgs,
                                 rcnn_imgs=rcnn_imgs,
                                 task=task,
                                 epoch=epoch,)
            return outputs, labels, extend_word_lists, imgids
        else:
            re_input_ids, re_token_type_ids, re_attention_mask, re_labels, images, aux_imgs, rcnn_imgs = batch
            if task == 're':
                input_ids = re_input_ids
                token_type_ids = re_token_type_ids
                attention_mask = re_attention_mask
                labels = re_labels
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=aux_imgs,
                                 rcnn_imgs=rcnn_imgs,
                                 task=task,
                                 epoch=epoch,)
            return outputs, labels, attention_mask

    def before_train(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'model' in name or name.startswith('re_classifier'):
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)
        self.re_optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.re_scheduler = get_linear_schedule_with_warmup(optimizer=self.re_optimizer,
                                                            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                            num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)