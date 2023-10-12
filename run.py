import argparse
import sys
import time
import traceback
import warnings

import numpy as np
from transformers import BertConfig, CLIPConfig, BertModel
from transformers.models.clip import CLIPProcessor

from models import *
from processor import *
from schedulers import *

sys.path.append("..")
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PROCESS_CLASS = {
    'bert-vit-inter-re': (MREProcessor, MREDataset),
}

MODEL_CLASS = {
    'bert-vit-inter-re': BertVitInterReModel,
}

DATA_PATH = {
    'mnre': {'train': 'data/mnre/txt/ours_train.txt',
             'dev': 'data/mnre/txt/ours_val.txt',
             'test': 'data/mnre/txt/ours_test.txt',
             'train_auximgs': 'data/mnre/txt/mre_train_dict.pth',  # {data_id : object_crop_img_path}
             'dev_auximgs': 'data/mnre/txt/mre_dev_dict.pth',
             'test_auximgs': 'data/mnre/txt/mre_test_dict.pth',
             'train_img2crop': 'data/mnre/img_detect/train/train_img2crop.pth',
             'dev_img2crop': 'data/mnre/img_detect/val/val_img2crop.pth',
             'test_img2crop': 'data/mnre/img_detect/test/test_img2crop.pth'},
}

IMG_PATH = {
    'mnre': {'train': 'data/mnre/img_org/train/',
             'dev': 'data/mnre/img_org/val/',
             'test': 'data/mnre/img_org/test'},
}

AUX_PATH = {
    'mnre': {
        'train': 'data/mnre/img_vg/train/crops',
        'dev': 'data/mnre/img_vg/val/crops',
        'test': 'data/mnre/img_vg/test/crops'
    },
}


def set_seed(seed=2022):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_logger(args):
    if args.do_test:
        args.experiment_name = args.load_path
        args.load_path = os.path.join(args.save_path, args.experiment_name)
        log_filename = "logs/" + args.experiment_name
        if args.write_path is not None:
            args.write_path = os.path.join(args.write_path, args.experiment_name)
    else:
        args.experiment_name = args.experiment_name + "_" + args.dataset_name + '_' + time.strftime('%Y_%m_%d_%H_%M_%S')
        log_filename = "logs/" + args.experiment_name
        args.save_path = os.path.join(args.save_path, args.experiment_name)
        if args.write_path is not None:
            args.write_path = os.path.join(args.write_path, args.experiment_name)
    print(args)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_filename,
                        level=logging.INFO)
    return logging.getLogger(__name__)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='test', type=str, help="The name of current experiment.")
    parser.add_argument('--model_name', default='mre', type=str, help="The name of bert.")
    parser.add_argument('--vit_name', default='vit', type=str, help="The name of vit.")
    parser.add_argument('--dataset_name', default='mnre', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base', type=str,
                        help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument('--num_epochs', default=20, type=int, help="Training epochs")
    parser.add_argument('--device', default='cpu', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--prompt_len', default=4, type=int)
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--aux_size', default=128, type=int, help="aux size")
    parser.add_argument('--rcnn_size', default=64, type=int, help="rcnn size")

    parser.add_argument('--log_mode', dest='log_mode', default='logger', help='The way of printing logs produced in '
                                                                              'training and testing procedure')
    parser.add_argument('--num_workers', default=10, type=int, help="number of process workers for dataloader")

    parser.add_argument('--ignore_idx', default=0, type=int)
    parser.add_argument('--crf_lr', default=5e-2, type=float, help="learning rate")
    parser.add_argument('--prompt_lr', default=3e-4, type=float, help="learning rate")

    return parser.parse_args()


def init_and_train_bert_vit_re(args, logger):
    data_process, dataset_class = DATA_PROCESS_CLASS[args.model_name]
    model_class = MODEL_CLASS[args.model_name]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.do_train:
        re_data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[
            args.dataset_name]
        re_path = 'data/' + args.dataset_name + '/ours_rel2id.json'

        clip_vit, clip_processor, aux_processor, rcnn_processor = None, None, None, None
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)  # args.vit_name="openai/clip-vit-base-patch32"
        aux_processor = CLIPProcessor.from_pretrained(args.vit_name)  # args.vit_name="openai/clip-vit-base-patch32"
        aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size  # aux_size=128
        rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size  # rcnn_size=64
        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model
        processor = data_process(re_data_path, re_path, args.bert_name, clip_processor=clip_processor,
                                 aux_processor=aux_processor, rcnn_processor=rcnn_processor)
        train_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size,
                                      rcnn_size=args.rcnn_size, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)

        dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size,
                                    rcnn_size=args.rcnn_size, mode='dev')
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True)

        test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size,
                                     rcnn_size=args.rcnn_size, mode='test', write_path=args.write_path,
                                     do_test=args.do_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

        re_dict = processor.get_relation_dict()
        tokenizer = processor.tokenizer  # 30526 tokens, be used to convert textual tokens to ids

        # start training, validating and testing
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)
        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()

        model = model_class(re_label_mapping=re_dict,
                            tokenizer=tokenizer,
                            args=args,
                            vision_config=vision_config,
                            text_config=text_config,
                            clip_model_dict=clip_model_dict,
                            bert_model_dict=text_model_dict, )

        trainer = BertVitReTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                                   re_dict=re_dict, model=model, args=args, logger=logger, writer=None)
        trainer.train()


if __name__ == "__main__":
    args = parse_argument()
    # set_seed(args.seed)  # set seed, default is 1
    logger = get_logger(args)
    logger.info(args)
    try:
        TRAINER = {
            'bert-vit-inter-re': init_and_train_bert_vit_re,
        }
        if args.model_name in TRAINER.keys():
            TRAINER[args.model_name](args, logger)
        else:
            raise Exception(f"The model {args.model_name} is not implemented!")
    except Exception:
        traceback.print_exc()
        logger.info(traceback.print_exc())
