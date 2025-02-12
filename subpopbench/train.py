import argparse
import collections
import json
import os
import random
import sys
import time
import scipy
import numpy as np
import pandas as pd
import PIL
import pickle
from sklearn.preprocessing import MinMaxScaler
import copy
import torch
import torchvision
import torch.utils.data
from tensorboard_logger import Logger

from subpopbench import hparams_registry
from subpopbench.dataset import datasets
from subpopbench.learning import algorithms, early_stopping
from subpopbench.utils import misc, eval_helper
from subpopbench.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subpopulation Shift Benchmark')
    # training
    parser.add_argument('--dataset', type=str, default="Waterbirds", choices=datasets.DATASETS)
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    parser.add_argument('--output_folder_name', type=str, default='debug')
    parser.add_argument('--train_attr', type=str, default="yes", choices=['yes', 'no'])
    # others
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--tb_log_all', action='store_true')
    # two-stage related
    parser.add_argument('--stage1_folder', type=str, default='vanilla')
    parser.add_argument('--stage1_algo', type=str, default='ERM')
    # early stopping
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--es_strategy', choices=['metric'], default='metric')
    parser.add_argument('--es_metric', type=str, default='min_group:accuracy')
    parser.add_argument('--es_patience', type=int, default=5, help='Stop after this many checkpoints w/ no improvement')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    # CMNIST data params
    parser.add_argument('--cmnist_label_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_attr_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_spur_prob', type=float, default=0.2)
    parser.add_argument('--cmnist_flip_prob', type=float, default=0.25)
    # CMNISTV2 data params
    parser.add_argument('--cmnistv2_difficult', type=str, default='5pct')
    # architectures and pre-training sources
    parser.add_argument('--image_arch', default='resnet_sup_in1k',
                        choices=['resnet_sup_in1k', 'resnet_sup_in21k', 'resnet_simclr_in1k', 'resnet_barlow_in1k',
                                 'vit_sup_in1k', 'vit_sup_in21k', 'vit_clip_oai', 'vit_clip_laion', 'vit_sup_swag',
                                 'vit_dino_in1k', 'resnet_dino_in1k'])
    parser.add_argument('--text_arch', default='bert-base-uncased',
                        choices=['bert-base-uncased', 'gpt2', 'xlm-roberta-base',
                                 'allenai/scibert_scivocab_uncased', 'distilbert-base-uncased'])
    # proposed model config
    parser.add_argument('--gen_weights', action='store_true', help='for generating sample level weights') 
    parser.add_argument('--switch_train_valid', action='store_true', help='decide if switch train and valid, only useful for genreating weights and finetune with the proposed model') 
    parser.add_argument('--sample_weights_path', type=str, default='', help='path to load sample weights')
    parser.add_argument('--weighted_sampling', action='store_true', help='if specified, we apply weights on sampling data for training')
    parser.add_argument('--attr2weights', action='store_true', help='if specified, the dataloader will return float values for self.a')
    parser.add_argument('--tau_valid', type=float, help='calibration value during final stage validation', default=1.0)
    parser.add_argument('--tau_train', type=float, help='calibration value during final stage trianing', default=1.0)
    parser.add_argument('--sample_weights_path_valid', type=str, default='', help='path to load sample weights for validation set')
    parser.add_argument('--sample_weights_path_valid_test', type=str, default='', help='path to load sample weights for validation set, used to match test')
    parser.add_argument('--p_maj', type=float, help='p_maj ratio', default=.99)
    
    
    args = parser.parse_args()
    
    if args.sample_weights_path_valid != '':
        assert args.weighted_sampling == False, 'the proposed method cannot have this'
    
    # override attr2weights, if we pass weights during training, then we use weights for attributes
    args.attr2weights = args.sample_weights_path != '' or args.sample_weights_path_valid != ''

    start_step = 0
    store_prefix = f"{args.dataset}_{args.cmnist_label_prob}_{args.cmnist_attr_prob}_{args.cmnist_spur_prob}" \
                   f"_{args.cmnist_flip_prob}" if args.dataset == "CMNIST" else args.dataset
    args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    args.output_folder_name += "_attrYes" if args.train_attr == 'yes' else "_attrNo"

    misc.prepare_folders(args)
    args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.dataset == "CMNIST":
        hparams.update({'cmnist_label_prob': args.cmnist_attr_prob,
                        'cmnist_attr_prob': args.cmnist_attr_prob,
                        'cmnist_spur_prob': args.cmnist_spur_prob,
                        'cmnist_flip_prob': args.cmnist_flip_prob})
    if args.dataset == 'CMNISTV2':
        hparams.update({'cmnistv2_difficult': args.cmnistv2_difficult})
        
    if args.dataset in ['CMNISTV2', 'CivilCommentsFine', 'Waterbirds', 'CelebA', 'MultiNLI']: # Other datasets might require changes to work with the propsoed method
        hparams.update({'attr2weights': args.attr2weights})

    hparams.update({
        'image_arch': args.image_arch,
        'text_arch': args.text_arch
    })

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in vars(datasets):
        if args.dataset == 'CMNISTV2' and args.cmnistv2_difficult == '0.5pct':
            hparams.update({
                            'lr': 0.0001 # 1e-4 only for proposed
                            })

        # a dataset switch between train and valid
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'tr' if not args.switch_train_valid else 'va', hparams, train_attr=args.train_attr) # this is alwasys the actual trianing (swtich or not)
        
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'va' if not args.switch_train_valid else 'tr', hparams) # this is always the actual validation (switch or not)
        
        
        if args.switch_train_valid:
            print('PAY ATTENTION!!! train and valid datasets are switched on purpose.')
            
            
        if args.sample_weights_path != '': # load sample_weights_file and need to check if len(self.a) == len(self.idx), always point to dataset for training (whether or not switched), if switched, then valid is used here, otherwise is train
            assert args.sample_weights_path != '', 'need to specify sample_weights_path for weighted sampling'
            assert args.sample_weights_path_valid != '', 'need to specify sample_weights_path_valid for proper weights generation'
            
            if args.dataset in ['CMNISTV2', 'CelebA', 'Waterbirds', 'CivilCommentsFine']:     
                
                s_maj_dict = misc.calculate_label_ratios(train_dataset.a)
                y_maj_dict = misc.calculate_label_ratios(train_dataset.y)
                y_maj_dict_val = misc.calculate_label_ratios(val_dataset.y)
                p_y_adjustment_train = np.asarray(list(map(lambda y: y_maj_dict.get(y), train_dataset.y)))
                p_y_adjustment_val = np.asarray(list(map(lambda y: y_maj_dict_val.get(y), train_dataset.y)))
                
                if args.dataset == 'CMNISTV2':
                    K = L = 10
                else:
                    K = L = 2
                    if args.dataset == 'CivilCommentsFine': # override
                        L = 8
                ttt = np.abs(np.load(args.sample_weights_path) - np.load(args.sample_weights_path_valid))

                if args.dataset == 'Waterbirds':
                    diff_train_val_p_y_x = misc.obtain_proper_sample_weights(np.load(args.sample_weights_path), 
                                                                args.tau_train, adjustment=0.)
                else:
                    ### p_y adjustment*: empirically observed good performance but can remove both p_y_adjustment_train and p_y_adjustment_val depending on practical observations
                    diff_train_val_p_y_x = misc.obtain_proper_sample_weights(np.abs(np.load(args.sample_weights_path)*p_y_adjustment_train - np.load(args.sample_weights_path_valid)*p_y_adjustment_val), args.tau_train, adjustment=0.)
                
                # print conditional probabilty between y and s, this is only for checking the dataset statistics, irrelavant to training
                misc.calculate_conditional_probabilities(np.asarray(train_dataset.y),
                                                          np.asarray(train_dataset.a))
                
                if args.train_attr == 'yes':
                    b = misc.g_fn_v3(None, train_dataset.y, y_maj_dict, p_maj=args.p_maj, K=K, L=L, ss=train_dataset.a)
                else:
                    if args.dataset == 'Waterbirds':
                        b = misc.g_fn_v2_waterbirds(diff_train_val_p_y_x, train_dataset.y, y_maj_dict, p_maj=args.p_maj, K=K, L=L)
                    elif args.dataset in ['CMNISTV2', 'CivilCommentsFine']:
                        b = misc.g_fn_v2_multiplier(diff_train_val_p_y_x, train_dataset.y, y_maj_dict, p_maj=args.p_maj, K=K, L=L, multiplier=10000.)

            else:
                assert args.sample_weights_path_valid != '', 'need to specify this for proper training'
                b = misc.obtain_proper_sample_weights(np.load(args.sample_weights_path_valid), args.tau_train) / misc.obtain_proper_sample_weights(np.load(args.sample_weights_path), args.tau_train)

            train_dataset.a = b
            print('OVERRIDING self.a for the training set as sample_weights are offered.')
            print('make sure you are in the finetune mode. Otherwise this weights should not be used')
            assert len(train_dataset.a) == len(train_dataset.x), 'the size of the weights self.a should match the size of the data'
            
        if args.sample_weights_path_valid_test != '':
            assert args.sample_weights_path_valid_test != '', 'need to specify sample_weights_path_valid for weighted sampling'
            if args.dataset in ['Waterbirds', 'CMNISTV2', 'CelebA', 'CivilCommentsFine']:
                g = np.array([1.] * len(val_dataset.x))
            else:
                g = 1. / len(val_dataset.x) / misc.obtain_proper_sample_weights(np.load(args.sample_weights_path_valid_test), args.tau_valid)
            val_dataset.a = g
            print('OVERRIDING self.a for the validation set as sample_weights are offered.')
            print('make sure you are in the finetune mode. Otherwise this weights should not be used')
            assert len(val_dataset.a) == len(val_dataset.x), 'the size of the weights self.a should match the size of the data'
        
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)
        
        #### uncommon below to print conditional probability for only the reference purposes
#         misc.calculate_conditional_probabilities(np.asarray(test_dataset.y),
#                                                           np.asarray(test_dataset.a))
    else:
        raise NotImplementedError
    

    if args.algorithm == 'DFR':
        train_dataset = vars(datasets)[args.dataset](
            args.data_dir, 'va', hparams, train_attr=args.train_attr, subsample_type='group')

    num_workers = 0 #train_dataset.N_WORKERS #check if infinite loader has an issue
    input_shape = train_dataset.INPUT_SHAPE
    num_labels = train_dataset.num_labels
    num_attributes = train_dataset.num_attributes
    data_type = train_dataset.data_type
    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ

    hparams.update({
        "steps": n_steps
    })
    
    print(f"Dataset:\n\t[train]\t{len(train_dataset)} (with{'' if args.train_attr == 'yes' else 'out'} attributes)"
          f"\n\t[val]\t{len(val_dataset)}\n\t[test]\t{len(test_dataset)}")

    if hparams['group_balanced']:
        # if attribute not available, groups degenerate to classes
        train_weights = np.asarray(train_dataset.weights_g)
        train_weights /= np.sum(train_weights)
    elif args.sample_weights_path != '' and args.weighted_sampling: # this can only happend to the actual training part, for validation, we nevery need this weighted sampling as we need to consider all samples in the validation set (regardless of switch or not)
        # 2. apply softmax
        print('assign training weights from sample_weights_path for sampling')
        train_weights = train_dataset.a
    else:
        train_weights = None

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=train_weights, 
        batch_size=hparams['batch_size'],
        num_workers=num_workers
    )


    split_names =  ['tr'] + ['va'] + vars(datasets)[args.dataset].EVAL_SPLITS 
    
    tmp_datasets_for_eval = []
    for split_idx, split in enumerate(split_names):
        if split in ['tr', 'va']:
            tmp_dataset = vars(datasets)[args.dataset](args.data_dir, split, hparams)
            if not args.switch_train_valid:
                if split == 'tr':
                    tmp_dataset.a = train_dataset.a
                elif split == 'va':
                    tmp_dataset.a = val_dataset.a
            else: 
                if split == 'tr':
                    tmp_dataset.a = val_dataset.a
                elif split == 'va':
                    tmp_dataset.a = train_dataset.a
            assert len(tmp_dataset.a) == len(tmp_dataset.x), 'something migth be wrong with valid train concepts'
            tmp_datasets_for_eval.append(copy.deepcopy(tmp_dataset))
        else: # including test reinit as hparams might changed, also other datasets go here.
            tmp_datasets_for_eval.append(vars(datasets)[args.dataset](args.data_dir, split, hparams))
            

            
    eval_loaders = [FastDataLoader(
        dataset=dset,
        batch_size=max(64, hparams['batch_size'] * 2),
        num_workers=num_workers)
        for dset in tmp_datasets_for_eval
    ]
    
    if args.switch_train_valid: # we still need to switch to masure tr and va matches the notion of actual training and validation
        assert 'tr' in split_names and 'va' in split_names # only allow switch when checking both tr and va
        # switch two eval_loaders so eval_train correponds 'va' in split names and vice versa
        assert split_names[0] == 'tr' and split_names[1] == 'va'
        eval_loaders[0], eval_loaders[1] = eval_loaders[1], eval_loaders[0]
        
    
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(data_type, input_shape, num_labels, num_attributes,
                                len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes)

    es_group = args.es_metric.split(':')[0]
    es_metric = args.es_metric.split(':')[1]
    es = early_stopping.EarlyStopping(
        patience=args.es_patience, lower_is_better=early_stopping.lower_is_better[es_metric])
    best_model_path = os.path.join(args.output_dir, 'model.best.pkl')

    # load stage1 model if using 2-stage algorithm
    if 'CRT' in args.algorithm or 'DFR' in args.algorithm:
        args.pretrained = os.path.join(
            args.output_dir.replace(args.output_folder_name, args.stage1_folder), hparams['stage1_model']
        ).replace(args.algorithm, args.stage1_algo)
        args.pretrained = args.pretrained.replace(
            f"seed{args.pretrained[args.pretrained.find('seed') + len('seed')]}", 'seed0')
        assert os.path.isfile(args.pretrained)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_dict'].items():
            if 'classifier' not in k and 'network.1.' not in k:
                new_state_dict[k] = v
        algorithm.load_state_dict(new_state_dict, strict=False)
        print(f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]")
        print(f"===> Pre-trained model loaded: '{args.pretrained}'")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['start_step']
            args.best_val_acc = checkpoint['best_val_acc']
            algorithm.load_state_dict(checkpoint['model_dict'])
            es = checkpoint['early_stopper']
            print(f"===> Loaded checkpoint '{args.resume}' (step [{start_step}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    algorithm.to(device)

    train_minibatches_iterator = iter(train_loader)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = len(train_dataset) / hparams['batch_size']

    def save_checkpoint(save_dict, filename='model.pkl'):
        if args.skip_model_save:
            return
        filename = os.path.join(args.output_dir, filename)
        torch.save(save_dict, filename)

    last_results_keys = None
    for step in range(start_step, n_steps):
        if args.use_es and es.early_stop:
            print(f"Early stopping at step {step} with best {args.es_metric}={es.best_score}.")
            break
        step_start_time = time.time()

        # new for finite order
        try:
            # Try to fetch the next batch
            i, x, y, a = next(train_minibatches_iterator)
        except StopIteration:
            # If DataLoader is exhausted, reinitialize the iterator
            train_minibatches_iterator = iter(train_loader)
            i, x, y, a = next(train_minibatches_iterator)
            
        minibatch_device = (i, x.to(device), y.to(device), a.to(device))

        algorithm.train()
        if args.steps != 1: # meaning we only run valid without training so we skip
            step_vals = algorithm.update(minibatch_device, step)
            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)
        else:
            print('no model is updated as script is used for pure validation or gen_weights for the loaded pretrain models')
            
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            curr_metrics = {split: eval_helper.eval_metrics(algorithm, loader, device,
                                                            gen_weights=args.gen_weights and split == 'va', # only when gen_weights and split == ['va'] we gen weights
                                                           with_weights=(split in ['va', 'va_2', 'tr_2'] and args.sample_weights_path_valid_test != '')) 
                            for split, loader in zip(split_names, eval_loaders)}            
            full_val_metrics = copy.deepcopy(curr_metrics['va'])
            
            # delete this as this is not a scalar, may introduce error in the latter part
            if args.gen_weights:
                del curr_metrics['va']['aligned_weights']
                

            for split in sorted(split_names):
                results[f'{split}_avg_acc'] = curr_metrics[split]['overall']['accuracy']
                results[f'{split}_worst_acc'] = curr_metrics[split]['min_group']['accuracy']   

            results_keys = list(results.keys())
            if results_keys != last_results_keys:
                print("\n")
                misc.print_row([key for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=12)
            
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results.update({
                'hparams': hparams,
                'args': vars(args),
            })
            results.update(curr_metrics)

            epochs_path = os.path.join(args.output_dir, 'results.json')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            save_dict = {
                "args": vars(args),
                "best_es_metric": es.best_score,
                "start_step": step + 1,
                "num_labels": num_labels,
                "num_attributes": train_dataset.num_attributes,
                "model_input_shape": input_shape,
                "model_hparams": hparams,
                "model_dict": algorithm.state_dict(),
                "early_stopper": es,
                "aligned_weights": full_val_metrics['aligned_weights'] if args.gen_weights else None, #(new, also save weights when saving the model)
            }
            
            save_checkpoint(save_dict)
            if args.steps == 1:
                print('checkpoint saved')
                print(len(full_val_metrics['aligned_weights']))
            
            # delete this as this is not a scalar, may introduce error in the latter part
            if args.gen_weights:
                del full_val_metrics['aligned_weights']
                
            

            # tensorboard logger
            for key in checkpoint_vals.keys() - {'step_time'}:
                tb_logger.log_value(key, results[key], step)
            for key in split_names:
                tb_logger.log_value(f"{key}_avg_acc", results[f"{key}_avg_acc"], step)
                tb_logger.log_value(f"{key}_worst_acc", results[f"{key}_worst_acc"], step)
            if args.tb_log_all:
                for key1 in full_val_metrics:
                    for key2 in full_val_metrics[key1]:
                        if isinstance(full_val_metrics[key1][key2], dict):
                            for key3 in full_val_metrics[key1][key2]:
                                tb_logger.log_value(f"{key1}_{key2}_{key3}", full_val_metrics[key1][key2][key3], step)
                        else:
                            tb_logger.log_value(f"{key1}_{key2}", full_val_metrics[key1][key2], step)
            if hasattr(algorithm, 'optimizer'):
                tb_logger.log_value('learning_rate', algorithm.optimizer.param_groups[0]['lr'], step)

            if args.use_es:
                if args.es_strategy == 'metric':
                    es_metric_val = full_val_metrics[es_group][es_metric]

                es(es_metric_val, step, save_dict, best_model_path)
                tb_logger.log_value('es_metric', es_metric_val, step)

            checkpoint_vals = collections.defaultdict(lambda: [])

    # load best model and get metrics on eval sets
    if args.use_es and not args.skip_model_save:
        algorithm.load_state_dict(torch.load(os.path.join(args.output_dir, "model.best.pkl"))['model_dict'])

    algorithm.eval()
    # if regular train valid, then we check valid, otherwise we check 'tr' as tr becomes valid after switch
    split_names = ['va' if not args.switch_train_valid else 'tr'] + vars(datasets)[args.dataset].EVAL_SPLITS
    
    del eval_loaders # save memory
    
    final_eval_loaders = [FastDataLoader(
        dataset=dset,
        batch_size=max(32, hparams['batch_size'] * 2),
        num_workers=num_workers)
        for dset in [vars(datasets)[args.dataset](args.data_dir, split, hparams) for split in split_names]
    ]
    final_results = {split: eval_helper.eval_metrics(algorithm, loader, device)
                     for split, loader in zip(split_names, final_eval_loaders)}
    pickle.dump(final_results, open(os.path.join(args.output_dir, 'final_results.pkl'), 'wb'))

    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{final_results['te']['overall']['accuracy']:.3f}]\n"
          f"\tworst:\t[{final_results['te']['min_group']['accuracy']:.3f}]")
    print("Group-wise accuracy:")
    for split in final_results.keys():
        print('\t[{}] group-wise {}'.format(
            split, (np.array2string(
                pd.DataFrame(final_results[split]['per_group']).T['accuracy'].values,
                separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
