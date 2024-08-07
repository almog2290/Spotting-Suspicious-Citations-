from model import GMAE_node
from data import GraphDataModule

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor 
from regression import generate_split, evaluate
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from fileUtils import saveToLocalFile , createModelTxTFiles
from graphUtils import prepare_data
from cosineSim import createGmaeVectors , calc_edges_restore


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    parser = GMAE_node.add_model_specific_args(parser)
    #parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    #dm = GraphDataModule.from_argparse_args(args)
    dm = GraphDataModule(
        dataset_name = args.dataset_name,
        num_workers=args.num_workers,
        batch_size =args.batch_size,
        seed=args.seed,
        l1=args.l1,
        l2=args.l2
    )
    n_node_features = dm.dataset.num_node_features

    # round accuracies for each threshold
    roundsAccuracies = []
    roundsAccuracies2 = []

    # generate split
    split = generate_split(dm.dataset.data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = GMAE_node.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_node_features=n_node_features,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            mask_ratio=args.mask_ratio,
            n_val_sampler=dm.n_val_sampler,
        )
    else:
        model = GMAE_node(
            n_node_features=n_node_features,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            mask_ratio=args.mask_ratio,
            n_val_sampler=dm.n_val_sampler,
        )
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'train_loss'
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=2,
        mode='min',
        save_last=True,
    )

    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)

    #trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(
        accelerator='auto',
        precision=str(args.precision)+'-mixed',
        #max_steps = args.max_steps,
        max_epochs=1000,
        default_root_dir=args.default_root_dir,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_epoch,
        enable_progress_bar=True,
        #fast_dev_run=True
    )
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
    prepare_data(dm)
    patience = 5
    earlystop_callback = EarlyStopping(
        monitor=metric,
        patience=patience,
        mode='min', # trainer will stop when the value stopped decreasing
        check_on_train_epoch_end=True,
    )
    #trainer.callbacks.append(earlystop_callback)
    
    
    if not args.test and not args.validate:
        rounds = 50
        for i in range(rounds):
            print('round ->',i+1)            
            print('Evaluating.....')
            if(i>0):
                model = GMAE_node.init_model(args,dm,n_node_features)
                trainer = pl.Trainer(
                    accelerator='auto',
                    precision=str(args.precision)+'-mixed',
                    #max_steps = args.max_steps,
                    max_epochs=1000,
                    default_root_dir=args.default_root_dir,
                    reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_epoch,
                    enable_progress_bar=True,
                    #fast_dev_run=True
                )
                trainer.callbacks.append(checkpoint_callback)
                trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

                # prepare data in each start of the round
                prepare_data(dm)
                patience = 5
                earlystop_callback = EarlyStopping(
                    monitor=metric,
                    patience=patience,   
                    mode='min', # trainer will stop when the value stopped decreasing
                    check_on_train_epoch_end=True,
                )
                #trainer.callbacks.append(earlystop_callback)

            trainer.fit(model, datamodule=dm)
            model.eval()
            
            #create game latent space vectors
            gameVectors=createGmaeVectors(model,dm)

            threshold = 0.95
            acc = calc_edges_restore (dm.dictTestEdges , gameVectors ,threshold , dm)

            threshold2 = 0.9
            acc2 = calc_edges_restore (dm.dictTestEdges , gameVectors,threshold2 , dm)
            

            #save each round accuracy
            roundsAccuracies.append(acc)
            roundsAccuracies2.append(acc2)
            
            #Save and Load the results of correct_edges_final
            resultsFileName='allUndirectedEdges.pkl'
            saveToLocalFile(resultsFileName,dm.allUndirectedEdges)

            #Save and Load the results of roundsAccuracies
            roundsAccuraciesFileName='roundsAccuracies_R={}_T={}.pkl'.format(rounds,threshold)
            saveToLocalFile(roundsAccuraciesFileName,roundsAccuracies)

            #Save and Load the results of roundsAccuracies2
            roundsAccuraciesFileName='roundsAccuracies_R={}_T={}.pkl'.format(rounds,threshold2)
            saveToLocalFile(roundsAccuraciesFileName,roundsAccuracies2)

        createModelTxTFiles(rounds , threshold , threshold2 , dm.allUndirectedEdges , roundsAccuracies , roundsAccuracies2)

        print('######### Final accuracies ####################')
        # Count average accuracy for each threshold
        avgAcc = sum(roundsAccuracies)/len(roundsAccuracies)
        print('avgAcc {} -> {}'.format(threshold, avgAcc))

        avgAcc2 = sum(roundsAccuracies2)/len(roundsAccuracies2)
        print('avgAcc {} -> {}'.format(threshold2, avgAcc2))
        
    if args.test or args.validate:
        model_path = dirpath + '/last.ckpt'
        print(model_path)
        model = type(model).load_from_checkpoint(model_path)
        print('Evaluating.....')
        acc = evaluate(model, dm, split)
        print('Acc:', acc)

if __name__ == '__main__':
    cli_main()


