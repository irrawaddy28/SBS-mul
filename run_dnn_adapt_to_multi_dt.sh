#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script creates a multilingual nnet. The training is done in 3 stages:
# 1. FMLLR features: It generates fmllr features from the multilingual training data.
# 2. DBN Pre-training: To initialize the nnet, it can 
#    a) train a dbn using the multilingual fmllr features or
#    b) use an existing pre-trained dbn or dnn from the user
# 3. DNN cross-entropy training: It fine-tunes the initialized nnet using 
#    the multilingual training data (deterministic transcripts).
#

# Usage: $0 --precomp-dnn "exp/dnn4e-fmllr_multisoftmax/final.nnet" "AR CA HG MD UR" "SW"
# Usage: $0 --precomp-dbn "exp/dnn4_pretrain-dbn/6.dbn" "AR CA HG MD UR" "SW"

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
precomp_dbn=
precomp_dnn= 
train_iters=20
use_delta=false
# End of config.

. utils/parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [ $# != 7 ]; then
	echo "usage: $0 <train lang> <test lang> <gmmdir> <alidir> <data_fmllr> <nnetinitdir> <nnetoutdir>"	
fi

TRAIN_LANG=$1
TEST_LANG=$2
gmmdir=$3       # exp/tri3c/${TEST_LANG}
alidir=$4       # exp/tri3c_ali/${TEST_LANG}
data_fmllr=$5   # data-fmllr-tri3c/${TEST_LANG}
nnetinitdir=$6
nnetoutdir=$7

UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')
[[ ! -z ${precomp_dnn} ]] && use_dbn=false || use_dbn=true 
$use_dbn && echo "Using a pre-trained DBN to init target DNN" || echo "Using pre-trained DNN to init target DNN"

#echo ==========================
#if [ $stage -le 0 ]; then
#steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
  #data/${UNILANG_CODE}/train data/${UNILANG_CODE}/lang $gmmdir ${alidir} 2>&1 | tee ${alidir}/align.log
#fi
#echo ==========================

if [ $stage -le 1 ]; then
  # Store fMLLR features, so we can train on them easily
    
  # eval
  for lang in ${TRAIN_LANG} ${TEST_LANG}; do
	dir=$data_fmllr/$lang/eval
	steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
		--transform-dir $gmmdir/decode_eval_$lang \
		$dir data/$lang/eval $gmmdir $dir/log $dir/data || exit 1
	steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
	utils/validate_data_dir.sh --no-text $dir
  done
  
  # dev
  for lang in ${TRAIN_LANG} ${TEST_LANG}; do
    dir=$data_fmllr/$lang/dev
    steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
       --transform-dir $gmmdir/decode_dev_$lang \
       $dir data/$lang/dev $gmmdir $dir/log $dir/data || exit 1
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    utils/validate_data_dir.sh --no-text $dir
  done
  
  # train
  dir=$data_fmllr/${UNILANG_CODE}/train
  steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $alidir \
     $dir data/${UNILANG_CODE}/train $alidir $dir/log $dir/data || exit 1
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  utils/validate_data_dir.sh --no-text $dir 
  
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

#if [[ ! -z ${precomp_dbn} ]]; then
#	nnetinitdir=exp/dnn4_pretrain-dbn/${TEST_LANG}/outdbn  #out-of-domain dbn (dbn learned from a corpus different than target corpus)
#elif [[ ! -z ${precomp_dnn} ]]; then
#	nnetinitdir=exp/dnn4_pretrain-dbn/${TEST_LANG}/outdnn  #out-of-domain dnn (dnn learned from a corpus different than  target corpus)
#else
#	nnetinitdir=exp/dnn4_pretrain-dbn/${TEST_LANG}/indbn #in-domain dbn (dbn learned from this target corpus)
#fi
if [ $stage -le 2 ]; then
# First check for pre-computed DBN dir. Then try pre-computed DNN dir. If both fail, generate DBN now.
  mkdir -p $nnetinitdir
  if [[ ! -z ${precomp_dbn} ]]; then
	echo "using pre-computed dbn ${precomp_dbn}"	
	
	# copy the dbn and feat xform from dbn dir	
	cp -r ${precomp_dbn} $nnetinitdir 
	
	# Comment lines below if you want to compute feature xform estmn from the adaptation data (SBS)
	#cp $(dirname ${precomp_dbn})/final.feature_transform $dir
	#feature_transform=$dir/final.feature_transform
	#feature_transform_opt=$(echo "--feature-transform $feature_transform")  
  elif [[ ! -z ${precomp_dnn} ]]; then
	echo "using pre-computed dnn ${precomp_dnn}"	
	
	# replace the softmax layer of the precomp dnn with a random init layer
	nnet_init=$nnetinitdir/nnet.init
	rm -rf ${nnet_init}
	perl local/utils/nnet/renew_nnet_softmax.sh $gmmdir/final.mdl ${precomp_dnn} ${nnet_init}
	
	# Comment lines below if you want to compute feature xform estmn from the adaptation data (SBS)	
	#cp $(dirname ${precomp_dnn})/final.feature_transform $dir  
	#feature_transform=$dir/final.feature_transform
	#feature_transform_opt=$(echo "--feature-transform $feature_transform")
  else
    echo "train with a randomly initialized DBN"
    
	# Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)	
	(tail --pid=$$ -F $nnetinitdir/log/pretrain_dbn.log 2>/dev/null)& # forward log
	$cuda_cmd $nnetinitdir/log/pretrain_dbn.log \
		steps/nnet/pretrain_dbn.sh --nn-depth 6 --hid-dim 1024 \
		--cmvn-opts "--norm-means=true --norm-vars=true" \
		--delta-opts "--delta-order=2" --splice 5 \
		--rbm-iter 20 $data_fmllr/${UNILANG_CODE}/train $nnetinitdir || exit 1;  
  fi
fi


dir=$nnetoutdir
#if [[ ! -z ${precomp_dbn} ]]; then
#	dir=exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/outdbn  #out-of-domain dbn (dbn learned from a corpus different than target corpus)
#elif [[ ! -z ${precomp_dnn} ]]; then
#	dir=exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/outdnn  #out-of-domain dnn (dnn learned from a corpus different than  target corpus)
#else
#	dir=exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/indbn #in-domain dbn (dbn learned from this target corpus)
#fi
if [ $stage -le 3 ]; then
  # Train the DNN optimizing per-frame cross-entropy.  
  ali=$alidir
  feature_transform=
  #dir=${nnetinitdir}_dnn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log    
  # Train
  if ${use_dbn}; then
  # Initialize NN training with a DBN
  dbn=${nnetinitdir}/6.dbn
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh  --dbn $dbn --hid-layers 0 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --learn-rate 0.008 \
    $data_fmllr/${UNILANG_CODE}/train_tr90 $data_fmllr/${UNILANG_CODE}/train_cv10 data/${UNILANG_CODE}/lang $ali $ali $dir || exit 1;
  else
  nnet_init=${nnetinitdir}/nnet.init
  # Initialize NN training with the hidden layers of a DNN
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --nnet-init ${nnet_init} --hid-layers 0 \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta-opts "--delta-order=2" --splice 5 \
    --learn-rate 0.008 \
    $data_fmllr/${UNILANG_CODE}/train_tr90 $data_fmllr/${UNILANG_CODE}/train_cv10 data/${UNILANG_CODE}/lang $ali $ali $dir || exit 1;
  fi
fi


if [ $stage -le 4 ]; then
  # Nnet decode:
  exp_dir=$gmmdir
  for L in ${TRAIN_LANG} ${TEST_LANG}; do
    echo "Decoding $L"
    
    graph_dir=${exp_dir}/graph_text_G_$L
    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_text_G $exp_dir $graph_dir || exit 1; }
  
    (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
	  $graph_dir $data_fmllr/$L/dev $dir/decode_dev_text_G_$L || exit 1;) &
    (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
      $graph_dir $data_fmllr/$L/eval $dir/decode_eval_text_G_$L || exit 1;) &
      
    (cd $dir; ln -s  decode_dev_text_G_$L decode_dev_$L; ln -s decode_eval_text_G_$L decode_eval_$L)
  done
  wait
fi

echo "Done: `date`"
exit 0;

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
