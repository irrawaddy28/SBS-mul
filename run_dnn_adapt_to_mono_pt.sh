#!/bin/bash

# Copyright 2015-2016  University of Illinois (Author: Amit Das)
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

# /run_dnn_adapt_to_mono_pt.sh "SW" exp/tri3c_map/SW exp/tri3cpt_ali/SW exp/dnn4_pretrain-dbn_dnn/SW/indbn/final.nnet data-fmllr-tri3c_map/SW exp/dnn4_pretrain-dbn_dnn/SW/indbn_pt

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0 # resume training with --stage=N
feats_nj=4
train_nj=8
decode_nj=4
train_iters=20
l2_penalty=0
replace_softmax=true
# End of config.

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 [options] <lang code> <gmmdir> <precomputed dnn> <fmllr fea-dir> <nnet output dir>" 
   echo "e.g.: $0 --replace-softmax true SW exp/tri3b_map_SW_pt exp/pretrain-dnn/final.nnet data-fmllr-tri3b exp/dnn"
   echo ""
fi


TEST_LANG=$1     # "SW" (we want the DNN to fine-tune to this language using its PT's)
gmmdir=$2        # exp/tri3c/${TEST_LANG}
alidir=$3        # exp/tri3cpt_ali/${TEST_LANG}
precomp_dnn=$4   # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/indbn/final.nnet
data_fmllr=$5    # data-fmllr-tri3c/${TEST_LANG}
nnet_dir=$6      # exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/indbn_pt

for f in $gmmdir/final.mdl $alidir/post.*.gz $precomp_dnn; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

#if [ $stage -le 0 ]; then
  #graph_dir=$gmmdir/graph_oracle_LG      
  #[[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$lang/lang_test_oracle_LG $gmmdir $graph_dir >& $graph_dir/mkgraph.log; }  
#fi

if [ $stage -le 1 ]; then
  # Store fMLLR features, so we can train on them easily
    
  # eval
  for lang in ${TEST_LANG}; do
	dir=$data_fmllr/$lang/eval
	steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
		--transform-dir $gmmdir/decode_eval_$lang \
		$dir data/$lang/eval $gmmdir $dir/log $dir/data || exit 1
	steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
	utils/validate_data_dir.sh --no-text $dir	
  done
  
  # dev
  for lang in ${TEST_LANG}; do
   dir=$data_fmllr/$lang/dev
   steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev_$lang \
     $dir data/$lang/dev $gmmdir $dir/log $dir/data || exit 1
   steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
   utils/validate_data_dir.sh --no-text $dir  
  done
  
  # train
  for lang in ${TEST_LANG}; do
   dir=$data_fmllr/$lang/train
   steps/nnet/make_fmllr_feats.sh --nj $feats_nj --cmd "$train_cmd" \
     --transform-dir ${gmmdir} \
     $dir data/$lang/train $gmmdir $dir/log $dir/data || exit 1
   steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
   utils/validate_data_dir.sh --no-text $dir
  done
  
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

labels_trf="\"ark:gunzip -c ${alidir}/post.*.gz| post-to-pdf-post $alidir/final.mdl ark:- ark:- |\" "
labels_cvf=${labels_trf}
echo "soft-labels = ${labels_trf}"
if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy. 
  ali=$alidir
  feature_transform=  # calculate unsupervised feat xform based on the adaptation data in steps/nnet/train_pt.sh
  mkdir -p $nnet_dir
  nnet_init=$nnet_dir/nnet.init	  
  if [[ ${replace_softmax} == "true" ]]; then 
   perl local/nnet/renew_nnet_softmax.sh $gmmdir/final.mdl ${precomp_dnn} ${nnet_init}
  else
   cp ${precomp_dnn} ${nnet_init}
  fi
  echo "nnet_init = ${nnet_init}"
  
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  #Initialize NN training with the hidden layers of a DNN
  $cuda_cmd $nnet_dir/log/train_nnet.log \
  local/nnet/train_pt.sh --nnet-init ${nnet_init} --hid-layers 0 \
	--cmvn-opts "--norm-means=true --norm-vars=true" \
	--delta-opts "--delta-order=2" --splice 5 \
	--learn-rate 0.008 \
	--labels-trainf  ${labels_trf} \
	--labels-crossvf ${labels_cvf} \
	--copy-feats "false" \
	--train-iters ${train_iters} \
	--train-opts "--l2-penalty ${l2_penalty}" \
  $data_fmllr/${TEST_LANG}/train_tr90 $data_fmllr/${TEST_LANG}/train_cv10 dummy_lang $ali $ali $nnet_dir || exit 1; 
  echo "Done training nnet in: $nnet_dir"
fi

# Decode
if [ $stage -le 4 ]; then
  # Nnet decode:
  exp_dir=$gmmdir
  dir=$nnet_dir
  for L in ${TEST_LANG}; do
    echo "Decoding $L"    
    graph_dir=${exp_dir}/graph_text_G_$L
    [[ -d $graph_dir ]] || { mkdir -p $graph_dir; utils/mkgraph.sh data/$L/lang_test_text_G $exp_dir $graph_dir || exit 1; }
  
    (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
	  $graph_dir $data_fmllr/$L/dev $dir/decode_dev_text_G_$L || exit 1;) &
    (steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
      $graph_dir $data_fmllr/$L/eval $dir/decode_eval_text_G_$L || exit 1;) &     
    (cd $dir; ln -s  decode_dev_text_G_$L decode_dev_$L; ln -s decode_eval_text_G_$L decode_eval_$L)    
  done
fi

echo "Done: `date`"

exit 0;

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
