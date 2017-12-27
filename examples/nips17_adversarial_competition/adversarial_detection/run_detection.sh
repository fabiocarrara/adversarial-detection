#!/bin/bash

set -e
export TF_CPP_MIN_LOG_LEVEL=2

#for ATTACK in 'attacks_output/fgsm' 'attacks_output/noop' 'attacks_output/random_noise' 'targeted_attacks_output/step_target_class' 'targeted_attacks_output/iter_target_class'; do
#python classification_stats.py \
#  --input_dir="/code/intermediate_results/${ATTACK}" \
#  --input_truelabels="/code/output_dir/all_classification.csv" \
#  --output_file="classification_stats.csv" \
#  --checkpoint_path="inception_v3.ckpt" 
#done

export PYTHONPATH=/opt/faiss:$PYTHONPATH

RESULTS_DIR=detection_results
mkdir -p ${RESULTS_DIR}

## TODO rerun to extract avg_pool again.. 
## hacky code just for now
#cd search_db/avg_pool
#mkdir -p maxpool_5a_3x3 mixed_5b mixed_5c mixed_5d mixed_6a mixed_6b mixed_6c mixed_6d mixed_6e mixed_7a mixed_7b mixed_7c prelogits
#cd ../..
#cd search_db/max_pool
#mkdir -p maxpool_5a_3x3 mixed_5b mixed_5c mixed_5d mixed_6a mixed_6b mixed_6c mixed_6d mixed_6e mixed_7a mixed_7b mixed_7c prelogits
#cd ../..

#ALL_FEATS="MaxPool_5a_3x3,Mixed_5b,Mixed_5c,Mixed_5d,Mixed_6a,Mixed_6b,Mixed_6c,Mixed_6d,Mixed_6e,Mixed_7a,Mixed_7b,Mixed_7c,PreLogits"
#ALL_DB="search_db/avg_pool/maxpool_5a_3x3/database_lmdb,search_db/avg_pool/mixed_5b/database_lmdb,search_db/avg_pool/mixed_5c/database_lmdb,search_db/avg_pool/mixed_5d/database_lmdb,search_db/avg_pool/mixed_6a/database_lmdb,search_db/avg_pool/mixed_6b/database_lmdb,search_db/avg_pool/mixed_6c/database_lmdb,search_db/avg_pool/mixed_6d/database_lmdb,search_db/avg_pool/mixed_6e/database_lmdb,search_db/avg_pool/mixed_7a/database_lmdb,search_db/avg_pool/mixed_7b/database_lmdb,search_db/avg_pool/mixed_7c/database_lmdb,search_db/avg_pool/prelogits/database_lmdb,search_db/max_pool/maxpool_5a_3x3/database_lmdb,search_db/max_pool/mixed_5b/database_lmdb,search_db/max_pool/mixed_5c/database_lmdb,search_db/max_pool/mixed_5d/database_lmdb,search_db/max_pool/mixed_6a/database_lmdb,search_db/max_pool/mixed_6b/database_lmdb,search_db/max_pool/mixed_6c/database_lmdb,search_db/max_pool/mixed_6d/database_lmdb,search_db/max_pool/mixed_6e/database_lmdb,search_db/max_pool/mixed_7a/database_lmdb,search_db/max_pool/mixed_7b/database_lmdb,search_db/max_pool/mixed_7c/database_lmdb,search_db/max_pool/prelogits/database_lmdb"
#python build_database.py --input_dir /ImageNet/train --feature_pool "avg_pool,max_pool" --feature_layer "${ALL_FEATS}" --output_database "${ALL_DB}"
## end hacky code

for AGGREGATION in 'max_pool' 'avg_pool'; do
for FEATURE in 'MaxPool_5a_3x3' 'Mixed_5b' 'Mixed_5c' 'Mixed_5d' 'Mixed_6a' 'Mixed_6b' 'Mixed_6c' 'Mixed_6d' 'Mixed_6e' 'Mixed_7a' 'Mixed_7b' 'Mixed_7c' 'PreLogits'; do
  # 0. CREATE DB_DIR
  FEAT_CODE=${FEATURE,,} # to lowecase
  DB_DIR="search_db/${AGGREGATION}/${FEAT_CODE}"
  mkdir -p ${DB_DIR}
  
  # 1. BUILD INDEX if not already built
  if [ ! -f "${DB_DIR}/database.fidx" ]; then
    # 2. EXTRACT FEATURES if not already extracted
    if [ ! -d "${DB_DIR}/database_lmdb" ]; then 
      python build_database.py --input_dir /ImageNet/train --feature_pool ${AGGREGATION} --feature_layer ${FEATURE} --output_database ${DB_DIR}/database_lmdb
    fi
    python build_index.py ${DB_DIR}/database_lmdb ${DB_DIR}/database.fidx --train_list search_db/index_train_list.txt
  fi
  
  # 3. SCORE ADVERSARIALS
  for KNN_SCORING in 'knn' 'wknn' 'dwknn' 'lwknn' 'ldwknn'; do
    SCORES_FILE="${RESULTS_DIR}/scores_${AGGREGATION}_${FEAT_CODE}_${KNN_SCORING}.csv"
    if [ -f ${SCORES_FILE} ]; then
      echo "SKIPPING: FEAT=${FEAT_CODE} AGGR=${AGGREGATION} KNN_SCORING=${KNN_SCORING}"
    else
      echo "RUNNING: FEAT=${FEAT_CODE} AGGR=${AGGREGATION} KNN_SCORING=${KNN_SCORING}"
      for ATTACK in 'attacks_output/fgsm' 'attacks_output/noop' 'attacks_output/random_noise' 'targeted_attacks_output/step_target_class' 'targeted_attacks_output/iter_target_class' ; do
        python detection.py \
          --input_dir="/code/intermediate_results/${ATTACK}" \
          --output_file="${RESULTS_DIR}/scores_${AGGREGATION}_${FEAT_CODE}_${KNN_SCORING}.csv" \
          --checkpoint_path="inception_v3.ckpt" \
          --feature_layer="${FEATURE}" \
          --features_database="${DB_DIR}/database.fidx" \
          --labels_database="search_db/database_index_labels.npy" \
          --k_neighbors=1000 \
          --knn_scoring="${KNN_SCORING}"
      done
    fi
    # python evaluate.py -f ${FEATURE} -s ${KNN_SCORING} -p PCA-256 ${SCORES_FILE} /code/output_dir/all_classification.csv
  done
done
done
