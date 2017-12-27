import argparse
import faiss
import lmdb

import numpy as np

from tqdm import tqdm


def main(args):
  fidx = None
  env = lmdb.open(args.lmdb_database)
  
  if args.train_list:
    with env.begin() as txn:
      with open(args.train_list, 'rb') as train_list:
        xt = []
        for k in train_list:
          v = txn.get(k.rstrip())
          x = np.fromstring(v, dtype=np.float32).reshape(1, -1)
          xt.append(x)
    xt = np.vstack(xt)
    print xt.shape
    if fidx is None:
      dim = xt.shape[1]
      pca_size = int(args.index.split(',')[0][3:]) if 'PCA' in args.index else 0
      if dim < pca_size:
        args.index = 'Flat'
      fidx = faiss.index_factory(dim, args.index)
    fidx.train(xt)
    del xt
  
  with env.begin() as txn:
    for k, v in tqdm(txn.cursor(), total=env.stat()['entries']):
      x = np.fromstring(v, dtype=np.float32).reshape(1, -1)
      if fidx is None:
        fidx = faiss.index_factory(x.shape[1], args.index)
      fidx.add(x)
      
  faiss.write_index(fidx, args.faiss_index)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Create FAISS index from LMDB storage')
  parser.add_argument('lmdb_database', type=str, help='Location of the LMDB database')
  parser.add_argument('faiss_index', type=str, help='Location of the FAISS index to create')
  parser.add_argument('-i', '--index', type=str, default='PCA256,Flat', help='Index type (see FAISS index_factory())')
  parser.add_argument('-t', '--train_list', type=str, help='File containing IDs of features to be used to train the index (for PCA)')
  
  args = parser.parse_args()
  main(args)


