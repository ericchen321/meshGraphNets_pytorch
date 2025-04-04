# -*- encoding: utf-8 -*-
'''
@File    :   parse_tfrecord.py
@Author  :   jianglx 
@Version :   1.0
@Contact :   jianglx@whu.edu.cn
'''
#解析tfrecord解析数据，存为hdf5文件
import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py

def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


if __name__ == '__main__':
    # tf.enable_resource_variables()
    # tf.enable_eager_execution()

    tf_datasetPath='data/flag_simple'

    # for split in ['train', 'test', 'valid']:
    for split in ['test', 'valid']:
        ds = load_dataset(tf_datasetPath, split)
        save_path='h5/'+ split  +'.h5'
        # f = h5py.File(save_path, "w")
        print(save_path)

        print(f"split: {split}")
        for index, d in enumerate(ds):
            mesh_pos = d['mesh_pos'].numpy()
            node_type = d['node_type'].numpy()
            # velocity = d['velocity'].numpy()
            cells = d['cells'].numpy()
            # pressure = d['pressure'].numpy()
            world_pos = d['world_pos'].numpy()
            # data = ("pos", "node_type", "velocity", "cells", "pressure")
            data = ("mesh_pos", "node_type", "cells", "world_pos")
            # g = f.create_group(str(index))
            # for k in data:
            #   g[k] = eval(k)
            
            # print(index)
            # check keys
            # if index == 0:
            #    for k, v in d.items():
            #        print(k, type(v))
            #        print(v.numpy().shape)
            if index == 0:
              # identify all unique node types in node_type[0, :]
              unique_node_types = np.unique(node_type[0, :])
              print(f"unique_node_types: {unique_node_types}")
              print(f"mesh_pos[0, 0]: {mesh_pos[0, 0]}")
              print(f"cells[0, 0]: {cells[0, 0]}")
        # f.close()