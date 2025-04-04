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
import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

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

    tf_datasetPath='/scratch-ssd/Repos/meshgraphnets/data/flag_simple'

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
            vertices = world_pos[0,:,[0,1]].transpose()
            faces = cells[0]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            
            # Extract unique edges
            edges = mesh.edges_unique

            # Convert edges into line segments for Matplotlib
            segments = [(vertices[e[0]], vertices[e[1]]) for e in edges]

            # Create plot
            fig, ax = plt.subplots(figsize=(6, 6))

            # Add edges as line collection
            line_collection = LineCollection(segments, colors='black', linewidths=0.8)
            ax.add_collection(line_collection)

            # Scatter plot for vertices
            sc = ax.scatter(vertices[:, 0], vertices[:, 1], color='red', s=10, zorder=2)

            # **Set mesh-space scaling**
            ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax.set_aspect('equal')  # **Ensures 1:1 aspect ratio**
            plt.title("Mesh in Mesh Space")
            plt.savefig("./fig.png")

            # **Function to update vertices each frame**
            def update(frame):
                global vertices

                # Update vertices (example: add small random movement)
                vertices = world_pos[frame,:,[0,1]].transpose()
                # Update scatter points
                segments = [(vertices[e[0]], vertices[e[1]]) for e in edges]
                line_collection.set_segments(segments)
                sc.set_offsets(vertices)

                return line_collection, sc

            # **Create animation**
            num_frames = len(mesh_pos)
            ani = FuncAnimation(fig, update, frames=num_frames, interval=100)

            # **Save as GIF**
            ani.save("./mesh_animation.gif", writer=PillowWriter(fps=10))
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